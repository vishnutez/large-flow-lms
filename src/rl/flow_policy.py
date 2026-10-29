"""Flow decoding as a stochastic policy over denoising steps.

``rollout`` reproduces ``FlowModelWrapper.generate`` with the per-position SDE
sampler (``sde_gamma > 0``, ``sde_vp=True``) and records every network input.
With the network input z_in as the state, each step after the first is a
Gaussian transition

    z_in[k+1] ~ N(alpha[k+1] * euler_k(z_in[k]), (coef[k+1] * sigma)^2 I)

whose mean depends on the model only through the denoiser output at step k.

``packed_step_means`` recomputes those means for any set of steps in one
forward pass over [prompt + response | noisy block groups]. Each group sees
the clean tokens before its block and itself, which is what the block saw
through the KV cache during the rollout, so the means match the rollout's up
to numerics. The clean positions also give the AR head's logits.

Supported: per-position time schedules, ``sde_vp``, no prefix guidance
(cfg_scale=1), with or without self-conditioning.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from src.lafa.cache import trim_kv_cache
from src.lafa.flow_model import _cut_at_stop, _stop_ids


@dataclass
class BlockRecord:
    """Everything needed to recompute one block's transitions."""

    start: int  # absolute position of the block's first slot
    times: Tensor  # [K, B] conditioning time per step (alpha * grid time)
    next_times: Tensor  # [K, B] grid time each Euler step moves to
    alpha: Tensor  # [K, B] rewind factor per step (1 where no noise)
    coef: Tensor  # [K, B] noise std per step, in units of sigma
    z_in: Tensor  # [K, B, d] network input per step, before self-conditioning
    self_cond: Tensor | None  # [K, B, d] self-conditioning vector per step
    means: Tensor  # [K-1, B, d] float32 transition means from the rollout
    tokens: list[int] = field(default_factory=list)


@dataclass
class Rollout:
    prompt_ids: Tensor  # [P]
    response_ids: Tensor  # [L]
    blocks: list[BlockRecord]
    sigma: float  # noise scale: model._noise_sigma * model.noise_scale_mult
    num_forwards: int


def _check_supported(model):
    if model._two_time or model.logit_shift or model.noisy_causal:
        raise ValueError("RL rollouts need a single-time, unshifted, block-bidirectional model")
    if not getattr(model, "time_warp_per_position", False):
        raise ValueError("RL rollouts need per-position time schedules")


def _sde_coefficients(grid, step, gamma):
    """Rewind factor, noise std (units of sigma) and conditioning time at a step."""
    t_val, t_out = grid[:, step], grid[:, step + 1]
    alpha = (1.0 - gamma * (t_out - t_val)).clamp(0.0, 1.0)
    alpha = torch.where(t_val > 0.0, alpha, torch.ones_like(alpha))
    coef = torch.sqrt(((1.0 - alpha) * (1.0 + alpha - 2.0 * alpha * t_val)).clamp(min=0.0))
    return alpha, coef, alpha * t_val, t_out


def _euler(z, x_hat, t_from, t_to):
    """Float32 Euler step toward the predicted clean embedding, per slot."""
    tf = t_from[..., None].float()
    to = t_to[..., None].float()
    return z.float() + (to - tf) * (x_hat.float() - z.float()) / (1.0 - tf).clamp(min=1e-6)


@torch.no_grad()
def rollout(model, prompt_ids, *, max_new_tokens, num_steps, sde_gamma, seed):
    """Decode one response with the SDE sampler, recording every step.

    Matches ``model.generate(..., grid_mode='warp', sde_gamma=sde_gamma,
    sde_vp=True, cfg_scale=1, seed=seed)`` token for token.
    """
    _check_supported(model)
    device = next(model.backbone.parameters()).device
    dtype = model.embed_tokens.weight.dtype
    E = model.embed_tokens.weight
    B = model.bd_size
    sigma = model._noise_sigma * model.noise_scale_mult
    torch.manual_seed(seed)

    prompt_ids = prompt_ids.reshape(1, -1).to(device)
    prompt_len = prompt_ids.shape[1]
    eos_ids = _stop_ids(model.backbone.config.eos_token_id)
    grid = model._get_generation_time_grid(num_steps, device, grid_mode="warp")  # [B, K+1]

    orig_attn = getattr(model.backbone.config, "_attn_implementation", None)
    model.backbone.config._attn_implementation = "eager"
    gen_ids = prompt_ids[0].tolist()
    blocks, num_forwards = [], 0
    try:
        ones = torch.ones(1, prompt_len, device=device, dtype=dtype)
        _, past, _ = model._forward_backbone(
            model.embed_tokens(prompt_ids), model._compute_t_bn(ones),
            attn_mask=None, use_cache=True)

        while len(gen_ids) - prompt_len < max_new_tokens:
            start = len(gen_ids)
            pos = torch.arange(start, start + B, device=device).unsqueeze(0)
            cache_len = past.get_seq_length()
            mask = torch.zeros(1, 1, B, cache_len + B, dtype=dtype, device=device)

            def denoise(z, t):
                logits, _, _ = model._forward_backbone(
                    z, model._compute_t_bn(t.to(dtype)[None, :]), mask,
                    past_key_values=past, use_cache=False, position_ids=pos)
                trim_kv_cache(past, past.get_seq_length() - cache_len)
                return logits[0]

            # Scale noise exactly as the decoder does, so bf16 rounding matches.
            e_t = (torch.randn(1, B, model._d_model, device=device, dtype=dtype)
                   * model._noise_sigma * model.noise_scale_mult)
            sc = None
            if model.enable_self_cond:
                probs = F.softmax(denoise(e_t, grid[:, 0]).float(), -1).to(dtype)
                sc = (probs @ E).unsqueeze(0)
                num_forwards += 1

            rec = dict(times=[], next_times=[], alpha=[], coef=[], z_in=[], self_cond=[], means=[])
            euler_f32 = None
            for step in range(num_steps):
                alpha, coef, t_fwd, t_out = _sde_coefficients(grid, step, sde_gamma)
                # The decoder only draws step noise when the SDE is on.
                noise = (torch.randn_like(e_t) * model._noise_sigma * model.noise_scale_mult
                         if sde_gamma > 0 else torch.zeros_like(e_t))
                if step > 0:
                    # Transition mean: the previous Euler step, rewound. Float32,
                    # with the same formula packed_step_means uses.
                    rec["means"].append(alpha[None, :, None].float() * euler_f32)
                z_in = alpha.to(dtype)[None, :, None] * e_t + coef.to(dtype)[None, :, None] * noise
                inp = model._apply_self_cond(z_in, sc) if model.enable_self_cond else z_in
                logits = denoise(inp, t_fwd)
                num_forwards += 1
                for key, value in (("times", t_fwd), ("next_times", t_out), ("alpha", alpha),
                                   ("coef", coef), ("z_in", z_in[0]),
                                   ("self_cond", sc[0] if sc is not None else None)):
                    rec[key].append(value)
                probs = F.softmax(logits.float(), dim=-1).to(dtype)
                if model.enable_self_cond:
                    sc = (probs @ E).unsqueeze(0)
                if step < num_steps - 1:
                    # The trajectory follows the decoder's bf16 path; the recorded
                    # mean uses float32 so it isolates forward-pass differences.
                    e_t = _euler(z_in, (probs @ E).unsqueeze(0), t_fwd, t_out).to(dtype)
                    x_f32 = F.softmax(logits.float(), dim=-1) @ E.float()
                    euler_f32 = _euler(z_in, x_f32.unsqueeze(0), t_fwd, t_out)

            remaining = max_new_tokens - (len(gen_ids) - prompt_len)
            new_tokens, hit_eos = _cut_at_stop(logits[:remaining].float().argmax(-1).tolist(), eos_ids)
            blocks.append(BlockRecord(
                start=start,
                times=torch.stack(rec["times"]), next_times=torch.stack(rec["next_times"]),
                alpha=torch.stack(rec["alpha"]), coef=torch.stack(rec["coef"]),
                z_in=torch.stack(rec["z_in"]),
                self_cond=torch.stack(rec["self_cond"]) if model.enable_self_cond else None,
                means=(torch.cat(rec["means"]) if rec["means"]
                       else torch.zeros(0, B, model._d_model, device=device)),
                tokens=new_tokens))
            gen_ids.extend(new_tokens)
            if hit_eos or len(gen_ids) - prompt_len >= max_new_tokens:
                break

            ids = torch.tensor([new_tokens], device=device)
            commit_pos = torch.arange(start, start + len(new_tokens), device=device).unsqueeze(0)
            ones = torch.ones(1, len(new_tokens), device=device, dtype=dtype)
            _, past, _ = model._forward_backbone(
                model.embed_tokens(ids), model._compute_t_bn(ones), attn_mask=None,
                past_key_values=past, use_cache=True, position_ids=commit_pos)
    finally:
        if orig_attn is not None:
            model.backbone.config._attn_implementation = orig_attn

    return Rollout(
        prompt_ids=prompt_ids[0].cpu(),
        response_ids=torch.tensor(gen_ids[prompt_len:], dtype=torch.long),
        blocks=blocks, sigma=sigma, num_forwards=num_forwards)


def packed_attention_mask(seq_len, groups, dtype, device):
    """Additive mask for [clean (seq_len) | groups], each group (start, size).

    Clean positions are causal. A group sees clean positions before its start
    and every slot of its own group.
    """
    total = seq_len + sum(size for _, size in groups)
    mask = torch.full((total, total), float("-inf"), dtype=dtype, device=device)
    mask[:seq_len, :seq_len] = torch.triu(mask[:seq_len, :seq_len], diagonal=1)
    row = seq_len
    for start, size in groups:
        mask[row:row + size, :start] = 0.0
        mask[row:row + size, row:row + size] = 0.0
        row += size
    return mask


def packed_step_means(model, ro, steps):
    """Recompute transition means for ``steps`` of every block in one forward.

    Step k's denoiser output gives the mean of the transition into step k+1,
    so valid steps are 0 .. K-2. Returns (means [len(steps), num_blocks, B, d]
    float32, clean logits [P + L, V] for the AR head). Gradients flow when
    called outside ``torch.no_grad``.
    """
    _check_supported(model)
    device = next(model.backbone.parameters()).device
    dtype = model.embed_tokens.weight.dtype
    E = model.embed_tokens.weight
    B = model.bd_size
    clean_ids = torch.cat([ro.prompt_ids, ro.response_ids]).to(device)
    seq_len = clean_ids.numel()

    groups, z, sc, times, positions = [], [], [], [], []
    for k in steps:
        for blk in ro.blocks:
            groups.append((blk.start, B))
            z.append(blk.z_in[k])
            times.append(blk.times[k])
            positions.append(torch.arange(blk.start, blk.start + B, device=device))
            if blk.self_cond is not None:
                sc.append(blk.self_cond[k])
    d = model._d_model
    z = torch.stack(z).to(device) if z else torch.zeros(0, B, d, device=device, dtype=dtype)
    noisy = model._apply_self_cond(z, torch.stack(sc).to(device)) if sc else z
    embeds = torch.cat([model.embed_tokens(clean_ids), noisy.reshape(-1, d)])
    t = torch.cat([torch.ones(seq_len, device=device), *[ti.to(device) for ti in times]])
    position_ids = torch.cat([torch.arange(seq_len, device=device), *positions])
    mask = packed_attention_mask(seq_len, groups, dtype, device)

    orig_attn = getattr(model.backbone.config, "_attn_implementation", None)
    model.backbone.config._attn_implementation = "eager"
    try:
        logits, _, _ = model._forward_backbone(
            embeds[None], model._compute_t_bn(t.to(dtype)[None]), mask[None, None],
            use_cache=False, position_ids=position_ids[None])
    finally:
        if orig_attn is not None:
            model.backbone.config._attn_implementation = orig_attn

    noisy_logits = logits[0, seq_len:].reshape(len(groups), B, logits.shape[-1])
    x_hat = F.softmax(noisy_logits.float(), dim=-1) @ E.float()
    means = []
    for i, (k, blk) in enumerate((k, blk) for k in steps for blk in ro.blocks):
        step = _euler(z[i], x_hat[i], blk.times[k].to(device), blk.next_times[k].to(device))
        means.append(blk.alpha[k + 1].to(device)[:, None].float() * step)
    means = (torch.stack(means) if means else x_hat.new_zeros(0, B, d)).reshape(
        len(steps), len(ro.blocks), B, d)
    return means, logits[0, :seq_len]


def transition_logprob(ro, step, means, per_dim=False):
    """Gaussian log-prob of each block's recorded z_in[step+1] under ``means``.

    ``means`` is [num_blocks, B, d]. Returns [num_blocks] in float64; the
    normalizing constant, which cancels in ratios, is dropped (it is ~1e5 and
    would swamp float32). Slots with zero noise are skipped. ``per_dim``
    averages over the noisy coordinates instead of summing (Flow-GRPO's
    convention, which keeps the term's scale independent of B and d).
    """
    out = []
    for blk, mean in zip(ro.blocks, means):
        std = (blk.coef[step + 1] * ro.sigma).to(mean.device).double()  # [B]
        live = std > 0
        diff = blk.z_in[step + 1].to(mean.device).double() - mean.double()
        logp = -((diff[live] ** 2).sum(-1) / (2.0 * std[live] ** 2)).sum()
        if per_dim:
            logp = logp / max(int(live.sum()) * diff.shape[-1], 1)
        out.append(logp)
    return torch.stack(out)
