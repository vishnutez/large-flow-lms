"""Lafa-NFT: forward-process RL for Lafa (DiffusionNFT / NFT).

Rollouts come from any sampler (the deployed ODE decoder); only their tokens
and rewards are used. The update is Lafa's own training loss on the rollouts,
noised with the training forward process, with each term contrasted between
an implicit positive and negative policy built from the trained model θ and
an old model θ_old (an EMA of θ):

    token term    p⁺ = (1-β) p_old + β p_θ,  p⁻ = (1+β) p_old - β p_θ
    embedding     x̂⁺ = (1-β) x̂_old + β x̂_θ,  x̂⁻ = (1+β) x̂_old - β x̂_θ   (x̂ = pE)
    AR head       NFT: positive -log R, negative -log max((1 - r̂ R)/(1 - r̂), ε)

Positives (reward 1) fit the ⁺ policy to their tokens, negatives (reward 0)
fit the ⁻ policy. NFT's negative policy is the β = r̂/(1 - r̂) case of the ⁻
form. Lafa's velocity is affine in x̂ at fixed z_t and t, so mixing x̂ is
DiffusionNFT's velocity mixing. See our_work/research-notes.md, "Alternative:
forward-process RL".
"""

import math
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from src.rl.lora import has_lora, merge_lora


def st_clamp_min(x: Tensor, floor) -> Tensor:
    """max(x, floor) in the forward pass, the gradient of x in the backward
    pass (NFT's straight-through clip).

    Written as clamp + (x - x.detach()) so the forward value is exact: the
    form x + (clamp - x).detach() cancels catastrophically in float32 when
    x << -1e7 (it returned 0 for x ≈ -1e8, so log gave inf/NaN; NFT v2 step 184).
    """
    return torch.clamp(x, min=floor).detach() + (x - x.detach())


def build_sequence(prompt_ids: Tensor, response_ids: Tensor, block_size: int, pad_id: int):
    """[prompt + response] padded to a multiple of the block size, with labels
    marking the response (prompt and padding are -100), as Lafa trains."""
    ids = torch.cat([prompt_ids, response_ids]).long()
    pad = (-ids.numel()) % block_size
    labels = ids.clone()
    labels[:prompt_ids.numel()] = -100
    if pad:
        ids = torch.cat([ids, ids.new_full((pad,), pad_id)])
        labels = torch.cat([labels, labels.new_full((pad,), -100)])
    return ids[None], labels[None]


@dataclass
class NoisedInput:
    """One draw of Lafa's training forward process, shared by θ and θ_old."""

    e_1: Tensor  # [1, L, d] clean embeddings
    e_t: Tensor  # [1, L, d] noised embeddings, before self-conditioning
    e_prev: Tensor | None  # [1, L, d] self-conditioning prediction, or None
    t_bn: Tensor
    position_ids: Tensor
    attn_mask: Tensor
    cond_mask: Tensor  # [1, L] 1 = response (noised), 0 = prompt/pad
    flow_valid: Tensor  # [1, L] response slots in complete response blocks


@torch.no_grad()
def noised_input(model_old, input_ids, labels, *, noise_scale, self_cond_prob):
    """Noise a sequence as Lafa's training ``forward`` does: per-position times
    from the time warper, noise at ``noise_scale`` x sigma_E, packed attention
    mask. Self-conditioning (with probability ``self_cond_prob``) predicts
    from θ_old so both models see the same input."""
    model = model_old
    cond_mask = model._answer_corruption_mask(labels)
    saved = model.noise_scale_mult
    model.noise_scale_mult = noise_scale
    try:
        e_1, e_t, t_bn, position_ids, attn_mask, _ = model._prepare_flow_input(input_ids, cond_mask)
    finally:
        model.noise_scale_mult = saved
    e_prev = None
    if model.enable_self_cond and torch.rand(()) < self_cond_prob:
        logits, _, _ = model._forward_backbone(
            torch.cat([e_1, e_t], dim=1), t_bn, attn_mask, use_cache=False,
            position_ids=position_ids)
        probs = F.softmax(logits[:, e_t.shape[1]:].float(), dim=-1).to(e_1.dtype)
        e_prev = probs @ model.embed_tokens.weight
    flow_valid = model._full_block_loss_mask(labels != -100)
    return NoisedInput(e_1, e_t, e_prev, t_bn, position_ids, attn_mask, cond_mask, flow_valid)


def packed_logits(model, inp: NoisedInput) -> Tensor:
    """Logits [1, 2L, V] of ``model`` on the shared noised input; the
    self-conditioning layers are the model's own."""
    e_t = inp.e_t
    if inp.e_prev is not None:
        e_t = model._apply_self_cond(e_t, inp.e_prev)
        m = inp.cond_mask.unsqueeze(-1).to(e_t.dtype)
        e_t = m * e_t + (1.0 - m) * inp.e_1
    logits, _, _ = model._forward_backbone(
        torch.cat([inp.e_1, e_t], dim=1), inp.t_bn, inp.attn_mask, use_cache=False,
        position_ids=inp.position_ids)
    return logits


def logit_mix(z, z_old, beta, positive):
    """log softmax of the logit mixture: (1 - β) z_old + β z for the positive
    policy, (1 + β) z_old - β z for the negative (geometric mixing of p_θ and
    p_old). Both are proper distributions for any β, so no ratio or clip is
    needed and the gradient in z is ±β (onehot - p), bounded by √2 β."""
    delta = beta * (z - z_old)
    return F.log_softmax(z_old + delta if positive else z_old - delta, dim=-1)


def logit_mix_logprob(z, z_old, target, beta, positive):
    """``logit_mix`` at ``target``."""
    return logit_mix(z, z_old, beta, positive).gather(-1, target[:, None])[:, 0]


def ppo_trust(loss, log_q, positive, ppo_clip, dual_clip):
    """PPO's trust region on a per-token loss: no gradient where q = p_θ/p_old
    already moved past 1 ± ``ppo_clip`` in the branch's direction (q > 1 + δ
    for a positive, q < 1 - δ for a negative), nor where a negative's q
    exceeds ``dual_clip`` (dual-clip PPO; there the gradient grows like q).
    Either bound is off at 0. Forward values are unchanged.
    Returns (loss, mask of clipped tokens)."""
    q = log_q.detach().exp()
    keep = torch.ones_like(q, dtype=torch.bool)
    if ppo_clip > 0.0:
        keep &= q <= 1.0 + ppo_clip if positive else q >= 1.0 - ppo_clip
    if dual_clip > 0.0 and not positive:
        keep &= q <= dual_clip
    return torch.where(keep, loss, loss.detach()), ~keep


def nft_terms(logits, old_logits, input_ids, labels, flow_valid, embed, *, positive,
              accuracy, beta, eps, adaptive=False, mix="prob", ppo_clip=0.0, dual_clip=0.0):
    """Summed (not averaged) loss terms for one sequence, and their counts.

    ``accuracy`` is r̂ of the sequence's question group (0 < r̂ < 1).
    ``adaptive`` uses DiffusionNFT's self-normalized x0 regression for the
    embedding term: per-dimension mean, divided by sg(mean |x̂ - E[x]|) over the
    sequence's tokens and dimensions (default: squared distance).
    ``mix`` builds the token-level ⁺/⁻ policies (token CE and AR) from
    probabilities (``"prob"``, NFT; the negative needs the ε clip) or from
    logits (``"logit"``, ``logit_mix_logprob``). With either, the embedding
    term mixes x̂ linearly (DiffusionNFT's form). ``"geo"`` is ``"logit"``
    with the embedding term taken from the same mixed distribution,
    x̂± = softmax(mixed logits) E, so CE and MSE share one p± (see
    our_work/geometric-mixing.md).
    ``ppo_clip`` > 0 or ``dual_clip`` > 0 adds ``ppo_trust`` to the token CE
    and AR terms; their clip stats then count the tokens it stopped (not the
    ε floor).
    Returns dict(token, embed, ar, n_flow, n_ar, ...stats).
    """
    L = input_ids.shape[1]
    ids = input_ids[0]
    valid = flow_valid[0]
    x = ids[valid]
    lp = F.log_softmax(logits[0, L:][valid].float(), dim=-1)
    lp_old = F.log_softmax(old_logits[0, L:][valid].float(), dim=-1)
    lp_x = lp.gather(-1, x[:, None])[:, 0]
    lpo_x = lp_old.gather(-1, x[:, None])[:, 0]

    if mix not in ("prob", "logit", "geo"):
        raise ValueError(f"unknown mix {mix!r}")
    lp_mix = None  # geo: log p± over the vocabulary, shared by CE and MSE
    # Denoiser, token level: -log p⁺(x) or -log max(p⁻(x), eps * p_old(x)).
    if mix in ("logit", "geo") and not (positive and beta == 1.0):
        lp_mix = logit_mix(logits[0, L:][valid].float(), old_logits[0, L:][valid].float(),
                           beta, positive)
        token = -lp_mix.gather(-1, x[:, None])[:, 0]
        token_clip = torch.zeros_like(lp_x)
    elif positive:
        if beta == 1.0:
            token = -lp_x
        elif 0.0 < beta < 1.0:
            token = -torch.logaddexp(lpo_x + math.log1p(-beta), lp_x + math.log(beta))
        else:
            raise ValueError("the positive mixture needs 0 < beta <= 1")
        token_clip = torch.zeros_like(lp_x)
    else:
        ratio = (1.0 + beta) - beta * torch.exp(lp_x - lpo_x)  # p⁻(x) / p_old(x)
        token = -(lpo_x + torch.log(st_clamp_min(ratio, eps)))
        token_clip = (ratio < eps).float()

    if ppo_clip > 0.0 or dual_clip > 0.0:
        token, token_clip = ppo_trust(token, lp_x - lpo_x, positive, ppo_clip, dual_clip)
        token_clip = token_clip.float()

    # Denoiser, embedding level (DiffusionNFT's flow-matching contrast on x̂ = pE).
    E = embed.float()
    if mix == "geo":
        x_mix = (lp if lp_mix is None else lp_mix).exp() @ E
    else:
        x_hat, x_hat_old = lp.exp() @ E, lp_old.exp() @ E
        x_mix = ((1.0 - beta) * x_hat_old + beta * x_hat if positive
                 else (1.0 + beta) * x_hat_old - beta * x_hat)
    diff = x_mix - E[x]
    if adaptive:
        scale = diff.detach().abs().mean().clamp_min(1e-5) if diff.numel() else 1.0
        emb = (diff ** 2).mean(-1) / scale
    else:
        emb = (diff ** 2).sum(-1)

    # AR head, NFT: clean position n predicts token n+1.
    ar_valid = labels[0, 1:] != -100
    y = ids[1:][ar_valid]
    lp_ar = F.log_softmax(logits[0, :L - 1][ar_valid].float(), dim=-1).gather(-1, y[:, None])[:, 0]
    lpo_ar = F.log_softmax(old_logits[0, :L - 1][ar_valid].float(), dim=-1).gather(
        -1, y[:, None])[:, 0]
    if positive:
        ar = -lp_ar
        ar_clip = torch.zeros_like(lp_ar)
    elif mix in ("logit", "geo"):
        ar = -logit_mix_logprob(logits[0, :L - 1][ar_valid].float(),
                                old_logits[0, :L - 1][ar_valid].float(), y,
                                accuracy / (1.0 - accuracy), positive=False)
        ar_clip = torch.zeros_like(lp_ar)
    else:
        ratio = (1.0 - accuracy * torch.exp(lp_ar - lpo_ar)) / (1.0 - accuracy)
        ar = -torch.log(st_clamp_min(ratio, eps))
        ar_clip = (ratio < eps).float()

    if ppo_clip > 0.0 or dual_clip > 0.0:
        ar, ar_clip = ppo_trust(ar, lp_ar - lpo_ar, positive, ppo_clip, dual_clip)
        ar_clip = ar_clip.float()

    with torch.no_grad():
        log_ratio = torch.cat([lp_x - lpo_x, lp_ar - lpo_ar])
    return dict(token=token.sum(), embed=emb.sum(), ar=ar.sum(),
                n_flow=int(valid.sum()), n_ar=int(ar_valid.sum()),
                token_clip=float(token_clip.sum()), ar_clip=float(ar_clip.sum()),
                log_ratio_abs_max=float(log_ratio.abs().max()) if log_ratio.numel() else 0.0)


class OldPolicyEMA:
    """θ_old as a float32 EMA of the float32 masters, sharded like the
    optimizer, mirrored into a bf16 model for the no-grad forward.

    A bf16 EMA cannot absorb 1e-6 updates, hence float32.

    With LoRA, the bf16 copy also holds the merged weights (``merge_lora``),
    refreshed after every update: θ_old only decodes and scores, and merged
    it runs one matmul per adapted layer instead of three.
    """

    def __init__(self, optimizer, policy_model, old_model):
        names = {id(p): n for n, p in policy_model.named_parameters()}
        self.opt = optimizer
        self.old_model = old_model
        self.old_params = [old_model.get_parameter(names[id(p)]) for p in optimizer.params]
        self.ema = {i: m.detach().clone() for i, m in optimizer.masters.items()}
        for p in old_model.parameters():
            p.requires_grad_(False)
        self.merge = merge_lora if has_lora(old_model) else None
        if self.merge:
            self.merge(old_model)

    @torch.no_grad()
    def update(self, eta):
        """θ_old ← η θ_old + (1 - η) θ, then refresh the bf16 copy on every rank."""
        for i, e in self.ema.items():
            e.mul_(eta).add_(self.opt.masters[i], alpha=1.0 - eta)
            self.old_params[i].copy_(e)
        if dist.is_initialized():
            for p, owner in zip(self.old_params, self.opt.owner):
                dist.broadcast(p.data, src=owner)
        if self.merge:
            self.merge(self.old_model)

    def shard_state(self):
        return {i: e.detach().cpu() for i, e in self.ema.items()}

    @torch.no_grad()
    def load_shards(self, paths):
        found = set()
        for path in paths:
            shard = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            for i in self.ema:
                if i in shard:
                    self.ema[i].copy_(shard[i])
                    found.add(i)
        if found != set(self.ema):
            raise RuntimeError(f"EMA checkpoint misses {len(set(self.ema) - found)} tensors")
        self.update(1.0)  # copy into the bf16 model without moving the EMA


def eta_schedule(iteration, rate=0.001, max_eta=0.5):
    """DiffusionNFT's default: η_i = min(rate * i, max_eta)."""
    return min(rate * iteration, max_eta)


def nft_step(model, old_model, ema, optimizer, tokenizer, questions, *, group_size, num_steps,
             max_new_tokens, minibatches, beta, eps, embed_weight, ar_weight, noise_scale,
             self_cond_prob, max_grad_norm, eta, seed, reward_fn, truncated_negative=True,
             terms_fn=None, weighting="nft", adv_clip=5.0, normalize="token",
             adaptive_weight=False, token_weight=1.0, keep_uniform=False, mix="prob",
             ppo_clip=0.0, dual_clip=0.0, updates=1, rollout_batch=0):
    """One Lafa-NFT iteration on this rank's ``questions``.

    1. Roll out ``group_size`` answers per question with θ_old and the ODE
       sampler (sde_gamma=0; seeds differ, so initial noise differs).
       Truncated answers count as negatives. ``rollout_batch`` > 0 decodes
       the rank's answers that many at a time (``batched_rollouts``; a
       different random stream than the default one-at-a-time ``rollout``).
    2. Weight the answers (``weighting``):
       - ``"nft"`` (v1, v2): each answer trains one branch (positive if
         correct) with weight ω = 1 - r̂.
       - ``"adv"`` (ppo-lafa): weight 1; the terms function applies the
         advantage R - r̂ itself (``src/rl/ppo_lafa.py``).
       - ``"dn"`` (DiffusionNFT): A = (R - group mean) /
         (std of all rewards in the batch), clipped to ±``adv_clip``,
         r = 0.5 + 0.5 A / ``adv_clip``; each answer trains the positive
         branch with weight r and the negative with 1 - r, both scaled by
         ``adv_clip``/β (DiffusionNFT's ``ori_policy_loss * adv_clip_max``).
       Only groups with 0 < r̂ < 1 are kept unless ``keep_uniform`` (dn only;
       DiffusionNFT keeps them, with r = 0.5 they only anchor θ to θ_old).
    3. ``updates`` passes over the kept answers, each reshuffled and split
       into ``minibatches`` optimizer steps (``updates`` × ``minibatches``
       steps; passes after the first are off-policy, as in PPO). ``normalize``
       ``"token"`` averages each term over the tokens of the minibatch on all
       ranks; ``"answer"`` averages each answer's terms over its own tokens,
       then over the minibatch's answers on all ranks (DiffusionNFT).
    4. θ_old ← η θ_old + (1 - η) θ.

    ``terms_fn`` computes the per-sequence terms (default ``nft_terms``, v1;
    ``src.rl.nft_v2.nft_terms_v2`` for v2; ``src.rl.nft_dn.dn_terms`` for
    DiffusionNFT). ``adaptive_weight`` passes ``adaptive=True`` to it.
    ``token_weight``/``embed_weight``/``ar_weight`` scale the three terms
    (0 drops a term). ``mix``, ``ppo_clip`` and ``dual_clip`` are passed to the
    terms function (``nft_terms``).

    Every rank takes the same number of optimizer steps (collectives match).
    Returns sums over this rank (``count`` = rollouts) for logging.
    """
    from src.rl.flow_policy import batched_rollouts, rollout

    if weighting not in ("nft", "dn", "adv") or normalize not in ("token", "answer"):
        raise ValueError(f"unknown weighting {weighting!r} or normalize {normalize!r}")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    device = next(model.parameters()).device
    sums = dict(count=0, reward=0.0, tokens=0, forwards=0, mixed_groups=0, kept=0,
                truncated=0, token_loss=0.0, embed_loss=0.0, ar_loss=0.0, token_clip=0.0,
                ar_clip=0.0, n_flow=0, n_ar=0, log_ratio_abs_max=0.0, rollout_s=0.0,
                update_s=0.0, grad_norm=0.0, skipped_answers=0, skipped_steps=0)
    start = time.perf_counter()
    groups = []
    if rollout_batch > 0:
        flat = batched_rollouts(old_model, [q["prompt_ids"] for q in questions
                                            for _ in range(group_size)],
                                max_new_tokens=max_new_tokens, num_steps=num_steps,
                                seed=seed, batch_size=rollout_batch)
    for qi, q in enumerate(questions):
        if rollout_batch > 0:
            group = flat[qi * group_size:(qi + 1) * group_size]
        else:
            group = [rollout(old_model, q["prompt_ids"], max_new_tokens=max_new_tokens,
                             num_steps=num_steps, sde_gamma=0.0,
                             seed=seed * 1_000_003 + qi * group_size + g)
                     for g in range(group_size)]
        rewards = []
        for ro in group:
            truncated = truncated_negative and ro.response_ids.numel() >= max_new_tokens
            r = 0.0 if truncated else reward_fn(
                tokenizer.decode(ro.response_ids, skip_special_tokens=True), q["answer"])
            rewards.append(r)
            sums["count"] += 1
            sums["reward"] += r
            sums["truncated"] += int(truncated)
            sums["tokens"] += ro.response_ids.numel()
            sums["forwards"] += ro.num_forwards
        groups.append((group, rewards))

    if weighting == "dn":
        # Global reward std over all ranks (DiffusionNFT's global_std=True).
        stats = torch.tensor([sum(sum(r) for _, r in groups),
                              sum(x * x for _, r in groups for x in r),
                              sum(len(r) for _, r in groups)], dtype=torch.float64, device=device)
        if dist.is_initialized():
            dist.all_reduce(stats)
        n = max(stats[2].item(), 1.0)
        mean = stats[0].item() / n
        std = math.sqrt(max(stats[1].item() / n - mean * mean, 0.0)) + 1e-4
    kept = []
    for group, rewards in groups:
        acc = sum(rewards) / len(rewards)
        if 0.0 < acc < 1.0:
            sums["mixed_groups"] += 1
        elif weighting in ("nft", "adv") or not keep_uniform:
            continue
        for ro, r in zip(group, rewards):
            ids, labels = build_sequence(ro.prompt_ids, ro.response_ids, model.bd_size, pad_id)
            item = dict(ids=ids, labels=labels, positive=r == 1.0, accuracy=acc)
            if weighting == "dn":
                a = min(max((r - acc) / std, -adv_clip), adv_clip)
                item["r"] = 0.5 + 0.5 * a / adv_clip
            kept.append(item)
    sums["kept"] = len(kept)
    sums["rollout_s"] = time.perf_counter() - start

    start = time.perf_counter()
    gen = torch.Generator().manual_seed(seed)
    world = dist.get_world_size() if dist.is_initialized() else 1
    extra = dict(adaptive=True) if adaptive_weight else {}
    if mix != "prob":
        extra["mix"] = mix
    if ppo_clip > 0.0 or dual_clip > 0.0:
        extra.update(ppo_clip=ppo_clip, dual_clip=dual_clip)
    norms = []
    for mb in range(updates * minibatches):
        if mb % minibatches == 0:
            order = torch.randperm(len(kept), generator=gen).tolist()
        batch = [kept[j] for j in order[mb % minibatches::minibatches]]
        for item in batch:
            item["valid"] = model._full_block_loss_mask(item["labels"] != -100)
        counts = torch.tensor([sum(int(it["valid"].sum()) for it in batch),
                               sum(int((it["labels"][0, 1:] != -100).sum()) for it in batch),
                               len(batch)],
                              dtype=torch.float64, device=device)
        if dist.is_initialized():
            dist.all_reduce(counts)
        n_flow, n_ar = max(counts[0].item(), 1.0), max(counts[1].item(), 1.0)
        n_answers = max(counts[2].item(), 1.0)
        for item in batch:
            ids, labels = item["ids"].to(device), item["labels"].to(device)
            inp = noised_input(old_model, ids, labels, noise_scale=noise_scale,
                               self_cond_prob=self_cond_prob)
            with torch.no_grad():
                old_logits = packed_logits(old_model, inp)
            logits = packed_logits(model, inp)
            if weighting == "nft":
                branches = [(item["positive"], 1.0 - item["accuracy"], 1.0)]
            elif weighting == "adv":
                branches = [(item["positive"], 1.0, 1.0)]
            else:
                branches = [(True, item["r"]), (False, 1.0 - item["r"])]
                branches = [(pos, w * adv_clip / beta, w) for pos, w in branches if w > 0.0]
            loss = 0.0
            for i, (positive, weight, log_weight) in enumerate(branches):
                terms = (terms_fn or nft_terms)(logits, old_logits, ids, labels, inp.flow_valid,
                                  model.embed_tokens.weight, positive=positive,
                                  accuracy=item["accuracy"], beta=beta, eps=eps, **extra)
                # x world: the optimizer averages gradients over ranks; this makes
                # the result a sum over all ranks' tokens (answers) divided by the
                # global count.
                # Terms with weight 0 are left out, so a non-finite value
                # cannot leak in as 0 * inf = NaN.
                if normalize == "token":
                    nf, na, n = n_flow, n_ar, 1.0
                else:
                    nf, na, n = max(terms["n_flow"], 1), max(terms["n_ar"], 1), n_answers
                per = sum(w * terms[key] / count for key, w, count in
                          (("token", token_weight, nf), ("embed", embed_weight, nf),
                           ("ar", ar_weight, na)) if w) / n
                loss = loss + weight * world * per
                for key in ("token", "embed", "ar"):
                    sums[f"{key}_loss"] += log_weight * terms[key].item()
                for key in ("token_clip", "ar_clip"):
                    sums[key] += log_weight * terms[key]
                if i == 0:
                    sums["n_flow"] += terms["n_flow"]
                    sums["n_ar"] += terms["n_ar"]
                sums["log_ratio_abs_max"] = max(sums["log_ratio_abs_max"],
                                                terms["log_ratio_abs_max"])
            if torch.is_tensor(loss) and not torch.isfinite(loss):
                print(f"[nft] skipping an answer with non-finite loss: positive="
                      f"{item['positive']} accuracy={item['accuracy']:.3f} "
                      f"terms={ {k: terms[k].item() for k in ('token', 'embed', 'ar')} } "
                      f"log_ratio_abs_max={terms['log_ratio_abs_max']:.1f}", flush=True)
                sums["skipped_answers"] += 1
            elif torch.is_tensor(loss):
                loss.backward()
            optimizer.accumulate()
        norm = optimizer.step(max_grad_norm)
        if not math.isfinite(norm):
            print(f"[nft] skipped an optimizer step: grad norm {norm}", flush=True)
            sums["skipped_steps"] += 1
        else:
            norms.append(norm)
    ema.update(eta)
    sums["grad_norm"] = sum(norms) / max(len(norms), 1)
    sums["update_s"] = time.perf_counter() - start
    return sums
