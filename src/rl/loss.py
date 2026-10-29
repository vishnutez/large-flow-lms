"""GRPO-style loss over flow transitions and AR-head tokens.

Both terms come from one ``packed_step_means`` forward and share the
sequence's advantage:

    L = flow_pg + ar_weight * ar_pg

``flow_pg`` is the clipped policy gradient over the stochastic denoising
transitions (K-1 per block), with each transition's log-prob averaged over
its coordinates (Flow-GRPO's convention). ``ar_pg`` is the clipped policy
gradient of the AR head over the response tokens. The AR head did not sample
those tokens (the flow did), so its ratio acts as a trust region around the
old AR head rather than an importance correction.

Old log-probs come from the packed forward (``old_logprobs``), not from the
rollout's recorded means. With one optimizer step per batch of rollouts the
old policy is the current one, so ``old=None`` uses the detached current
log-probs and saves a forward.

The rollout's recorded means are not corrected for (TODO in
our_work/research-notes.md: truncated importance weights). ``mismatch_stats`` reports
how far they are from the packed means.
"""

import torch
import torch.nn.functional as F

from src.rl.flow_policy import packed_step_means, transition_logprob


def ar_token_logprobs(clean_logits, ro):
    """AR-head log-prob of each response token, [L]. Position n predicts token n+1."""
    P, L = ro.prompt_ids.numel(), ro.response_ids.numel()
    logp = F.log_softmax(clean_logits[P - 1:P + L - 1].float(), dim=-1)
    return logp.gather(-1, ro.response_ids.to(logp.device)[:, None])[:, 0]


def policy_logprobs(model, ro, steps):
    """Flow transition log-probs [len(steps), num_blocks] (per coordinate), AR
    token log-probs [L], and the packed means [len(steps), num_blocks, B, d]."""
    means, clean_logits = packed_step_means(model, ro, steps)
    flow = torch.stack([transition_logprob(ro, k, means[i], per_dim=True)
                        for i, k in enumerate(steps)]) if steps else \
        torch.zeros(0, len(ro.blocks), dtype=torch.float64, device=clean_logits.device)
    return flow, ar_token_logprobs(clean_logits, ro), means


@torch.no_grad()
def old_logprobs(model, ro, steps):
    flow, ar, _ = policy_logprobs(model, ro, steps)
    return flow, ar


@torch.no_grad()
def mismatch_stats(ro, steps, means):
    """Per-block log-ratio packed / rollout (summed over coordinates)."""
    if not steps:
        return dict(mismatch_mean=0.0, mismatch_abs_max=0.0, mismatch_frac_gt1=0.0)
    ratios = torch.cat([
        transition_logprob(ro, k, means[i])
        - transition_logprob(ro, k, torch.stack([blk.means[k] for blk in ro.blocks]))
        for i, k in enumerate(steps)])
    return dict(mismatch_mean=ratios.mean().item(), mismatch_abs_max=ratios.abs().max().item(),
                mismatch_frac_gt1=(ratios.abs() > 1).float().mean().item())


def clipped_pg(logp, old_logp, advantage, clip):
    """PPO clipped surrogate, elementwise (to be minimized)."""
    ratio = torch.exp(logp - old_logp)
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)


def sequence_loss(model, ro, advantage, old, steps, *, clip=0.2, ar_weight=1.0):
    """Loss for one rollout. Each term is averaged over its own elements.

    ``old`` is ``old_logprobs(model, ro, steps)`` from before the update, or
    None when the update is on-policy. Returns (loss, stats).
    """
    flow, ar, means = policy_logprobs(model, ro, steps)
    old_flow, old_ar = old if old is not None else (flow.detach(), ar.detach())
    flow_terms = clipped_pg(flow, old_flow, advantage, clip)
    ar_terms = clipped_pg(ar.double(), old_ar.double(), advantage, clip)
    flow_loss = flow_terms.mean() if flow_terms.numel() else flow_terms.sum()
    ar_loss = ar_terms.mean()
    loss = flow_loss + ar_weight * ar_loss
    with torch.no_grad():
        stats = dict(
            flow_loss=flow_loss.item(), ar_loss=ar_loss.item(),
            flow_clip_frac=((flow - old_flow).exp() - 1).abs().gt(clip).float().mean().item()
            if flow.numel() else 0.0,
            ar_clip_frac=((ar - old_ar).exp() - 1).abs().gt(clip).float().mean().item(),
            **mismatch_stats(ro, steps, means))
    return loss.float(), stats
