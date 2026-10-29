"""PPO on Lafa's training loss (``--loss ppo-lafa``).

The advantage-weighted Lafa loss A·L (A = R - r̂, not std-normalized) with a
PPO trust region per token. Lafa's per-slot loss e = CE + MSE is -log of the
unnormalized density p_θ(x) exp(-‖p_θ E - E[x]‖²) (the MSE read as a Gaussian
log-likelihood with 2σ² = 1), so each noised slot gets one ratio

    w = exp(e_old - e_θ) = p_θ(x)/p_old(x) · exp(-(MSE_θ - MSE_old)),

and each AR position the ratio of next-token probabilities. Per token:

    ℓ = -min(w A, clip(w, 1 - δ, 1 + δ) A),  and for A < 0 also capped at
    -c A (dual clip: no gradient once w > c).

On-policy (w = 1) the gradient is A ∇(CE + MSE) and A ∇AR, i.e. ∇(A·L), the
same as v2's (see our_work/research-notes.md). Off-policy the clip is the trust region.
"""

import math

import torch
import torch.nn.functional as F


def ppo_surrogate(log_w, adv, ppo_clip, dual_clip):
    """Per-token PPO loss -min(w A, clip(w) A), dual-clipped for A < 0.

    log w is capped at 20 for the value: every ratio above 1 + δ (A > 0) or
    above c (A < 0) has zero gradient anyway, and the cap keeps exp finite
    so the unused branch cannot produce 0 · inf = NaN in the backward pass.
    Returns (loss, mask of tokens with no gradient)."""
    w = torch.exp(log_w.clamp(max=20.0))
    obj = torch.minimum(w * adv, w.clamp(1.0 - ppo_clip, 1.0 + ppo_clip) * adv)
    with torch.no_grad():
        wd = w.detach()
        clipped = wd > 1.0 + ppo_clip if adv > 0 else wd < 1.0 - ppo_clip
    if adv < 0 and dual_clip > 0.0:
        obj = torch.maximum(obj, torch.full_like(obj, dual_clip * adv))
        clipped = clipped | (w.detach() > dual_clip)
    return -obj, clipped


def ppo_lafa_terms(logits, old_logits, input_ids, labels, flow_valid, embed, *, positive,
                   accuracy, beta, eps, ppo_clip=0.2, dual_clip=3.0, **kwargs):
    """Summed per-sequence PPO terms; ``token`` carries the combined CE + MSE
    slot surrogate, ``embed`` is 0 (kept for the caller's interface).
    ``beta``/``eps`` are unused."""
    del beta, eps, kwargs
    if ppo_clip <= 0.0:
        raise ValueError("ppo-lafa needs --ppo-clip > 0")
    adv = (1.0 if positive else 0.0) - accuracy
    L = input_ids.shape[1]
    ids = input_ids[0]
    valid = flow_valid[0]
    x = ids[valid]
    lp = F.log_softmax(logits[0, L:][valid].float(), dim=-1)
    lp_old = F.log_softmax(old_logits[0, L:][valid].float(), dim=-1)
    E = embed.float()
    e_theta = -lp.gather(-1, x[:, None])[:, 0] + ((lp.exp() @ E - E[x]) ** 2).sum(-1)
    with torch.no_grad():
        e_old = -lp_old.gather(-1, x[:, None])[:, 0] + ((lp_old.exp() @ E - E[x]) ** 2).sum(-1)
    token, token_clip = ppo_surrogate(e_old - e_theta, adv, ppo_clip, dual_clip)

    ar_valid = labels[0, 1:] != -100
    y = ids[1:][ar_valid]
    lp_ar = F.log_softmax(logits[0, :L - 1][ar_valid].float(), dim=-1).gather(-1, y[:, None])[:, 0]
    with torch.no_grad():
        lpo_ar = F.log_softmax(old_logits[0, :L - 1][ar_valid].float(), dim=-1).gather(
            -1, y[:, None])[:, 0]
    ar, ar_clip = ppo_surrogate(lp_ar - lpo_ar, adv, ppo_clip, dual_clip)

    with torch.no_grad():
        log_ratio = torch.cat([e_old - e_theta.detach(), lp_ar.detach() - lpo_ar])
    return dict(token=token.sum(), embed=torch.zeros((), device=token.device), ar=ar.sum(),
                n_flow=int(valid.sum()), n_ar=int(ar_valid.sum()),
                token_clip=float(token_clip.sum()), ar_clip=float(ar_clip.sum()),
                log_ratio_abs_max=float(log_ratio.abs().max()) if log_ratio.numel() else 0.0)
