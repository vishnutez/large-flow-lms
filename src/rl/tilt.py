"""Sigmoid-tilt NFT (``--loss tilt``): an implicit ⁺/⁻ pair that is always valid.

v2's implicit negative (p_old - r̂ p_θ)/(1 - r̂) leaves the simplex once
p_θ > p_old/r̂. Here both implicit policies are tilts of p_old by a per-token
success probability σ built from the trained model:

    σ = F(β log(p_θ/p_old) + b),   α = Σ p_old σ,
    p⁺ = p_old σ / α,   p⁻ = p_old (1 - σ) / (1 - α),

so p⁺, p⁻ are distributions and α p⁺ + (1 - α) p⁻ = p_old for every θ, for
any increasing link F: ℝ → (0, 1) (our_work/tilt-lemma.md). The link only
shapes the off-policy gradient; the terms are rescaled so the on-policy
gradient is β (R - r̂) ∇(Lafa loss) for every link. Links (``link``):

    "kappa"    F_κ(s) = sigmoid(κ s)^(1/κ). κ = 1 is the sigmoid; as κ → ∞ it
               tends to min(eˢ, 1), where the pair becomes v2's (p⁺ = p_θ at
               β = 1). The push on a negative's token levels off at O(κ).
    "cloglog"  F(s) = 1 - exp(-eˢ). The push on a negative's token is
               r̂ q^β (v2's r̂ q at β = 1) and has no ceiling, so pair it with
               ``neg_clip``.

``neg_clip`` = c > 0 stops the token CE and AR gradient of a negative's token
once p_θ/p_old > c (v2's dual clip, ``ppo_trust``). The optimum stays σ = ρ
for every token whose optimal ratio is below c.

The loss is v2's (positives fit p⁺, negatives fit p⁻; token CE, embedding MSE
on x̂± = p± E, AR CE) with no floor. The deployed model is softmax(l_θ). b is
one scalar per slot (``b_mode``):

    "solve"  b such that α = r̂ (root find; its gradient is the implicit one)
    "fixed"  b = F⁻¹(r̂)

See our_work/sigmoid-tilt.md.
"""

import math

import torch
import torch.nn.functional as F

from src.rl.nft import ppo_trust

LINKS = ("kappa", "cloglog")


def log_link(s, kappa, link="kappa"):
    """log F(s) and log(1 - F(s)), stable at any s.

    kappa: F = sigmoid(κ s)^(1/κ); 1 - F ≈ e^(-κ s)/κ underflows in float32
    past κ s ≈ 90. cloglog: F = 1 - exp(-eˢ), log(1 - F) = -eˢ exactly."""
    if link == "cloglog":
        u = torch.exp(s.clamp(-15.0, 30.0))  # past 30, F = 1 in float32
        # Below s = -15, log(1 - e^(-eˢ)) = s - eˢ/2 + O(e^(2s)).
        log_f = torch.where(s < -15.0, s - torch.exp(s.clamp_max(-15.0)) / 2,
                            torch.log((-torch.expm1(-u)).clamp_min(1e-30)))
        return log_f, -torch.exp(s.clamp_max(80.0))
    if link != "kappa":
        raise ValueError(f"unknown link {link!r}")
    if kappa == 1.0:
        return F.logsigmoid(s), F.logsigmoid(-s)
    z = kappa * s
    log_f = F.logsigmoid(z) / kappa
    direct = torch.log((-torch.expm1(log_f)).clamp_min(1e-30))
    # Near F = 1: log(1 - e^l) = log(-l) + l/2 + O(l²), -l = softplus(-z)/κ.
    log_sp = torch.where(z > 20.0, -z, torch.log(F.softplus(-z).clamp_min(1e-30)))
    near = log_sp - math.log(kappa) + log_f / 2
    return log_f, torch.where(log_f < -1e-4, direct, near)


def log_dlog_link(s, kappa, link="kappa"):
    """log of (log F)'(s): the positive's gradient factor per unit s."""
    if link == "cloglog":
        # (log F)' = eˢ / (exp(eˢ) - 1) = exp(s - u - log(1 - e^(-u))), u = eˢ
        sc = s.clamp_min(-60.0)
        u = torch.exp(sc)
        return sc - u - torch.log((-torch.expm1(-u)).clamp_min(1e-30))
    return F.logsigmoid(-kappa * s)


def link_inverse(p, kappa, link="kappa"):
    """F⁻¹(p)."""
    if link == "cloglog":
        return math.log(-math.log1p(-p))
    return math.log(p) - math.log1p(-p ** kappa) / kappa


def onpolicy_scale(accuracy, kappa, link="kappa"):
    """(1 - r̂) / (log F)'(F⁻¹(r̂)): makes the on-policy gradient β (R - r̂) ∇ℓ."""
    if link == "cloglog":
        return accuracy / -math.log1p(-accuracy)
    return (1.0 - accuracy) / (1.0 - accuracy ** kappa)


def solve_b(lq, lp_old, target, kappa=1.0, iters=30, link="kappa"):
    """Per row, b with log Σ p_old F(lq + b) = ``target`` (= log r̂).

    Safeguarded Newton on the bracket b ∈ [F⁻¹(r̂) - max lq, F⁻¹(r̂) - min lq]
    (α ≤ r̂ at the low end, ≥ r̂ at the high end). No gradient."""
    with torch.no_grad():
        start = link_inverse(math.exp(target), kappa, link)
        lo = start - lq.max(-1).values
        hi = start - lq.min(-1).values
        b = torch.full_like(lo, start)
        for _ in range(iters):
            s = lq + b[:, None]
            ls, _ = log_link(s, kappa, link)
            f = torch.logsumexp(lp_old + ls, -1) - target
            lo = torch.where(f < 0, b, lo)
            hi = torch.where(f >= 0, b, hi)
            # d log α / db = E_{p⁺}[(log F)'(s)]
            df = torch.exp(torch.logsumexp(lp_old + ls + log_dlog_link(s, kappa, link), -1)
                           - (f + target)).clamp_min(1e-12)
            b = b - f / df
            b = torch.where((b <= lo) | (b >= hi), 0.5 * (lo + hi), b)
        return b


def tilt_logprobs(lp, lp_old, beta, accuracy, b_mode, kappa=1.0, link="kappa"):
    """log p⁺, log p⁻ (rows over the vocabulary) and b, from log p_θ and log p_old."""
    lq = beta * (lp - lp_old)
    if b_mode == "fixed":
        b = torch.full(lq.shape[:-1], link_inverse(accuracy, kappa, link),
                       dtype=lq.dtype, device=lq.device)
    elif b_mode == "solve":
        target = math.log(accuracy)
        b0 = solve_b(lq.detach(), lp_old, target, kappa, link=link)
        # One Newton step from the root: the value stays b0, the gradient is
        # the implicit one, db = -(∂ log α/∂lq) / (∂ log α/∂b).
        s0 = lq + b0[:, None]
        ls, _ = log_link(s0, kappa, link)
        log_a = torch.logsumexp(lp_old + ls, -1)
        df = torch.exp(torch.logsumexp(lp_old + ls + log_dlog_link(s0, kappa, link), -1)
                       - log_a).detach().clamp_min(1e-12)
        b = b0 - (log_a - target) / df
    else:
        raise ValueError(f"unknown b_mode {b_mode!r}")
    s = lq + b[:, None]
    log_sig, log_1msig = log_link(s, kappa, link)
    log_a = torch.logsumexp(lp_old + log_sig, -1, keepdim=True)
    log_1ma = torch.logsumexp(lp_old + log_1msig, -1, keepdim=True)
    return lp_old + log_sig - log_a, lp_old + log_1msig - log_1ma, b


def tilt_terms(logits, old_logits, input_ids, labels, flow_valid, embed, *, positive,
               accuracy, beta, eps, b_mode="solve", kappa=1.0, link="kappa", neg_clip=0.0,
               adaptive=False, **kwargs):
    """Summed per-sequence terms (``nft_terms``' interface): token CE and
    embedding MSE on the denoiser's p±, CE on the AR head's p±. ``eps`` is
    unused. The terms are scaled by ``onpolicy_scale``, so the on-policy
    gradient does not depend on the link. ``token_clip``/``ar_clip`` count
    tokens without gradient: saturated (the trained token's factor,
    (log F)' for a positive or -(log(1 - F))' for a negative, below 1e-3) or
    cut by ``neg_clip``."""
    del eps, kwargs
    if beta <= 0.0 or kappa <= 0.0:
        raise ValueError("tilt needs --beta > 0 and --tilt-kappa > 0")
    if link not in LINKS:
        raise ValueError(f"unknown link {link!r}")
    scale = onpolicy_scale(accuracy, kappa, link)
    L = input_ids.shape[1]
    ids = input_ids[0]
    valid = flow_valid[0]
    x = ids[valid]
    lp = F.log_softmax(logits[0, L:][valid].float(), dim=-1)
    lp_old = F.log_softmax(old_logits[0, L:][valid].float(), dim=-1).detach()
    lp_pos, lp_neg, b = tilt_logprobs(lp, lp_old, beta, accuracy, b_mode, kappa, link)
    lp_mix = lp_pos if positive else lp_neg
    token = -lp_mix.gather(-1, x[:, None])[:, 0]
    log_q_x = (lp.gather(-1, x[:, None])[:, 0] - lp_old.gather(-1, x[:, None])[:, 0])
    token, token_cut = ppo_trust(token, log_q_x, positive, 0.0, neg_clip)

    E = embed.float()
    diff = lp_mix.exp() @ E - E[x]
    if adaptive:
        s = diff.detach().abs().mean().clamp_min(1e-5) if diff.numel() else 1.0
        emb = (diff ** 2).mean(-1) / s
    else:
        emb = (diff ** 2).sum(-1)

    ar_valid = labels[0, 1:] != -100
    y = ids[1:][ar_valid]
    lp_ar = F.log_softmax(logits[0, :L - 1][ar_valid].float(), dim=-1)
    lpo_ar = F.log_softmax(old_logits[0, :L - 1][ar_valid].float(), dim=-1).detach()
    ar_pos, ar_neg, b_ar = tilt_logprobs(lp_ar, lpo_ar, beta, accuracy, b_mode, kappa, link)
    ar = -(ar_pos if positive else ar_neg).gather(-1, y[:, None])[:, 0]
    log_q_y = lp_ar.gather(-1, y[:, None])[:, 0] - lpo_ar.gather(-1, y[:, None])[:, 0]
    ar, ar_cut = ppo_trust(ar, log_q_y, positive, 0.0, neg_clip)

    with torch.no_grad():
        def stopped(lq, b, t, cut):
            s = lq.gather(-1, t[:, None])[:, 0] + b
            log_g = log_dlog_link(s, kappa, link)
            if not positive:
                log_f, log_1mf = log_link(s, kappa, link)
                log_g = log_g + log_f - log_1mf
            return ((log_g < math.log(1e-3)) | cut).float().sum()
        token_clip = stopped(beta * (lp - lp_old), b, x, token_cut)
        ar_clip = stopped(beta * (lp_ar - lpo_ar), b_ar, y, ar_cut)
        log_ratio = torch.cat([log_q_x, log_q_y])
    return dict(token=scale * token.sum(), embed=scale * emb.sum(), ar=scale * ar.sum(),
                n_flow=int(valid.sum()), n_ar=int(ar_valid.sum()),
                token_clip=float(token_clip), ar_clip=float(ar_clip),
                log_ratio_abs_max=float(log_ratio.abs().max()) if log_ratio.numel() else 0.0)
