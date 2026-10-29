"""Lafa-NFT v2: v1 (src/rl/nft.py) with the group baseline restored.

v1 used one fixed β for both branches of the denoiser terms, so on-policy a
correct answer got advantage +(1 - r̂) and a wrong one -(1 - r̂): not
group-relative (see our_work/research-notes.md, "v1's missing baseline"). v2 gives the
negative branch NFT's coupling β⁻ = r̂/(1 - r̂) for the token CE and the
embedding MSE, as v1 already did for the AR head:

    token CE   -log max_st((1 - r̂ R) / (1 - r̂), ε)        ((1+β⁻) - β⁻R)
    embedding  ‖(1+β⁻) x̂_old - β⁻ x̂_θ - E[x]‖²

The positive branch keeps the configured β (β⁺ = 1 by default). With
ω = 1 - r̂, every term then has on-policy advantage r - r̂ (Dr. GRPO).
"""

from src.rl.nft import nft_terms


def nft_terms_v2(logits, old_logits, input_ids, labels, flow_valid, embed, *, positive,
                 accuracy, beta, eps, **kwargs):
    """``nft_terms`` with β⁺ = ``beta`` and β⁻ = r̂/(1 - r̂)."""
    branch_beta = beta if positive else accuracy / (1.0 - accuracy)
    return nft_terms(logits, old_logits, input_ids, labels, flow_valid, embed,
                     positive=positive, accuracy=accuracy, beta=branch_beta, eps=eps,
                     **kwargs)
