"""DiffusionNFT's loss for Lafa: one fixed β for every term and both branches.

Pure DiffusionNFT (the baseline) is the embedding term alone, with soft
optimality r, the self-normalized x0 regression and per-answer averaging:

    nft_step(..., terms_fn=dn_terms, weighting="dn", adaptive_weight=True,
             normalize="answer", token_weight=0, ar_weight=0)

    r · ‖x̂⁺ - E[x]‖² / sg(mean|x̂⁺ - E[x]|) + (1 - r) · ‖x̂⁻ - E[x]‖² / sg(…),
    x̂± from (1 ∓ β) x̂_old ± β x̂_θ, scaled by adv_clip / β

Turning on the token CE or AR term gives them the same fixed-β mixtures
(-log p⁺, -log max_st((1+β) - βR, ε)); v2 differs from this in its weighting
(src/rl/nft_v2.py). See our_work/research-notes.md, "DiffusionNFT baseline and
ablations".
"""

from src.rl.nft import nft_terms


def dn_terms(logits, old_logits, input_ids, labels, flow_valid, embed, *, positive, accuracy,
             beta, eps, **kwargs):
    """``nft_terms`` with β for every branch and term. The AR negative takes
    β through NFT's form: (1 - aR)/(1 - a) = (1+β) - βR at a = β/(1+β); the
    AR positive is -log p_θ, the β = 1 mixture."""
    del accuracy
    return nft_terms(logits, old_logits, input_ids, labels, flow_valid, embed,
                     positive=positive, accuracy=beta / (1.0 + beta), beta=beta, eps=eps,
                     **kwargs)
