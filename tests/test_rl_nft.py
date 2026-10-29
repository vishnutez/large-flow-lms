import copy
import itertools
import math

import pytest
import torch

from src.rl.nft import (OldPolicyEMA, build_sequence, nft_step, nft_terms, noised_input,
                        packed_logits, st_clamp_min)
from tests.test_rl_flow_policy import tiny_model


def pair():
    """A policy and an identical old model (θ = θ_old)."""
    model = tiny_model(True)
    old = copy.deepcopy(model)
    for m in (model, old):
        m.embed_tokens.weight.requires_grad_(False)
        m.lm_head.weight.requires_grad_(False)
    return model, old


def sequence(model):
    torch.manual_seed(3)
    return build_sequence(torch.tensor([2, 3, 4, 5, 6]), torch.randint(1, 32, (11,)),
                          model.bd_size, pad_id=0)


def test_build_sequence_pads_and_masks():
    ids, labels = build_sequence(torch.tensor([2, 3, 4]), torch.tensor([7, 8, 9, 10]), 4, 0)
    assert ids.tolist() == [[2, 3, 4, 7, 8, 9, 10, 0]]
    assert labels.tolist() == [[-100, -100, -100, 7, 8, 9, 10, -100]]


def test_st_clamp_min():
    x = torch.tensor([0.5, 2.0], requires_grad=True)
    y = st_clamp_min(x, 1.0)
    assert y.tolist() == [1.0, 2.0]
    y.sum().backward()
    assert x.grad.tolist() == [1.0, 1.0]
    # Exact far below the floor (the old form returned 0 at -1e8 in float32).
    assert st_clamp_min(torch.tensor([-1.0e8, -3.0e9]), 1.0).tolist() == [1.0, 1.0]


def test_shared_input_gives_identical_logits():
    model, old = pair()
    ids, labels = sequence(model)
    inp = noised_input(old, ids, labels, noise_scale=4.0, self_cond_prob=1.0)
    assert inp.e_prev is not None and inp.flow_valid.sum() > 0
    with torch.no_grad():
        torch.testing.assert_close(packed_logits(model, inp), packed_logits(old, inp))


def grads(model, loss):
    model.zero_grad()
    loss.backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                      for p in model.parameters() if p.requires_grad])


def onpolicy_case(positive, accuracy=0.25):
    model, old = pair()
    ids, labels = sequence(model)
    inp = noised_input(old, ids, labels, noise_scale=4.0, self_cond_prob=0.0)
    with torch.no_grad():
        old_logits = packed_logits(old, inp)

    def terms():
        return nft_terms(packed_logits(model, inp), old_logits, ids, labels, inp.flow_valid,
                         model.embed_tokens.weight, positive=positive, accuracy=accuracy,
                         beta=1.0, eps=1.0)

    def reference():
        """Sum of log p_θ at the targets: denoiser slots and AR positions."""
        logits = packed_logits(model, inp)
        L = ids.shape[1]
        v = inp.flow_valid[0]
        lp = logits[0, L:][v].float().log_softmax(-1).gather(-1, ids[0][v][:, None]).sum()
        ar_v = labels[0, 1:] != -100
        lp_ar = logits[0, :L - 1][ar_v].float().log_softmax(-1).gather(
            -1, ids[0, 1:][ar_v][:, None]).sum()
        return lp, lp_ar

    return model, terms, reference


def test_onpolicy_positive_gradients_are_maximum_likelihood():
    model, terms, reference = onpolicy_case(positive=True)
    lp, lp_ar = reference()
    torch.testing.assert_close(grads(model, terms()['token']), grads(model, -reference()[0]))
    torch.testing.assert_close(grads(model, terms()['ar']), grads(model, -reference()[1]))


def test_onpolicy_negative_gradients_push_down():
    acc = 0.25
    model, terms, reference = onpolicy_case(positive=False, accuracy=acc)
    # Token term (β = 1): ratio 2 - R at R = 1 is 1 → gradient of +log p.
    torch.testing.assert_close(grads(model, terms()['token']), grads(model, reference()[0]))
    # AR term (NFT): gradient of r̂/(1 - r̂) · log p.
    torch.testing.assert_close(grads(model, terms()['ar']),
                               grads(model, acc / (1 - acc) * reference()[1]))


def test_embedding_term_onpolicy_is_denoising_mse():
    model, terms, _ = onpolicy_case(positive=True)
    t = terms()
    assert t['embed'] > 0 and t['n_flow'] > 0
    # Positive and negative embedding terms pull in opposite directions at θ = θ_old.
    g_pos = grads(model, terms()['embed'])
    model2, terms2, _ = onpolicy_case(positive=False)
    g_neg = grads(model2, terms2()['embed'])
    torch.testing.assert_close(g_neg, -g_pos)


def test_ema_update_and_sync():
    from src.rl.grpo import MasterAdamW
    model, old = pair()
    opt = MasterAdamW(model.parameters(), lr=1e-2)
    ema = OldPolicyEMA(opt, model, old)
    for p in opt.params:
        p.grad = torch.randn_like(p)
    opt.accumulate()
    opt.step(1.0)
    i = opt.owned[0]
    before = ema.ema[i].clone()
    ema.update(0.25)
    torch.testing.assert_close(ema.ema[i], 0.25 * before + 0.75 * opt.masters[i])
    torch.testing.assert_close(ema.old_params[i].float(), ema.ema[i])
    assert not any(p.requires_grad for p in old.parameters())


def test_nft_step_trains_policy_and_moves_old(monkeypatch):
    from src.rl.grpo import MasterAdamW

    class Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return ''

    model, old = pair()
    opt = MasterAdamW(model.parameters(), lr=1e-3)
    ema = OldPolicyEMA(opt, model, old)
    before = [p.detach().clone() for p in opt.params]
    old_before = [p.detach().clone() for p in ema.old_params]
    rewards = itertools.cycle([1.0, 0.0, 0.0])
    q = dict(prompt_ids=torch.tensor([2, 3, 4, 5, 6]), answer='')
    sums = nft_step(model, old, ema, opt, Tok(), [q], group_size=6, num_steps=3,
                    max_new_tokens=12, minibatches=2, beta=1.0, eps=1.0, embed_weight=1.0,
                    ar_weight=1.0, noise_scale=4.0, self_cond_prob=0.5, max_grad_norm=1.0,
                    eta=0.5, seed=1, reward_fn=lambda r, a: next(rewards),
                    truncated_negative=False)
    assert sums['count'] == 6 and sums['mixed_groups'] == 1 and sums['kept'] == 6
    assert sums['n_flow'] > 0 and sums['n_ar'] > 0
    assert any(not torch.equal(b, p) for b, p in zip(before, opt.params))
    assert any(not torch.equal(b, p) for b, p in zip(old_before, ema.old_params))


@pytest.mark.parametrize('mix', ['prob', 'logit', 'geo'])
@pytest.mark.parametrize('acc', [0.125, 0.5, 0.875])
def test_v2_onpolicy_advantages_are_group_relative(acc, mix):
    """At θ = θ_old, each v2 term's negative gradient is -r̂/(1-r̂) times its
    positive gradient, so with ω = 1 - r̂ the advantages are 1 - r̂ and -r̂."""
    from src.rl.nft_v2 import nft_terms_v2
    model, old = pair()
    ids, labels = sequence(model)
    inp = noised_input(old, ids, labels, noise_scale=4.0, self_cond_prob=0.0)
    with torch.no_grad():
        old_logits = packed_logits(old, inp)

    def term(positive, key):
        t = nft_terms_v2(packed_logits(model, inp), old_logits, ids, labels, inp.flow_valid,
                         model.embed_tokens.weight, positive=positive, accuracy=acc,
                         beta=1.0, eps=1.0, mix=mix)
        return grads(model, t[key])

    for key in ('token', 'embed', 'ar'):
        g_pos, g_neg = term(True, key), term(False, key)
        assert g_pos.abs().sum() > 0
        torch.testing.assert_close(g_neg, -acc / (1 - acc) * g_pos, rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize('adaptive', [False, True])
def test_dn_onpolicy_branches_are_opposite(adaptive):
    """DiffusionNFT (one β = 1): at θ = θ_old every term's negative gradient is
    minus its positive one, so r·L⁺ + (1-r)·L⁻ has advantage 2r - 1."""
    from src.rl.nft_dn import dn_terms
    model, old = pair()
    ids, labels = sequence(model)
    inp = noised_input(old, ids, labels, noise_scale=4.0, self_cond_prob=0.0)
    with torch.no_grad():
        old_logits = packed_logits(old, inp)

    def term(positive, key):
        t = dn_terms(packed_logits(model, inp), old_logits, ids, labels, inp.flow_valid,
                     model.embed_tokens.weight, positive=positive, accuracy=0.125,
                     beta=1.0, eps=1.0, adaptive=adaptive)
        return grads(model, t[key])

    for key in ('token', 'embed', 'ar'):
        g_pos, g_neg = term(True, key), term(False, key)
        assert g_pos.abs().sum() > 0
        torch.testing.assert_close(g_neg, -g_pos, rtol=1e-4, atol=1e-7)


def test_adaptive_embedding_term_is_self_normalized():
    model, old = pair()
    ids, labels = sequence(model)
    inp = noised_input(old, ids, labels, noise_scale=4.0, self_cond_prob=0.0)
    with torch.no_grad():
        logits = packed_logits(model, inp)
        t = nft_terms(logits, logits, ids, labels, inp.flow_valid, model.embed_tokens.weight,
                      positive=True, accuracy=0.5, beta=1.0, eps=1.0, adaptive=True)
        L, v = ids.shape[1], inp.flow_valid[0]
        E = model.embed_tokens.weight.float()
        diff = logits[0, L:][v].float().softmax(-1) @ E - E[ids[0][v]]
        expected = ((diff ** 2).mean(-1) / diff.abs().mean()).sum()
    torch.testing.assert_close(t['embed'], expected)


def run_dn_step(rewards, **kw):
    from src.rl.grpo import MasterAdamW
    from src.rl.nft_dn import dn_terms

    class Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return ''

    model, old = pair()
    opt = MasterAdamW(model.parameters(), lr=1e-3)
    ema = OldPolicyEMA(opt, model, old)
    before = [p.detach().clone() for p in opt.params]
    rewards = iter(rewards)
    q = dict(prompt_ids=torch.tensor([2, 3, 4, 5, 6]), answer='')
    sums = nft_step(model, old, ema, opt, Tok(), [q, q], group_size=4, num_steps=3,
                    max_new_tokens=12, minibatches=1, beta=1.0, eps=1.0, embed_weight=1.0,
                    ar_weight=0.0, noise_scale=4.0, self_cond_prob=0.0, max_grad_norm=1.0,
                    eta=0.5, seed=1, reward_fn=lambda r, a: next(rewards),
                    truncated_negative=False, terms_fn=dn_terms, weighting='dn',
                    normalize='answer', adaptive_weight=True, token_weight=0.0, **kw)
    moved = any(not torch.equal(b, p) for b, p in zip(before, opt.params))
    return sums, moved


def test_dn_step_keeps_mixed_groups_by_default():
    sums, moved = run_dn_step([1.0, 0.0, 0.0, 1.0] + [1.0] * 4)
    assert sums['kept'] == 4 and sums['mixed_groups'] == 1
    assert moved and sums['embed_loss'] > 0
    sums, _ = run_dn_step([1.0, 0.0, 0.0, 1.0] + [1.0] * 4, keep_uniform=True)
    assert sums['kept'] == 8


def test_dn_step_uniform_groups_have_no_onpolicy_gradient():
    """r = 0.5 for every answer of a uniform group: at θ = θ_old the branches
    cancel (up to rounding) and the gradient vanishes."""
    sums, _ = run_dn_step([1.0] * 4 + [0.0] * 4, keep_uniform=True)
    assert sums['kept'] == 8 and sums['mixed_groups'] == 0
    mixed, _ = run_dn_step([1.0, 0.0, 0.0, 1.0] + [1.0] * 4, keep_uniform=True)
    assert sums['grad_norm'] < 1e-4 * mixed['grad_norm']


def test_optimizer_skips_non_finite_gradients():
    from src.rl.grpo import MasterAdamW
    model, _ = pair()
    opt = MasterAdamW(model.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in opt.params]
    for p in opt.params:
        p.grad = torch.full_like(p, float('nan'))
    opt.accumulate()
    assert not math.isfinite(opt.step(1.0))
    assert all(torch.equal(b, p) for b, p in zip(before, opt.params))
    assert all(g.abs().sum() == 0 for g in opt.grads)


def test_logit_mix_gradient_is_bounded_off_policy():
    """Far from θ_old (log R ~ 40), the probability-mixture negative's
    straight-through gradient explodes; the logit mixture's stays ≤ √2 β."""
    from src.rl.nft import logit_mix_logprob
    torch.manual_seed(0)
    z_old = torch.randn(6, 50)
    z_old[:, 0] = -40.0  # θ_old gives the target almost no probability
    z = torch.randn(6, 50, requires_grad=True)
    target = torch.zeros(6, dtype=torch.long)
    beta = 7.0
    (-logit_mix_logprob(z, z_old, target, beta, positive=False)).sum().backward()
    assert torch.isfinite(z.grad).all()
    assert z.grad.norm(dim=-1).max() <= 2 ** 0.5 * beta + 1e-4
    # Positive at β = 1 is plain log-softmax.
    torch.testing.assert_close(logit_mix_logprob(z, z_old, target, 1.0, True),
                               z.log_softmax(-1)[:, 0])


def test_logit_mix_reaches_the_ar_negative():
    """mix="logit" must also build the AR negative from logits (no ε clip)."""
    from src.rl.nft import nft_terms
    torch.manual_seed(0)
    L, V, d = 6, 11, 4
    ids = torch.randint(0, V, (1, L))
    labels = ids.clone()
    labels[0, :2] = -100
    logits = torch.randn(1, 2 * L, V) * 3
    old = logits + torch.randn_like(logits) * 3
    valid = torch.zeros(1, L, dtype=torch.bool)
    valid[0, 2:] = True
    terms = nft_terms(logits, old, ids, labels, valid, torch.randn(V, d), positive=False,
                      accuracy=0.75, beta=3.0, eps=1.0, mix="logit")
    assert terms["ar_clip"] == 0.0 and terms["token_clip"] == 0.0


@pytest.mark.parametrize('positive', [True, False])
def test_ppo_trust_stops_gradient_outside_the_region(positive):
    """With --ppo-clip, a token's CE gradient is zero once q = p/p_old moved
    past 1 ± δ in its branch's direction, or (negative) past the dual clip;
    the forward value is unchanged."""
    from src.rl.nft import ppo_trust
    log_q = torch.log(torch.tensor([0.5, 0.9, 1.0, 1.1, 2.0, 50.0]))
    lp = torch.zeros(6, requires_grad=True)
    loss = -lp * torch.arange(1.0, 7.0)
    out, clipped = ppo_trust(loss, log_q, positive, 0.2, 3.0)
    torch.testing.assert_close(out, loss)
    out.sum().backward()
    expect = ([True, True, True, True, False, False] if positive
              else [False, True, True, True, True, False])
    assert (lp.grad != 0).tolist() == expect
    assert (~clipped).tolist() == expect


def test_ppo_clip_bounds_the_v2_negative_gradient_off_policy():
    """v2's straight-through negative has gradient ~β q in log p; with the dual
    clip, tokens far above θ_old (q ≫ 1) get none."""
    from src.rl.nft import nft_terms
    torch.manual_seed(0)
    L, V, d = 6, 11, 4
    ids = torch.randint(0, V, (1, L))
    labels = ids.clone()
    labels[0, :2] = -100
    logits = torch.randn(1, 2 * L, V)
    old = logits.clone()
    old[0, torch.arange(L - 1), ids[0, 1:]] -= 20.0  # q ≈ e^20 on every AR target
    old[0, torch.arange(L, 2 * L), ids[0]] -= 20.0  # and every denoiser target
    logits.requires_grad_(True)
    valid = torch.zeros(1, L, dtype=torch.bool)
    valid[0, 2:] = True
    kw = dict(positive=False, accuracy=0.75, beta=3.0, eps=1.0)
    st = nft_terms(logits, old, ids, labels, valid, torch.randn(V, d), **kw)
    g_st, = torch.autograd.grad(st['token'] + st['ar'], logits)
    clip = nft_terms(logits, old, ids, labels, valid, torch.randn(V, d), dual_clip=3.0, **kw)
    g_clip, = torch.autograd.grad(clip['token'] + clip['ar'], logits)
    assert g_st.abs().max() > 1e6 and g_clip.abs().max() == 0
    assert clip['token_clip'] == 4 and clip['ar_clip'] == 4


@pytest.mark.parametrize('positive', [True, False])
def test_geo_mix_shares_one_distribution_between_ce_and_embedding(positive):
    """mix="geo": the token CE is -log p±(x) and the embedding term is
    ‖p± E - E[x]‖² with the same p± = softmax(logit mixture); "logit" keeps
    the linear x̂ mixture."""
    from src.rl.nft import logit_mix, nft_terms
    torch.manual_seed(0)
    L, V, d = 6, 11, 4
    ids = torch.randint(0, V, (1, L))
    labels = ids.clone()
    labels[0, :2] = -100
    logits, old = torch.randn(1, 2 * L, V), torch.randn(1, 2 * L, V)
    valid = torch.zeros(1, L, dtype=torch.bool)
    valid[0, 2:] = True
    E = torch.randn(V, d)
    beta = 0.5 if positive else 3.0
    kw = dict(positive=positive, accuracy=0.75, beta=beta, eps=1.0)
    geo = nft_terms(logits, old, ids, labels, valid, E, mix='geo', **kw)
    logit = nft_terms(logits, old, ids, labels, valid, E, mix='logit', **kw)
    x = ids[0, 2:]
    lp = logit_mix(logits[0, L:][valid[0]], old[0, L:][valid[0]], beta, positive)
    torch.testing.assert_close(geo['token'], -lp.gather(-1, x[:, None]).sum())
    torch.testing.assert_close(geo['embed'], ((lp.exp() @ E - E[x]) ** 2).sum())
    torch.testing.assert_close(geo['token'], logit['token'])
    torch.testing.assert_close(geo['ar'], logit['ar'])
    assert not torch.allclose(geo['embed'], logit['embed'])


def test_dual_clip_alone_keeps_v2_inside_the_bound():
    """--dual-clip without --ppo-clip changes nothing for negative tokens with
    q <= c (the straight-through floor still applies) and nothing for positives."""
    from src.rl.nft import nft_terms
    torch.manual_seed(0)
    L, V, d = 6, 11, 4
    ids = torch.randint(0, V, (1, L))
    labels = ids.clone()
    labels[0, :2] = -100
    old = torch.randn(1, 2 * L, V)
    logits = (old + 0.1 * torch.randn_like(old)).requires_grad_(True)  # q near 1
    valid = torch.zeros(1, L, dtype=torch.bool)
    valid[0, 2:] = True
    E = torch.randn(V, d)
    for positive in (True, False):
        kw = dict(positive=positive, accuracy=0.75, beta=1.0 if positive else 3.0, eps=1.0)
        base = nft_terms(logits, old, ids, labels, valid, E, **kw)
        dual = nft_terms(logits, old, ids, labels, valid, E, dual_clip=3.0, **kw)
        g0, = torch.autograd.grad(base['token'] + base['ar'], logits)
        g1, = torch.autograd.grad(dual['token'] + dual['ar'], logits)
        torch.testing.assert_close(g0, g1)
        assert dual['token_clip'] == 0 and dual['ar_clip'] == 0
