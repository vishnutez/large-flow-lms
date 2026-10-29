import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from src.rl.tilt import (link_inverse, log_link, onpolicy_scale, solve_b, tilt_logprobs,
                         tilt_terms)
from tests.test_rl_ppo_lafa import case, lafa_loss


@pytest.mark.parametrize('b_mode,kappa', list(itertools.product(['solve', 'fixed'], [1.0, 16.0])))
def test_pair_is_valid_and_mixes_to_p_old_off_policy(b_mode, kappa):
    torch.manual_seed(0)
    lp_old = F.log_softmax(torch.randn(5, 13, dtype=torch.float64), -1)
    lp = F.log_softmax(5.0 * torch.randn(5, 13, dtype=torch.float64), -1)  # far off-policy
    acc = 0.3
    lp_pos, lp_neg, b = tilt_logprobs(lp, lp_old, 0.7, acc, b_mode, kappa)
    for l in (lp_pos, lp_neg):
        assert torch.isfinite(l).all()
        torch.testing.assert_close(l.exp().sum(-1), torch.ones(5, dtype=torch.float64))
    sig = log_link(0.7 * (lp - lp_old) + b[:, None], kappa)[0].exp()
    alpha = (lp_old.exp() * sig).sum(-1, keepdim=True)
    if b_mode == 'solve':
        torch.testing.assert_close(alpha[:, 0], torch.full((5,), acc, dtype=torch.float64))
    torch.testing.assert_close(alpha * lp_pos.exp() + (1 - alpha) * lp_neg.exp(), lp_old.exp())


@pytest.mark.parametrize('kappa', [1.0, 16.0])
def test_solve_b_gradient_is_the_implicit_one(kappa):
    torch.manual_seed(1)
    lp_old = F.log_softmax(torch.randn(3, 9, dtype=torch.float64), -1)
    lp = F.log_softmax(torch.randn(3, 9, dtype=torch.float64), -1).requires_grad_(True)
    acc, beta = 0.4, 1.3
    w = torch.randn(3, dtype=torch.float64)
    _, _, b = tilt_logprobs(lp, lp_old, beta, acc, 'solve', kappa)
    g, = torch.autograd.grad((w * b).sum(), lp)
    h = 1e-6
    fd = torch.zeros_like(lp)
    for i in range(3):
        for j in range(9):
            d = torch.zeros_like(lp)
            d[i, j] = h
            up = solve_b(beta * (lp.detach() + d - lp_old), lp_old, math.log(acc), kappa)
            dn = solve_b(beta * (lp.detach() - d - lp_old), lp_old, math.log(acc), kappa)
            fd[i, j] = (w * (up - dn)).sum() / (2 * h)
    torch.testing.assert_close(g, fd, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('positive,b_mode,kappa',
                         list(itertools.product([True, False], ['solve', 'fixed'], [1.0, 16.0])))
def test_onpolicy_gradient_is_beta_advantage_weighted_lafa_loss(positive, b_mode, kappa):
    ids, labels, valid, E, old = case()
    logits = old.clone().requires_grad_(True)
    acc, beta = 0.375, 0.8
    t = tilt_terms(logits, old, ids, labels, valid, E, positive=positive, accuracy=acc,
                   beta=beta, eps=1.0, b_mode=b_mode, kappa=kappa)
    g, = torch.autograd.grad(t['token'] + t['embed'] + t['ar'], logits)
    flow, ar = lafa_loss(logits, ids, labels, valid, E)
    adv = (1.0 if positive else 0.0) - acc
    g_ref, = torch.autograd.grad(beta * adv * (flow + ar), logits)
    torch.testing.assert_close(g, g_ref, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('kappa', [1.0, 16.0, 64.0])
def test_terms_stay_finite_far_off_policy(kappa):
    ids, labels, valid, E, old = case(seed=2)
    logits = (old + 30.0 * torch.randn_like(old)).requires_grad_(True)
    for positive in (True, False):
        t = tilt_terms(logits, old, ids, labels, valid, E, positive=positive, accuracy=0.875,
                       beta=1.0, eps=1.0, kappa=kappa)
        g, = torch.autograd.grad(t['token'] + t['embed'] + t['ar'], logits)
        assert all(math.isfinite(t[k].item()) for k in ('token', 'embed', 'ar'))
        assert torch.isfinite(g).all()


def test_log_link_is_stable_and_matches_float64():
    s = torch.linspace(-8.0, 8.0, 401, dtype=torch.float64)
    for kappa in (1.0, 4.0, 16.0, 64.0):
        z = kappa * s
        log_f = F.logsigmoid(z) / kappa
        ref = torch.log(-torch.expm1(log_f))
        got_f, got_1mf = log_link(s.float(), kappa)
        torch.testing.assert_close(got_f.double(), log_f, atol=1e-5, rtol=1e-5)
        ok = torch.isfinite(ref)
        torch.testing.assert_close(got_1mf.double()[ok], ref[ok], atol=1e-4, rtol=1e-4)
        assert torch.isfinite(got_1mf).all()
    # Past float32's range: 1 - F ≈ e^(-κs)/κ, still finite with a gradient.
    s = torch.tensor([10.0, 50.0], requires_grad=True)
    _, l1mf = log_link(s, 16.0)
    torch.testing.assert_close(l1mf.detach(), -16.0 * s.detach() - math.log(16.0))
    g, = torch.autograd.grad(l1mf.sum(), s)
    torch.testing.assert_close(g, torch.full((2,), -16.0))
    assert abs(link_inverse(0.3, 1.0) - math.log(0.3 / 0.7)) < 1e-12


def test_large_kappa_pair_is_v2_inside_the_feasible_region():
    # κ → ∞: p⁺ = p_θ, p⁻ = (p_old - r̂ p_θ)/(1 - r̂), while p_θ < p_old/r̂.
    torch.manual_seed(3)
    acc = 0.6
    lp_old = F.log_softmax(torch.randn(4, 11, dtype=torch.float64), -1)
    q = torch.exp(0.3 * torch.randn(4, 11, dtype=torch.float64))
    q = q.clamp(max=0.95 / acc)
    p = lp_old.exp() * q
    p = p / p.sum(-1, keepdim=True)
    assert (p < 0.97 * lp_old.exp() / acc).all()
    for b_mode in ('solve', 'fixed'):
        lp_pos, lp_neg, _ = tilt_logprobs(p.log(), lp_old, 1.0, acc, b_mode, kappa=2000.0)
        torch.testing.assert_close(lp_pos.exp(), p, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(lp_neg.exp(), (lp_old.exp() - acc * p) / (1 - acc),
                                   atol=1e-6, rtol=1e-5)


def test_tilt_step_trains():
    from src.rl.grpo import MasterAdamW
    from src.rl.nft import OldPolicyEMA, nft_step
    from tests.test_rl_nft import pair

    class Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return ''

    model, old = pair()
    opt = MasterAdamW(model.parameters(), lr=1e-3)
    ema = OldPolicyEMA(opt, model, old)
    before = [p.detach().clone() for p in opt.params]
    rewards = itertools.cycle([1.0, 0.0, 0.0])
    q = dict(prompt_ids=torch.tensor([2, 3, 4, 5, 6]), answer='')
    sums = nft_step(model, old, ema, opt, Tok(), [q], group_size=6, num_steps=3,
                    max_new_tokens=12, minibatches=2, beta=1.0, eps=1.0, embed_weight=1.0,
                    ar_weight=1.0, noise_scale=4.0, self_cond_prob=0.5, max_grad_norm=1.0,
                    eta=0.5, seed=1, reward_fn=lambda r, a: next(rewards),
                    truncated_negative=False, terms_fn=tilt_terms, weighting='adv')
    assert sums['kept'] == 6 and sums['skipped_answers'] == 0
    assert any(not torch.equal(b, p) for b, p in zip(before, opt.params))


# The complementary log-log link, F(s) = 1 - exp(-eˢ).

@pytest.mark.parametrize('b_mode', ['solve', 'fixed'])
def test_cloglog_pair_is_valid_and_mixes_to_p_old_off_policy(b_mode):
    torch.manual_seed(0)
    lp_old = F.log_softmax(torch.randn(5, 13, dtype=torch.float64), -1)
    lp = F.log_softmax(5.0 * torch.randn(5, 13, dtype=torch.float64), -1)
    acc = 0.3
    lp_pos, lp_neg, b = tilt_logprobs(lp, lp_old, 0.7, acc, b_mode, link='cloglog')
    for l in (lp_pos, lp_neg):
        assert torch.isfinite(l).all()
        torch.testing.assert_close(l.exp().sum(-1), torch.ones(5, dtype=torch.float64))
    sig = log_link(0.7 * (lp - lp_old) + b[:, None], 1.0, 'cloglog')[0].exp()
    alpha = (lp_old.exp() * sig).sum(-1, keepdim=True)
    if b_mode == 'solve':
        torch.testing.assert_close(alpha[:, 0], torch.full((5,), acc, dtype=torch.float64))
    torch.testing.assert_close(alpha * lp_pos.exp() + (1 - alpha) * lp_neg.exp(), lp_old.exp())


def test_cloglog_solve_b_gradient_is_the_implicit_one():
    torch.manual_seed(1)
    lp_old = F.log_softmax(torch.randn(3, 9, dtype=torch.float64), -1)
    lp = F.log_softmax(torch.randn(3, 9, dtype=torch.float64), -1).requires_grad_(True)
    acc, beta = 0.4, 1.3
    w = torch.randn(3, dtype=torch.float64)
    _, _, b = tilt_logprobs(lp, lp_old, beta, acc, 'solve', link='cloglog')
    g, = torch.autograd.grad((w * b).sum(), lp)
    h = 1e-6
    fd = torch.zeros_like(lp)
    for i in range(3):
        for j in range(9):
            d = torch.zeros_like(lp)
            d[i, j] = h
            up = solve_b(beta * (lp.detach() + d - lp_old), lp_old, math.log(acc), link='cloglog')
            dn = solve_b(beta * (lp.detach() - d - lp_old), lp_old, math.log(acc), link='cloglog')
            fd[i, j] = (w * (up - dn)).sum() / (2 * h)
    torch.testing.assert_close(g, fd, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('positive,b_mode', list(itertools.product([True, False], ['solve', 'fixed'])))
def test_cloglog_onpolicy_gradient_is_beta_advantage_weighted_lafa_loss(positive, b_mode):
    ids, labels, valid, E, old = case()
    logits = old.clone().requires_grad_(True)
    acc, beta = 0.375, 0.8
    t = tilt_terms(logits, old, ids, labels, valid, E, positive=positive, accuracy=acc,
                   beta=beta, eps=1.0, b_mode=b_mode, link='cloglog', neg_clip=5.0)
    g, = torch.autograd.grad(t['token'] + t['embed'] + t['ar'], logits)
    flow, ar = lafa_loss(logits, ids, labels, valid, E)
    adv = (1.0 if positive else 0.0) - acc
    g_ref, = torch.autograd.grad(beta * adv * (flow + ar), logits)
    torch.testing.assert_close(g, g_ref, atol=1e-5, rtol=1e-4)


def test_cloglog_negative_push_is_v2s_and_the_clip_stops_it():
    # One slot, b fixed: d(-log p⁻(x) · scale)/d log q(x) → r̂ q^β as the
    # token's share of α vanishes (v2: r̂ q for q > 1, β = 1).
    V, acc = 2000, 0.8
    lp_old = torch.full((1, V), -math.log(V), dtype=torch.float64)
    scale = onpolicy_scale(acc, 1.0, 'cloglog')
    for q in (0.5, 1.0, 2.0, 4.0):
        lq = torch.zeros(1, V, dtype=torch.float64)
        lq[0, 0] = math.log(q)
        lq.requires_grad_(True)
        _, lp_neg, _ = tilt_logprobs(lq, torch.zeros_like(lq), 1.0, acc, 'fixed', link='cloglog')
        # lp = lq here (p_old uniform cancels in the ratio); use lp_old = 0 rows.
        lp_neg = lp_neg + lp_old
        g, = torch.autograd.grad(-scale * lp_neg[0, 0], lq)
        assert abs(g[0, 0].item() - acc * q) < 2e-3 * max(1.0, q)
    # The clip: no token/AR gradient for a negative's token with q > c.
    ids, labels, valid, E, old = case()
    logits = (old + 3.0 * torch.randn_like(old)).requires_grad_(True)
    t = tilt_terms(logits, old, ids, labels, valid, E, positive=False, accuracy=acc,
                   beta=1.0, eps=1.0, link='cloglog', neg_clip=1e-6)
    g, = torch.autograd.grad(t['token'] + t['ar'], logits)
    assert t['token_clip'] == valid.sum().item() and torch.count_nonzero(g) == 0


def test_cloglog_terms_stay_finite_far_off_policy():
    ids, labels, valid, E, old = case(seed=2)
    logits = (old + 30.0 * torch.randn_like(old)).requires_grad_(True)
    for positive in (True, False):
        t = tilt_terms(logits, old, ids, labels, valid, E, positive=positive, accuracy=0.875,
                       beta=1.0, eps=1.0, link='cloglog', neg_clip=5.0)
        g, = torch.autograd.grad(t['token'] + t['embed'] + t['ar'], logits)
        assert all(math.isfinite(t[k].item()) for k in ('token', 'embed', 'ar'))
        assert torch.isfinite(g).all()
