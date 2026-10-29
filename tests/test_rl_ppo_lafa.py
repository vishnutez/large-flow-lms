import itertools

import pytest
import torch
import torch.nn.functional as F

from src.rl.ppo_lafa import ppo_lafa_terms, ppo_surrogate


def case(seed=0, L=6, V=11, d=4):
    torch.manual_seed(seed)
    ids = torch.randint(0, V, (1, L))
    labels = ids.clone()
    labels[0, :2] = -100
    valid = torch.zeros(1, L, dtype=torch.bool)
    valid[0, 2:] = True
    return ids, labels, valid, torch.randn(V, d), torch.randn(1, 2 * L, V)


def lafa_loss(logits, ids, labels, valid, E):
    """Per-sequence Lafa loss: Σ (CE + MSE) over noised slots, Σ AR."""
    L = ids.shape[1]
    x = ids[0][valid[0]]
    lp = F.log_softmax(logits[0, L:][valid[0]], -1)
    flow = (-lp.gather(-1, x[:, None])[:, 0] + ((lp.exp() @ E - E[x]) ** 2).sum(-1)).sum()
    ar_valid = labels[0, 1:] != -100
    y = ids[0, 1:][ar_valid]
    ar = -F.log_softmax(logits[0, :L - 1][ar_valid], -1).gather(-1, y[:, None]).sum()
    return flow, ar


@pytest.mark.parametrize('positive', [True, False])
def test_onpolicy_gradient_is_advantage_weighted_lafa_loss(positive):
    ids, labels, valid, E, old = case()
    logits = old.clone().requires_grad_(True)
    acc = 0.375
    t = ppo_lafa_terms(logits, old, ids, labels, valid, E, positive=positive, accuracy=acc,
                       beta=1.0, eps=1.0, ppo_clip=0.2, dual_clip=3.0)
    g, = torch.autograd.grad(t['token'] + t['ar'], logits)
    flow, ar = lafa_loss(logits, ids, labels, valid, E)
    adv = (1.0 if positive else 0.0) - acc
    g_ref, = torch.autograd.grad(adv * (flow + ar), logits)
    torch.testing.assert_close(g, g_ref)
    assert t['token_clip'] == 0 and t['ar_clip'] == 0


def test_surrogate_clips_like_ppo_with_dual_clip():
    log_w = torch.log(torch.tensor([0.5, 0.9, 1.1, 1.5, 2.5, 50.0, 1e30])).requires_grad_(True)
    for adv, expect in [(0.5, [1, 1, 1, 0, 0, 0, 0]), (-0.5, [0, 1, 1, 1, 1, 0, 0])]:
        loss, clipped = ppo_surrogate(log_w, adv, 0.2, 3.0)
        g, = torch.autograd.grad(loss.sum(), log_w)
        assert torch.isfinite(g).all() and torch.isfinite(loss).all()
        assert (g != 0).int().tolist() == expect
        assert (~clipped).int().tolist() == expect


def test_ppo_lafa_step_trains():
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
                    truncated_negative=False, terms_fn=ppo_lafa_terms, weighting='adv',
                    ppo_clip=0.2, dual_clip=3.0)
    assert sums['kept'] == 6 and sums['skipped_answers'] == 0
    assert any(not torch.equal(b, p) for b, p in zip(before, opt.params))
