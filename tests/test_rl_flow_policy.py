import pytest
import torch
from transformers import Qwen2Config

from src.lafa.configuration_lafa import LafaConfig
from src.lafa.flow_model import FlowModelWrapper
from src.lafa.modeling_lafa import LafaForCausalLM
from src.rl.flow_policy import packed_attention_mask, packed_step_means, rollout, transition_logprob
from src.rl.loss import clipped_pg, old_logprobs, sequence_loss


def tiny_model(self_cond):
    bc = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                     num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                     max_position_embeddings=128, tie_word_embeddings=False,
                     eos_token_id=None, pad_token_id=0)
    torch.manual_seed(42)
    model = LafaForCausalLM(LafaConfig(
        backbone_config=bc.to_dict(), noise_sigma=0.02,
        flow_config={'bd_size': 4, 'enable_self_cond': self_cond, 'freeze_embed': False,
                     'freeze_lm_head': False, 'ar_loss': True,
                     'time_warp_num_bins': 8, 'time_warp_per_position': True})).eval()
    model._ensure_time_warper('cpu').loss_ema.copy_(torch.rand(4, 8) + 0.5)
    if self_cond:  # zero-initialized; give them weight so self-conditioning matters
        for lin in (model.sc_lin_a, model.sc_lin_b):
            torch.nn.init.normal_(lin.weight, std=0.3)
    model.noise_scale_mult = 0.1
    return model


SETTINGS = dict(max_new_tokens=10, num_steps=3, sde_gamma=0.5, seed=7)


@pytest.mark.parametrize('self_cond', [False, True])
def test_rollout_matches_generate(self_cond):
    model = tiny_model(self_cond)
    prompt = torch.tensor([2, 3, 4])
    ro = rollout(model, prompt, **SETTINGS)
    expected, _ = FlowModelWrapper.generate(
        model, prompt[None], max_new_tokens=10, num_steps=3, grid_mode='warp',
        sde_gamma=0.5, sde_vp=True, cfg_scale=1.0, seed=7)
    assert torch.equal(ro.response_ids, expected[0, 3:])
    assert [b.start for b in ro.blocks] == [3, 7, 11]
    assert sum(len(b.tokens) for b in ro.blocks) == 10


@pytest.mark.parametrize('self_cond', [False, True])
def test_batched_rollouts_are_distinct_and_match_generate_batched(self_cond):
    from src.rl.flow_policy import batched_rollouts
    model = tiny_model(self_cond)
    model.noise_scale_mult = 100.0  # at 0.1 the tiny model's answers ignore the noise
    prompts = [torch.tensor([2, 3, 4])] * 4 + [torch.tensor([5, 6, 7, 8, 9])] * 3
    ros = batched_rollouts(model, prompts, max_new_tokens=10, num_steps=3, seed=7,
                           batch_size=4)
    assert [torch.equal(r.prompt_ids, p) for r, p in zip(ros, prompts)] == [True] * 7
    # each row has its own initial noise: repeats of a prompt differ (with
    # shared noise all 4 are identical)
    assert len({tuple(r.response_ids.tolist()) for r in ros[:4]}) == 4
    per_block = 3 + int(self_cond)
    for r in ros:
        assert 0 < r.response_ids.numel() <= 10
        assert r.num_forwards == -(-r.response_ids.numel() // 4) * per_block
    # the second chunk (3 rows, seed 7 * 1_000_003 + 1) is generate_batched's output
    ids = torch.tensor([[5, 6, 7, 8, 9]] * 3)
    seqs, _ = model.generate_batched(ids, torch.ones_like(ids), max_new_tokens=10,
                                     num_steps=3, seed=7 * 1_000_003 + 1, grid_mode='warp',
                                     independent_noise=True)
    assert [r.response_ids.tolist() for r in ros[4:]] == [s[5:] for s in seqs]


def test_batched_rollouts_do_not_replay_their_noise():
    """The caller's next random draws (NFT's forward-process noise) must not
    repeat the noise the rollouts were decoded from."""
    from src.rl.flow_policy import batched_rollouts
    model = tiny_model(False)
    batched_rollouts(model, [torch.tensor([2, 3, 4])] * 4, max_new_tokens=8, num_steps=3,
                     seed=5, batch_size=4)
    after = torch.randn(4 * 4 * 16)
    first = torch.randn(4, 4, 16, generator=torch.Generator().manual_seed(5 * 1_000_003))
    assert not torch.equal(after, first.flatten())


def test_packed_attention_mask():
    mask = packed_attention_mask(5, [(2, 2), (4, 2)], torch.float32, 'cpu') == 0
    assert mask[:5, :5].equal(torch.tril(torch.ones(5, 5, dtype=torch.bool)))
    assert not mask[:5, 5:].any()  # clean never sees noisy slots
    assert mask[5:7].nonzero()[:, 1].tolist() == [0, 1, 5, 6, 0, 1, 5, 6]
    assert mask[7:9].nonzero()[:, 1].tolist() == [0, 1, 2, 3, 7, 8, 0, 1, 2, 3, 7, 8]


@pytest.mark.parametrize('self_cond', [False, True])
def test_packed_means_match_rollout(self_cond):
    model = tiny_model(self_cond)
    ro = rollout(model, torch.tensor([2, 3, 4]), **SETTINGS)
    steps = list(range(SETTINGS['num_steps'] - 1))
    with torch.no_grad():
        means, clean_logits = packed_step_means(model, ro, steps)
    for i, k in enumerate(steps):
        for b, blk in enumerate(ro.blocks):
            torch.testing.assert_close(means[i, b], blk.means[k], rtol=1e-5, atol=1e-6)
        old = torch.stack([blk.means[k] for blk in ro.blocks])
        log_ratio = transition_logprob(ro, k, means[i]) - transition_logprob(ro, k, old)
        assert log_ratio.abs().max() < 1e-2
    # Clean half is the AR head on the full sequence.
    ids = torch.cat([ro.prompt_ids, ro.response_ids])
    assert clean_logits.shape == (ids.numel(), 32)


def test_packed_means_carry_gradients():
    model = tiny_model(True)
    ro = rollout(model, torch.tensor([2, 3, 4]), **SETTINGS)
    means, clean_logits = packed_step_means(model, ro, [0, 1])
    loss = -transition_logprob(ro, 0, means[0]).sum() + clean_logits.logsumexp(-1).mean()
    loss.backward()
    assert model.lm_head.weight.grad.abs().sum() > 0
    assert model.sc_lin_b.weight.grad.abs().sum() > 0


def test_ar_token_logprobs_match_plain_forward():
    model = tiny_model(True)
    ro = rollout(model, torch.tensor([2, 3, 4]), **SETTINGS)
    _, ar = old_logprobs(model, ro, [0, 1])
    ids = torch.cat([ro.prompt_ids, ro.response_ids])[None]
    with torch.no_grad():
        logits, _, _ = model._forward_backbone(
            model.embed_tokens(ids), model._compute_t_bn(torch.ones(ids.shape)), attn_mask=None)
    expected = logits[0, 2:-1].float().log_softmax(-1).gather(-1, ro.response_ids[:, None])[:, 0]
    torch.testing.assert_close(ar, expected, rtol=1e-5, atol=1e-5)
    # K=1 has no stochastic transitions; only the AR term remains.
    flow, ar_k1 = old_logprobs(model, ro, [])
    assert flow.shape == (0, len(ro.blocks))
    torch.testing.assert_close(ar_k1, ar)


def test_sequence_loss_is_advantage_weighted_policy_gradient():
    model = tiny_model(True)
    ro = rollout(model, torch.tensor([2, 3, 4]), **SETTINGS)
    old = old_logprobs(model, ro, [0, 1])
    loss, stats = sequence_loss(model, ro, 1.5, old, [0, 1], ar_weight=0.5)
    # At the old policy the ratios are 1, so the loss is -A (1 + ar_weight).
    assert abs(loss.item() + 1.5 * 1.5) < 1e-4
    assert stats['flow_clip_frac'] == 0 and stats['ar_clip_frac'] == 0
    loss.backward()
    assert model.lm_head.weight.grad.abs().sum() > 0
    # Negative advantage flips the gradient.
    grad = model.lm_head.weight.grad.clone()
    model.zero_grad()
    sequence_loss(model, ro, -1.5, old, [0, 1], ar_weight=0.5)[0].backward()
    torch.testing.assert_close(model.lm_head.weight.grad, -grad)


def test_clipped_pg():
    old = torch.zeros(3)
    logp = torch.log(torch.tensor([0.5, 1.0, 2.0]))
    torch.testing.assert_close(clipped_pg(logp, old, 1.0, 0.2), -torch.tensor([0.5, 1.0, 1.2]))
    torch.testing.assert_close(clipped_pg(logp, old, -1.0, 0.2), torch.tensor([0.8, 1.0, 2.0]))


def test_gsm8k_reward():
    from src.rl.grpo import gsm8k_reward
    gold = 'She pays 3 * 6 = 18 dollars.\n#### 1,800'
    assert gsm8k_reward('So 1800 in total. The answer is $1,800.', gold) == 1.0
    assert gsm8k_reward('The answer is 18.', gold) == 0.0
    assert gsm8k_reward('The answer is 1800.\nQ: next question 5', gold) == 1.0
    assert gsm8k_reward('no number', gold) == 0.0


@pytest.mark.parametrize('minibatches', [1, 3])
def test_grpo_step_updates_trainable_params(monkeypatch, minibatches):
    import itertools
    from src.rl import grpo

    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return ''

    rewards = itertools.cycle([1.0, 0.0, 0.0])  # a mixed group
    monkeypatch.setattr(grpo, 'gsm8k_reward', lambda response, answer: next(rewards))
    model = tiny_model(True)
    model.embed_tokens.weight.requires_grad_(False)
    model.lm_head.weight.requires_grad_(False)
    opt = grpo.MasterAdamW(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in opt.params]
    q = dict(prompt_ids=torch.tensor([2, 3, 4]), answer='')
    steps, olds = [], []
    step, loss_fn = opt.step, grpo.sequence_loss
    monkeypatch.setattr(opt, 'step', lambda *a: steps.append(1) or step(*a))
    monkeypatch.setattr(grpo, 'sequence_loss',
                        lambda m, ro, a, old, *args, **kw: olds.append(old) or loss_fn(
                            m, ro, a, old, *args, **kw))
    sums = grpo.grpo_step(model, opt, Tok(), [q], group_size=6, num_steps=3, sde_gamma=0.5,
                          max_new_tokens=8, clip=0.2, ar_weight=1.0, max_grad_norm=1.0, seed=1,
                          minibatches=minibatches)
    assert sums['count'] == 6 and sums['mixed_groups'] == 1 and sums['trained'] == 6
    # One optimizer step per minibatch; off-policy minibatches use the
    # log-probs from before the first step.
    assert len(steps) == minibatches
    assert all((o is None) == (minibatches == 1) for o in olds)
    assert 0.0 <= sums['flow_clip_frac'] <= 6 and 0.0 <= sums['ar_clip_frac'] <= 6
    assert sums['reward'] == 2.0
    assert any(not torch.equal(b, p) for b, p in zip(before, opt.params))
    assert all(torch.equal(m.to(opt.params[i].dtype), opt.params[i]) for i, m in opt.masters.items())
    assert model.embed_tokens.weight.grad is None


def test_master_adamw_checkpoint_roundtrip(tmp_path):
    from src.rl.grpo import MasterAdamW

    def trained_model():
        model = tiny_model(True)
        opt = MasterAdamW(model.parameters(), lr=1e-2)
        for p in opt.params:
            p.grad = torch.randn_like(p)
        opt.accumulate()
        opt.step(1.0)
        return model, opt

    model, opt = trained_model()
    torch.save(opt.shard_state(), tmp_path / 'optim-rank000-of-001.pt')
    fresh = tiny_model(True)
    restored = MasterAdamW(fresh.parameters(), lr=1e-2)
    restored.load_shards([tmp_path / 'optim-rank000-of-001.pt'])
    for i in opt.owned:
        assert torch.equal(restored.masters[i], opt.masters[i])
        assert torch.equal(restored.params[i], opt.params[i])
        a, b = opt.opt.state[opt.masters[i]], restored.opt.state[restored.masters[i]]
        assert torch.equal(a['exp_avg'], b['exp_avg']) and torch.equal(a['step'], b['step'])
    # The next step matches too.
    for o in (opt, restored):
        torch.manual_seed(0)
        for p in o.params:
            p.grad = torch.randn_like(p)
        o.accumulate()
        o.step(1.0)
    assert all(torch.equal(restored.masters[i], opt.masters[i]) for i in opt.owned)
