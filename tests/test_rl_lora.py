import itertools

import torch

from src.lafa.modeling_lafa import LafaForCausalLM
from src.rl.lora import (TARGETS, LoRALinear, add_lora, load_adapters, merge_lora,
                         merged_state_dict, save_adapters, unmerge_lora)
from tests.test_rl_flow_policy import tiny_model


def clean_logits(model, ids):
    return model(input_ids=ids).logits


def test_lora_starts_at_the_base_model_and_trains_only_adapters():
    base, model = tiny_model(True), tiny_model(True)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    adapted = add_lora(model, rank=4, alpha=8)
    assert len(adapted) == 2 * len(TARGETS)  # 2 layers
    assert all(isinstance(m, LoRALinear) for m in adapted.values())
    torch.testing.assert_close(clean_logits(model, ids), clean_logits(base, ids))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(('lora_A', 'lora_B')) for n in trainable)
    # Base weights keep their names; only adapter tensors are new.
    assert set(base.state_dict()) == {k for k in model.state_dict()
                                      if not k.endswith(('lora_A', 'lora_B'))}


def test_same_seed_gives_same_adapters():
    a, b = tiny_model(False), tiny_model(False)
    for m in (a, b):
        torch.manual_seed(3)
        add_lora(m, rank=4, alpha=8)
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p, q), n


def test_merged_export_loads_as_a_plain_model_with_the_same_outputs(tmp_path):
    model = tiny_model(True)
    add_lora(model, rank=4, alpha=8)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(std=5.0)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    model.save_pretrained(tmp_path, state_dict=merged_state_dict(model))
    loaded = LafaForCausalLM.from_pretrained(tmp_path).eval()
    loaded.noise_scale_mult = model.noise_scale_mult  # a runtime setting, not saved
    assert not any(isinstance(m, LoRALinear) for m in loaded.modules())
    assert not any('lora' in k for k in loaded.state_dict())
    torch.testing.assert_close(clean_logits(loaded, ids), clean_logits(model, ids),
                               atol=1e-5, rtol=1e-4)
    prompt = torch.tensor([[2, 3, 4]])
    kw = dict(max_new_tokens=8, num_steps=3, grid_mode='warp', seed=1)
    assert torch.equal(loaded.generate_flow(prompt, **kw)[0], model.generate_flow(prompt, **kw)[0])


def test_nft_step_with_lora_moves_only_adapters():
    from src.rl.grpo import MasterAdamW
    from src.rl.nft import OldPolicyEMA, nft_step

    class Tok:
        pad_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return ''

    model, old = tiny_model(False), tiny_model(False)
    for m in (model, old):
        torch.manual_seed(0)
        add_lora(m, rank=4, alpha=8)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = MasterAdamW(model.parameters(), lr=1e-2)
    ema = OldPolicyEMA(opt, model, old)
    rewards = itertools.cycle([1.0, 0.0, 0.0])
    q = dict(prompt_ids=torch.tensor([2, 3, 4, 5, 6]), answer='')
    nft_step(model, old, ema, opt, Tok(), [q], group_size=6, num_steps=3, max_new_tokens=12,
             minibatches=2, beta=1.0, eps=1.0, embed_weight=1.0, ar_weight=1.0,
             noise_scale=4.0, self_cond_prob=0.5, max_grad_norm=1.0, eta=0.5, seed=1,
             reward_fn=lambda r, a: next(rewards), truncated_negative=False)
    moved = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert moved and all(n.endswith(('lora_A', 'lora_B')) for n in moved)
    # θ_old follows the adapters too.
    assert any(not torch.equal(p, before[n]) for n, p in old.named_parameters()
               if n.endswith('lora_B'))


def test_saved_adapters_rebuild_the_same_model(tmp_path):
    model = tiny_model(True)
    torch.manual_seed(0)
    add_lora(model, rank=4, alpha=8)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(std=5.0)
    save_adapters(model, tmp_path, base_model='unused', rank=4, alpha=8)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['adapter.json', 'adapter.safetensors']
    rebuilt = tiny_model(True)
    torch.manual_seed(1)  # a different A init: the saved A must overwrite it
    load_adapters(rebuilt, tmp_path)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    torch.testing.assert_close(clean_logits(rebuilt, ids), clean_logits(model, ids))


def test_lora_merge_script_writes_the_merged_model(tmp_path, monkeypatch):
    import sys

    from scripts.lora_merge import main
    base = tiny_model(True)
    base.save_pretrained(tmp_path / 'base')
    model = tiny_model(True)
    add_lora(model, rank=4, alpha=8)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(std=5.0)
    export = tmp_path / 'step-0025'
    save_adapters(model, export, base_model=str(tmp_path / 'base'), rank=4, alpha=8)

    class Tok:
        def save_pretrained(self, path):
            pass

    monkeypatch.setattr('scripts.lora_merge.AutoTokenizer.from_pretrained', lambda *a, **k: Tok())
    monkeypatch.setattr(sys, 'argv', ['lora_merge.py', str(export), '--dtype', 'float32'])
    main()
    merged = LafaForCausalLM.from_pretrained(tmp_path / 'step-0025-merged').eval()
    assert not any('lora' in k for k in merged.state_dict())
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    # the adapters move these logits by ~0.3
    torch.testing.assert_close(clean_logits(merged, ids), clean_logits(model, ids),
                               atol=1e-5, rtol=1e-4)


def unmerged_logits(model, ids):
    saved = {m: m.lora_merged for m in model.modules() if isinstance(m, LoRALinear)}
    for m in saved:
        m.lora_merged = None
    out = clean_logits(model, ids)
    for m, w in saved.items():
        m.lora_merged = w
    return out


def test_merged_forward_matches_the_adapter_forward_and_refreshes():
    model = tiny_model(True)
    add_lora(model, rank=4, alpha=8)
    layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    with torch.no_grad():
        for m in layers:
            m.lora_B.normal_(std=5.0)
    merge_lora(model)
    base_w = [m.weight.clone() for m in layers]
    torch.testing.assert_close(clean_logits(model, ids), unmerged_logits(model, ids),
                               atol=1e-5, rtol=1e-4)
    with torch.no_grad():  # new adapters: the merged weights follow on the next merge
        for m in layers:
            m.lora_B.normal_(std=5.0)
    stale = clean_logits(model, ids)
    merge_lora(model)
    assert not torch.allclose(stale, clean_logits(model, ids), atol=1e-3)
    torch.testing.assert_close(clean_logits(model, ids), unmerged_logits(model, ids),
                               atol=1e-5, rtol=1e-4)
    assert all(torch.equal(m.weight, w) for m, w in zip(layers, base_w))  # W untouched
    merged = clean_logits(model, ids)
    unmerge_lora(model)
    assert all(m.lora_merged is None for m in layers)
    out = clean_logits(model, ids)
    torch.testing.assert_close(out, merged, atol=1e-5, rtol=1e-4)
    out.sum().backward()  # the adapter forward again: gradients reach the adapters
    assert all(m.lora_B.grad is not None for m in layers)


def test_old_policy_ema_keeps_the_merged_old_model_in_sync():
    from src.rl.grpo import MasterAdamW
    from src.rl.nft import OldPolicyEMA

    model, old = tiny_model(False), tiny_model(False)
    for m in (model, old):
        torch.manual_seed(0)
        add_lora(m, rank=4, alpha=8)
    opt = MasterAdamW(model.parameters(), lr=1e-2)
    ema = OldPolicyEMA(opt, model, old)
    assert all(m.lora_merged is not None for m in old.modules() if isinstance(m, LoRALinear))
    assert all(m.lora_merged is None for m in model.modules() if isinstance(m, LoRALinear))
    with torch.no_grad():
        for i, master in opt.masters.items():
            master.add_(torch.randn_like(master))
    ema.update(0.0)  # θ_old ← θ (the masters)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    torch.testing.assert_close(clean_logits(old, ids), unmerged_logits(old, ids),
                               atol=1e-5, rtol=1e-4)
    assert not torch.allclose(clean_logits(old, ids), clean_logits(tiny_model(False), ids),
                              atol=1e-3)
