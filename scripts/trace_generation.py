"""Record per-forward decoding trajectories from a Lafa model for visualization.

For every forward pass the tracer stores its role, top-k token distributions,
entropy, head-averaged attention (per layer group, uint8) and, for flow
decoding, the block's embedding trajectory projected to 2D. Tracing only
reads tensors and draws no random numbers; each trace is checked against an
untraced run with the same seed.

    python scripts/trace_generation.py --model lafa-hf/lafa-8B \
        --modes flow,speculative --qids 0,1,2,3,4,5 --output-dir output/traces
"""
import argparse
import base64
import inspect
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IM_END = '<|im_end|>'


def _b64(array):
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).decode()


def _round(values, digits=4):
    return [round(float(v), digits) for v in values]


def _extract_answer(text):
    """lm-eval gsm8k_cot flexible-extract: the last number in the response."""
    matches = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', text)
    if not matches:
        return None
    value = [m for m in matches[-1] if m][0]
    return re.sub(r'[,$]|\.$', '', value)


class Tracer:
    """Temporarily wraps a Lafa model's decoding internals and records each pass."""

    def __init__(self, model, mode, cfg_scale=1.0, top_k=8, num_groups=4):
        self.model, self.mode, self.top_k = model, mode, top_k
        self.use_cfg = abs(cfg_scale - 1.0) > 1e-6
        self.cfg_scale = cfg_scale
        self.N = model.bd_size
        layers = model.backbone.model.layers
        bounds = np.linspace(0, len(layers), num_groups + 1).round().astype(int)
        self.layer_groups = [[int(a), int(b)] for a, b in zip(bounds[:-1], bounds[1:])]
        self.num_layers = len(layers)
        E = model.embed_tokens.weight
        self.E32 = E.detach().float()
        self.En = F.normalize(self.E32, dim=-1)
        self.passes, self.blocks = [], []
        self._handles, self._patched = [], []
        self._rows = None          # query rows whose attention is kept
        self._attn = []            # per-layer [rows, keys] for the current pass
        self._embed_ids = None     # ids of the most recent embed_tokens call
        self._t = None             # most recent source time passed to _compute_t_bn
        self._z = None             # pre-self-conditioning state of the next pass
        self._block = None         # flow block currently being denoised
        self._prev_pred = None     # speculative: AR prediction for the next token
        self._last_draft = None    # speculative: most recent drafted block

    # ---- patching ----
    def __enter__(self):
        m = self.model
        self._orig_fb = m._forward_backbone
        self._sig = inspect.signature(self._orig_fb)
        self._patch('_forward_backbone', self._forward_backbone)
        orig_t = m._compute_t_bn

        def compute_t_bn(t, tgt=False):
            if not tgt:
                self._t = t.detach()
            return orig_t(t, tgt)
        self._patch('_compute_t_bn', compute_t_bn)
        orig_sc = m._apply_self_cond

        def apply_self_cond(e_t, e_prev):
            self._z = e_t.detach()
            return orig_sc(e_t, e_prev)
        self._patch('_apply_self_cond', apply_self_cond)
        orig_denoise = m._euler_denoise_block

        def euler_denoise_block(past_key_values, *args, **kwargs):
            self._start_block(past_key_values.get_seq_length())
            try:
                return orig_denoise(past_key_values, *args, **kwargs)
            finally:
                self._block['open'] = False
        self._patch('_euler_denoise_block', euler_denoise_block)

        def embed_hook(module, inputs, output):
            self._embed_ids = inputs[0].detach().flatten().tolist()
        self._handles.append(m.embed_tokens.register_forward_hook(embed_hook))
        for layer in m.backbone.model.layers:
            self._handles.append(layer.self_attn.register_forward_hook(self._attn_hook))
        return self

    def __exit__(self, *exc):
        for name in self._patched:
            del self.model.__dict__[name]
        for handle in self._handles:
            handle.remove()
        return False

    def _patch(self, name, fn):
        self.model.__dict__[name] = fn
        self._patched.append(name)

    def _attn_hook(self, module, inputs, output):
        if self._rows is None:
            return
        weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if weights is None:
            raise RuntimeError('Attention weights unavailable; eager attention is required')
        self._attn.append(weights[0].float().mean(0)[self._rows])  # [rows, keys]

    # ---- per-pass recording ----
    def _forward_backbone(self, *args, **kwargs):
        bound = self._sig.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        embeds, past, use_cache = a['packed_embeds'], a['past_key_values'], a['use_cache']
        length = embeds.shape[1]
        cache_len = past.get_seq_length() if past is not None else 0
        role = self._classify(length, use_cache, a['attn_mask'], cache_len, past)
        keep_attn = role not in ('prefill', 'commit')
        self._rows = slice(None) if keep_attn else None
        self._attn = []
        z = self._z if self._z is not None and self._z.shape[1] == length else None
        self._z = None
        out = self._orig_fb(*args, **kwargs)
        self._rows = None
        self._record(role, out[0], embeds, z, cache_len, length)
        return out

    def _classify(self, length, use_cache, mask, cache_len, past):
        N = self.N
        if past is None or (use_cache and cache_len == 0):
            return 'prefill'
        if self.mode == 'flow':
            block = self._block
            if block is not None and block['open']:
                i = block['calls']
                block['calls'] += 1
                per_step = 2 if self.use_cfg else 1
                n_init = per_step if self.model.enable_self_cond else 0
                cond = ((i if i < n_init else i - n_init) % per_step) == 0
                kind = 'cond' if cond else 'uncond'
                if i < n_init:
                    return f'sc-init:{kind}'
                return f'step{(i - n_init) // per_step + 1}:{kind}'
            return 'commit'
        if length == N:
            return 'draft' if not use_cache else 'commit'
        if length == N + 1:
            return 'recover' if use_cache else 'recover-sc'
        if length == N * (N + 1):
            return 'fused' if use_cache else 'fused-sc'
        return 'other'

    def _topk(self, logits):
        probs = F.softmax(logits.float(), dim=-1)
        top = probs.topk(self.top_k, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
        return probs, top, entropy

    def _attention(self):
        if not self._attn:
            return None
        stacked = torch.stack(self._attn)  # [layers, rows, keys]
        groups = [stacked.mean(0)] + [stacked[a:b].mean(0) for a, b in self.layer_groups]
        g = torch.stack(groups)  # [1 + groups, rows, keys]
        g = g / g.amax(-1, keepdim=True).clamp_min(1e-12)
        return g

    def _record(self, role, logits, embeds, z, cache_len, length):
        rec = dict(i=len(self.passes), role=role, ctx=cache_len, rows=length)
        if self._t is not None and self._t.shape[-1] == length:
            rec['t'] = _round(self._t.reshape(-1).tolist())
        if role == 'prefill':
            rec['rows'] = 0
            self.passes.append(rec)
            if self.mode == 'speculative':
                self._prev_pred = logits[0, -1]
            return
        if role == 'commit':
            rec['ids'] = list(self._embed_ids or [])
            self.passes.append(rec)
            return
        logits = logits[0]
        probs, top, entropy = self._topk(logits)
        rec['topk'] = [[[int(i), round(float(p), 4)] for i, p in zip(ti, tp)]
                       for ti, tp in zip(top.indices.tolist(), top.values.tolist())]
        rec['entropy'] = _round(entropy.tolist(), 3)
        attn = self._attention()
        if attn is not None:
            rec['keys'] = attn.shape[-1]
            rec['attn'] = _b64((attn.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy())
        if self.mode == 'flow':
            self._record_flow(rec, probs, embeds, z)
        else:
            self._record_speculative(rec, logits)
        self.passes.append(rec)

    # ---- flow decoding ----
    def _start_block(self, cache_len):
        if self._block is not None:
            self._finish_block()
        self._block = dict(start=cache_len, open=True, calls=0, steps=[])

    def _record_flow(self, rec, probs, embeds, z):
        block = self._block
        rec['block'] = len(self.blocks)
        pred = probs @ self.E32  # predicted clean embedding [N, d]
        state = (z if z is not None else embeds)[0].float()
        entry = dict(pass_i=rec['i'], role=rec['role'], z=state, pred=pred,
                     top=[r[:3] for r in rec['topk']])
        if rec['role'].endswith('uncond') and block['steps']:
            cond = block['steps'][-1]
            cond['cfg'] = entry['pred'] + self.cfg_scale * (cond['pred'] - entry['pred'])
        block['steps'].append(entry)

    def _finish_block(self, final_ids=None):
        block = self._block
        self._block = None
        if block is None or not block['steps']:
            return
        self.blocks.append(dict(start=block['start'], steps=block['steps'], final=final_ids))

    def finalize_flow(self, all_ids):
        """Project each block position's trajectory once final tokens are known."""
        if self._block is not None:
            self._finish_block()
        out = []
        for b, block in enumerate(self.blocks):
            positions = []
            for r in range(self.N):
                pos = block['start'] + r
                if pos >= len(all_ids):
                    break
                final_id = all_ids[pos]
                steps = [s for s in block['steps'] if not s['role'].endswith('uncond')]
                cands = []
                for s in steps:
                    for tid, _ in s['top'][r]:
                        if tid not in cands and len(cands) < 12:
                            cands.append(tid)
                if final_id not in cands:
                    cands.append(final_id)
                zs = torch.stack([s['z'][r] for s in steps])
                preds = torch.stack([s['pred'][r] for s in steps])
                cfgs = torch.stack([s.get('cfg', s['pred'])[r] for s in steps])
                cand_e = self.E32[cands]
                pts = torch.cat([zs, preds, cfgs, cand_e]).cpu()
                center = pts.mean(0, keepdim=True)
                _, _, vh = torch.linalg.svd(pts - center, full_matrices=False)
                xy = ((pts - center) @ vh[:2].T).tolist()
                n = len(steps)
                final_e = self.En[final_id]
                nearest = (F.normalize(zs, dim=-1) @ self.En.T).argmax(-1).tolist()
                positions.append(dict(
                    pos=pos, final=final_id,
                    passes=[s['pass_i'] for s in steps], roles=[s['role'] for s in steps],
                    z=[_round(p, 3) for p in xy[:n]],
                    pred=[_round(p, 3) for p in xy[n:2 * n]],
                    cfg=[_round(p, 3) for p in xy[2 * n:3 * n]],
                    cands=[[cid] + _round(p, 3) for cid, p in zip(cands, xy[3 * n:])],
                    z_norm=_round(zs.norm(dim=-1).tolist(), 3),
                    pred_norm=_round(preds.norm(dim=-1).tolist(), 3),
                    z_cos=_round((F.normalize(zs, dim=-1) @ final_e).tolist(), 4),
                    pred_cos=_round((F.normalize(preds, dim=-1) @ final_e).tolist(), 4),
                    nearest=nearest,
                ))
            out.append(dict(start=block['start'], positions=positions))
        return out

    # ---- self-speculative decoding ----
    def _record_speculative(self, rec, logits):
        N, role = self.N, rec['role']
        if role == 'draft':
            rec['draft'] = self._last_draft = logits.argmax(-1).tolist()
            return
        if role == 'recover':
            rejected = self._last_draft[0]
            rec['proposed'] = list(self._last_draft)
            rec['check'] = self._check(rejected)
            rec['accepted'] = [self._embed_ids[0]]
            rec['draft'] = self._last_draft = logits[1:].argmax(-1).tolist()
            self._prev_pred = logits[0]
            return
        if role == 'fused':
            D = list(self._embed_ids)
            draft_idx = [i * (N + 1) for i in range(N)]
            rec['proposed'] = D
            rec['check'] = self._check(D[0])
            verify, j = [], 1
            for i in range(1, N):
                pred = int(logits[draft_idx[i - 1]].argmax())
                verify.append(pred)
                if pred == D[i] and j == i:
                    j += 1
            rec['verify'] = verify
            rec['accepted'] = D[:j]
            rec['window'] = j - 1
            win = [draft_idx[j - 1] + 1 + k for k in range(N)]
            rec['draft'] = self._last_draft = logits[win].argmax(-1).tolist()
            self._prev_pred = logits[draft_idx[j - 1]]

    def _check(self, proposed):
        probs = F.softmax(self._prev_pred.float(), dim=-1)
        top = probs.topk(self.top_k)
        return dict(topk=[[int(i), round(float(p), 4)] for i, p in zip(top.indices, top.values)],
                    proposed_p=round(float(probs[proposed]), 4))


def build_prompt(tokenizer, question):
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': f'Q: {question}\nA:'}],
        tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids


def stop_ids(model, tokenizer):
    """The checkpoint's EOS ids plus <|im_end|> (the 8B config EOS is <|endoftext|>)."""
    eos = model.generation_config.eos_token_id
    eos = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    return eos if im_end in eos else eos + [im_end]


def trace_one(model, tokenizer, prompt_ids, mode, args):
    kwargs = dict(gen_mode=mode, max_new_tokens=args.max_new_tokens, seed=args.seed,
                  noise_scale_override=args.noise_scale_override,
                  eos_token_id=stop_ids(model, tokenizer),
                  return_dict_in_generate=True)
    if mode == 'flow':
        kwargs.update(num_steps=args.num_steps, cfg_scale=args.cfg_scale)
    cfg = args.cfg_scale if mode == 'flow' else 1.0
    start = time.perf_counter()
    with Tracer(model, mode, cfg_scale=cfg, top_k=args.top_k) as tracer:
        traced = model.generate(prompt_ids, **kwargs)
    seconds = time.perf_counter() - start
    reference = model.generate(prompt_ids, **kwargs)
    if not torch.equal(traced.sequences, reference.sequences):
        raise RuntimeError('Tracing changed the generated tokens')
    ids = traced.sequences[0].tolist()
    blocks = tracer.finalize_flow(ids) if mode == 'flow' else []
    return tracer, ids, traced.stats[0], blocks, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='lafa-hf/lafa-8B')
    parser.add_argument('--revision')
    parser.add_argument('--modes', default='flow,speculative')
    parser.add_argument('--qids', default='0,1,2,3,4,5', help='GSM8K test indices')
    parser.add_argument('--num-steps', type=int, default=4)
    parser.add_argument('--cfg-scale', type=float, default=8.0)
    parser.add_argument('--noise-scale-override', type=float, default=0.1)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='output/traces')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from src.lafa.modeling_lafa import LafaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                              revision=args.revision)
    # The repo's model code, not the Hub copy, so list-valued EOS ids work.
    model = LafaForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, revision=args.revision).to(args.device).eval()
    data = load_dataset('openai/gsm8k', 'main', split='test')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model.rstrip('/').split('/')[-1]

    for qid in [int(q) for q in args.qids.split(',')]:
        row = data[qid]
        reference = row['answer'].split('####')[-1].strip().replace(',', '')
        prompt_ids = build_prompt(tokenizer, row['question']).to(args.device)
        for mode in args.modes.split(','):
            tracer, ids, stats, blocks, seconds = trace_one(model, tokenizer, prompt_ids, mode, args)
            prompt_len = prompt_ids.shape[1]
            response = tokenizer.decode(ids[prompt_len:], skip_special_tokens=False)
            response = response.split(IM_END)[0]
            extracted = _extract_answer(response)
            vocab = {i for i in ids}
            for p in tracer.passes:
                for row_top in p.get('topk', []):
                    vocab.update(i for i, _ in row_top)
                vocab.update(i for i, _ in p.get('check', {}).get('topk', []))
                for key in ('draft', 'proposed', 'verify', 'accepted', 'ids'):
                    vocab.update(p.get(key, []))
            for b in blocks:
                for pos in b['positions']:
                    vocab.update(c[0] for c in pos['cands'])
                    vocab.update(pos['nearest'])
            counted = stats.get('total_steps', 0)
            trace = dict(
                meta=dict(model=args.model, tag=tag, mode=mode, qid=qid,
                          question=row['question'], reference=reference,
                          extracted=extracted, correct=extracted == reference,
                          response=response, prompt_len=prompt_len,
                          gen_len=len(ids) - prompt_len, forwards=counted,
                          tpf=round((len(ids) - prompt_len) / max(counted, 1), 3),
                          seconds=round(seconds, 2), block_size=tracer.N,
                          num_layers=tracer.num_layers, layer_groups=tracer.layer_groups,
                          settings=dict(num_steps=args.num_steps if mode == 'flow' else 1,
                                        cfg_scale=args.cfg_scale if mode == 'flow' else 1.0,
                                        noise_scale=args.noise_scale_override,
                                        self_cond=bool(model.enable_self_cond) if mode == 'flow' else False,
                                        seed=args.seed),
                          stats={k: v for k, v in stats.items() if isinstance(v, (int, float, bool))}),
                ids=ids,
                vocab={str(i): tokenizer.decode([i]) for i in sorted(vocab)},
                passes=tracer.passes,
                blocks=blocks,
            )
            path = out_dir / f'{tag}-{mode}-q{qid}.json'
            with open(path, 'w') as f:
                json.dump(trace, f, separators=(',', ':'))
            print(f'{path}: {len(tracer.passes)} passes, {trace["meta"]["gen_len"]} tokens, '
                  f'TPF {trace["meta"]["tpf"]}, answer {extracted} vs {reference} '
                  f'({"correct" if trace["meta"]["correct"] else "wrong"}), '
                  f'{os.path.getsize(path) / 1e6:.2f} MB', flush=True)


if __name__ == '__main__':
    main()
