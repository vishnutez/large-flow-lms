"""Compare one-at-a-time and batched NFT rollouts on the same train questions.

For each mode, decodes ``group_size`` ODE answers per question with the base
model, exactly as ``nft_step`` would, and reports accuracy, mixed-group
fraction, distinct answers per group, length, truncation, and the mean AR
negative log-likelihood of the answers under the same model at unpadded
positions (the distribution the NFT update treats as p_old).

    sbatch slurm/run.sbatch scripts/rl_check_batched.py --questions 64
"""
import argparse
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from src.lafa.modeling_lafa import LafaForCausalLM
from src.rl.flow_policy import batched_rollouts, rollout
from src.rl.grpo import gsm8k_reward
from src.rl.nft import build_sequence, noised_input, packed_logits

p = argparse.ArgumentParser()
p.add_argument('--model', default='lafa-hf/lafa-7B')
p.add_argument('--questions', type=int, default=64)
p.add_argument('--group-size', type=int, default=8)
p.add_argument('--num-steps', type=int, default=2)
p.add_argument('--noise-scale', type=float, default=0.1)
p.add_argument('--max-new-tokens', type=int, default=512)
p.add_argument('--paired', type=int, default=0,
               help='instead: decode each answer from the same initial noise with rollout, '
                    'generate_batched at batch 1, 8 (copies) and 16 (two prompts, padded); '
                    'this many seeds per question')
args = p.parse_args()

device = torch.device('cuda')
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = LafaForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()
model.noise_scale_mult = args.noise_scale
im_end = tok.convert_tokens_to_ids('<|im_end|>')
eos = model.backbone.config.eos_token_id
eos = [eos] if isinstance(eos, int) else list(eos or [])
model.backbone.config.eos_token_id = eos + [im_end] * (im_end not in eos)

data = load_dataset('openai/gsm8k', 'main', split='train')
ids = list(range(len(data)))
random.Random(1234).shuffle(ids)
train_ids = ids[500:]
random.Random(0).shuffle(train_ids)
qs = []
for i in train_ids[:args.questions]:
    text = tok.apply_chat_template(
        [{'role': 'user', 'content': f"Q: {data[i]['question']}\nA:"}],
        tokenize=False, add_generation_prompt=True)
    qs.append(dict(prompt_ids=tok(text, add_special_tokens=False, return_tensors='pt').input_ids[0],
                   answer=data[i]['answer']))
G = args.group_size
common = dict(max_new_tokens=args.max_new_tokens, num_steps=args.num_steps)


def sequential(pair_seed):
    return [rollout(model, q['prompt_ids'], sde_gamma=0.0,
                    seed=pair_seed * 1_000_003 + qi * G + g, **common)
            for qi, q in enumerate(qs) for g in range(G)]


def batched(per_batch):
    out = []
    for s in range(0, len(qs), per_batch):  # nft_step: one call per rank-step
        part = qs[s:s + per_batch]
        out += batched_rollouts(model, [q['prompt_ids'] for q in part for _ in range(G)],
                                seed=s, batch_size=G * per_batch, **common)
    return out


@torch.no_grad()
def ar_nll(ro):
    i, labels = build_sequence(ro.prompt_ids, ro.response_ids, model.bd_size, 0)
    i, labels = i.to(device), labels.to(device)
    inp = noised_input(model, i, labels, noise_scale=4.0, self_cond_prob=0.0)
    logits = packed_logits(model, inp)
    L = i.shape[1]
    valid = labels[0, 1:] != -100
    lp = F.log_softmax(logits[0, :L - 1][valid].float(), -1)
    return -lp.gather(-1, i[0, 1:][valid][:, None]).sum().item(), int(valid.sum())


def report(name, ros):
    rewards = [gsm8k_reward(tok.decode(r.response_ids, skip_special_tokens=True), q['answer'])
               for r, q in zip(ros, (q for q in qs for _ in range(G)))]
    groups = [rewards[k:k + G] for k in range(0, len(rewards), G)]
    resp = [tuple(r.response_ids.tolist()) for r in ros]
    distinct = sum(len(set(resp[k:k + G])) for k in range(0, len(resp), G)) / len(groups)
    lens = [r.response_ids.numel() for r in ros]
    nll, n = map(sum, zip(*(ar_nll(r) for r in ros)))
    print(f'{name:28s} acc {sum(rewards) / len(rewards):.3f}  mixed '
          f'{sum(0 < sum(g) < G for g in groups) / len(groups):.3f}  distinct/group '
          f'{distinct:.2f}  len {sum(lens) / len(lens):6.1f}  trunc '
          f'{sum(x >= args.max_new_tokens for x in lens) / len(lens):.3f}  AR nll/token '
          f'{nll / n:.4f}', flush=True)


def paired(seeds):
    """Same initial noise in every mode (shared noise across a batch's rows),
    so differences come from batch numerics alone."""
    import math
    common_b = dict(max_new_tokens=args.max_new_tokens, num_steps=args.num_steps,
                    grid_mode='warp')
    res = {k: [] for k in ('b1', 'b8', 'b16')}
    for qi in range(0, len(qs) - 1, 2):
        pa, pb = qs[qi]['prompt_ids'], qs[qi + 1]['prompt_ids']
        for s in range(seeds):
            ref = rollout(model, pa, sde_gamma=0.0, seed=s, **common)
            outs = {}
            outs['b1'] = model.generate_batched(pa[None], torch.ones(1, pa.numel(), dtype=torch.long),
                                                seed=s, **common_b)[0][0]
            outs['b8'] = model.generate_batched(pa[None].expand(8, -1),
                                                torch.ones(8, pa.numel(), dtype=torch.long),
                                                seed=s, **common_b)[0][3]
            w = max(pa.numel(), pb.numel())
            ids = torch.zeros(16, w, dtype=torch.long)
            mask = torch.zeros(16, w, dtype=torch.long)
            for j, pr in enumerate([pa] * 8 + [pb] * 8):
                ids[j, w - pr.numel():] = pr
                mask[j, w - pr.numel():] = 1
            outs['b16'] = model.generate_batched(ids, mask, seed=s, **common_b)[0][3]
            n_ref, t_ref = ar_nll(ref)
            for k, seq in outs.items():
                resp = torch.tensor(seq[pa.numel():], dtype=torch.long)
                ro = type(ref)(prompt_ids=pa, response_ids=resp, blocks=[], sigma=0.0,
                               num_forwards=0)
                n, t = ar_nll(ro)
                res[k].append((resp.tolist() == ref.response_ids.tolist(), n / t - n_ref / t_ref))
    for k, v in res.items():
        d = [x for _, x in v]
        mean = sum(d) / len(d)
        se = math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1) / len(d))
        print(f'{k:4s} identical to rollout {sum(i for i, _ in v)}/{len(v)}   '
              f'paired AR nll/token diff {mean:+.4f} ± {se:.4f}', flush=True)


if args.paired:
    paired(args.paired)
    raise SystemExit

report('sequential (seed 1)', sequential(1))
report('sequential (seed 2)', sequential(2))
report('batched, 2 questions/call', batched(2))
report('batched, 1 question/call', batched(1))
