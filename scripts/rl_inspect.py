"""Decode validation questions with a GRPO checkpoint and the base model.

Rebuilds the policy from the checkpoint's float32 master shards (no HF export
needed), decodes the first N validation questions with the deterministic
sampler, and writes both models' responses side by side with lengths and
rewards.

    sbatch slurm/run.sbatch scripts/rl_inspect.py --run output/rl/grpo-7b-k2 --questions 64
"""
import argparse
import glob
import json

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from src.lafa.modeling_lafa import LafaForCausalLM
from src.rl.flow_policy import rollout
from src.rl.grpo import gsm8k_reward


def load_policy(run, args, train_embeddings, checkpoint=True):
    model = LafaForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to('cuda').eval()
    model.noise_scale_mult = args.noise_scale
    if checkpoint:
        if not train_embeddings:
            model.embed_tokens.weight.requires_grad_(False)
            model.lm_head.weight.requires_grad_(False)
        # Same ordering MasterAdamW uses: trainable parameters in model order.
        params = [p for p in model.parameters() if p.requires_grad]
        loaded = 0
        for path in sorted(glob.glob(f'{run}/checkpoint/optim-*.pt')):
            shard = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
            for i, entry in shard.items():
                params[i].data.copy_(entry['master'])
                loaded += 1
        assert loaded == len(params), (loaded, len(params))
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True)
    p.add_argument('--questions', type=int, default=64)
    p.add_argument('--output', default=None)
    args_cli = p.parse_args()
    with open(f'{args_cli.run}/args.json') as f:
        args = argparse.Namespace(**json.load(f))
    with open(f'{args_cli.run}/checkpoint/state.json') as f:
        step = json.load(f)['step']
    with open(f'{args_cli.run}/val_ids.json') as f:
        val_ids = json.load(f)[:args_cli.questions]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    data = load_dataset('openai/gsm8k', 'main', split='train')
    im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')

    results = {i: {} for i in val_ids}
    for name, use_ckpt in (('base', False), (f'step{step}', True)):
        model = load_policy(args_cli.run, args, args.train_embeddings, checkpoint=use_ckpt)
        eos = model.backbone.config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        model.backbone.config.eos_token_id = eos + [im_end] * (im_end not in eos)
        for i in val_ids:
            text = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': f"Q: {data[i]['question']}\nA:"}],
                tokenize=False, add_generation_prompt=True)
            ids = tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids[0]
            ro = rollout(model, ids, max_new_tokens=args.max_new_tokens,
                         num_steps=args.num_steps, sde_gamma=0.0, seed=i)
            response = tokenizer.decode(ro.response_ids, skip_special_tokens=True)
            results[i][name] = dict(tokens=ro.response_ids.numel(),
                                    hit_budget=ro.response_ids.numel() >= args.max_new_tokens,
                                    reward=gsm8k_reward(response, data[i]['answer']),
                                    response=response)
        del model
        torch.cuda.empty_cache()

    names = ['base', f'step{step}']
    for name in names:
        rows = [r[name] for r in results.values()]
        print(f"{name}: accuracy {sum(r['reward'] for r in rows) / len(rows):.3f} "
              f"mean tokens {sum(r['tokens'] for r in rows) / len(rows):.1f} "
              f"at budget {sum(r['hit_budget'] for r in rows)}", flush=True)
    out = args_cli.output or f'{args_cli.run}/inspect-step{step}.json'
    with open(out, 'w') as f:
        json.dump(dict(step=step, questions={str(i): dict(question=data[i]['question'],
                                                          answer=data[i]['answer'], **r)
                                             for i, r in results.items()}), f, indent=1)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
