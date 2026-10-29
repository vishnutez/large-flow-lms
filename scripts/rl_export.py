"""Export a GRPO checkpoint as a bf16 Hugging Face model directory.

Copies the float32 master weights from the checkpoint shards into the base
model and saves it with ``save_pretrained``, so evaluation tools can load it
like any released checkpoint. Runs on CPU (~15 GB of RAM for 7B).

    python scripts/rl_export.py output/rl/grpo-7b-k2                  # -> .../export-step<N>
    python scripts/rl_export.py output/rl/grpo-7b-k2 --checkpoint keep-step100
"""
import argparse
import glob
import json
import os

import torch
from transformers import AutoTokenizer

from src.lafa.modeling_lafa import LafaForCausalLM


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run')
    p.add_argument('--checkpoint', default='checkpoint', help='directory inside the run')
    p.add_argument('--output', default=None)
    args = p.parse_args()
    with open(f'{args.run}/args.json') as f:
        run_args = json.load(f)
    ckpt = f'{args.run}/{args.checkpoint}'
    with open(f'{ckpt}/state.json') as f:
        step = json.load(f)['step']
    out = args.output or f'{args.run}/export-step{step}'
    if os.path.exists(f'{out}/config.json'):
        print(f'{out} exists; skipping')
        return

    model = LafaForCausalLM.from_pretrained(run_args['model'], dtype=torch.bfloat16)
    if not run_args.get('train_embeddings'):
        model.embed_tokens.weight.requires_grad_(False)
        model.lm_head.weight.requires_grad_(False)
    # Same ordering MasterAdamW uses: trainable parameters in model order.
    params = [p for p in model.parameters() if p.requires_grad]
    loaded = 0
    for path in sorted(glob.glob(f'{ckpt}/optim-*.pt')):
        shard = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        for i, entry in shard.items():
            params[i].data.copy_(entry['master'])
            loaded += 1
    if loaded != len(params):
        raise RuntimeError(f'checkpoint has {loaded} of {len(params)} trainable tensors')
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(run_args['model'], trust_remote_code=True).save_pretrained(out)
    with open(f'{out}/grpo_step.json', 'w') as f:
        json.dump(dict(step=step, run=args.run), f)
    print(f'exported step {step} to {out}')


if __name__ == '__main__':
    main()
