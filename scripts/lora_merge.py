"""Merge a LoRA export (``step-NNNN`` of a ``--lora-rank`` run: adapters only)
into its base model and save a plain Lafa checkpoint the evaluation harness
loads like a released one (~15 GB for 7B; a few minutes on the login node).

    PYTHONPATH=. .venv/bin/python scripts/lora_merge.py output/rl/<run>/step-0150
    # -> output/rl/<run>/step-0150-merged
"""
import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from src.lafa.modeling_lafa import LafaForCausalLM
from src.rl.lora import ADAPTER_CONFIG, load_adapters, merged_state_dict


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('adapters', help='export directory with adapter.safetensors')
    p.add_argument('--out', default=None, help='default: <adapters>-merged')
    p.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float32'])
    args = p.parse_args()
    src = os.path.realpath(args.adapters)
    out = args.out or src.rstrip('/') + '-merged'
    with open(os.path.join(src, ADAPTER_CONFIG)) as f:
        base = json.load(f)['base_model']
    model = LafaForCausalLM.from_pretrained(base, dtype=getattr(torch, args.dtype)).eval()
    load_adapters(model, src)
    model.save_pretrained(out, state_dict=merged_state_dict(model))
    AutoTokenizer.from_pretrained(src, trust_remote_code=True).save_pretrained(out)
    print(f'merged {src} (base {base}) -> {out}')


if __name__ == '__main__':
    main()
