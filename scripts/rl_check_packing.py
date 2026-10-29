"""Check that packed log-probs reproduce SDE rollouts on a real checkpoint.

For each GSM8K train question: record a rollout, check its tokens match
``generate``, then recompute every transition mean in one packed forward and
compare with the rollout's. Reports the mean error relative to the step noise
and the per-block log-ratio, which should be near 0.

    sbatch slurm/run.sbatch scripts/rl_check_packing.py --model lafa-hf/lafa-7B
"""
import argparse
import json
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.lafa.flow_model import FlowModelWrapper
from src.lafa.modeling_lafa import LafaForCausalLM
from src.rl.flow_policy import packed_step_means, rollout, transition_logprob


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='lafa-hf/lafa-7B')
    parser.add_argument('--questions', type=int, default=4)
    parser.add_argument('--num-steps', type=int, default=2)
    parser.add_argument('--sde-gamma', type=float, default=0.5)
    parser.add_argument('--noise-scale', type=float, default=0.1)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float32'])
    parser.add_argument('--output', default='output/rl/check_packing.json')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # from_pretrained(dtype=float32) leaves the backbone in bf16, so cast after loading.
    model = LafaForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.to('cuda', getattr(torch, args.dtype)).eval()
    model.noise_scale_mult = args.noise_scale
    im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    eos = model.backbone.config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    model.backbone.config.eos_token_id = eos + [im_end] * (im_end not in eos)
    data = load_dataset('openai/gsm8k', 'main', split='train')

    report = []
    for qid in range(args.questions):
        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': f"Q: {data[qid]['question']}\nA:"}],
            tokenize=False, add_generation_prompt=True)
        prompt = tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids[0]
        start = time.perf_counter()
        ro = rollout(model, prompt, max_new_tokens=args.max_new_tokens,
                     num_steps=args.num_steps, sde_gamma=args.sde_gamma, seed=qid)
        rollout_s = time.perf_counter() - start
        expected, _ = FlowModelWrapper.generate(
            model, prompt[None].cuda(), max_new_tokens=args.max_new_tokens,
            num_steps=args.num_steps, grid_mode='warp', sde_gamma=args.sde_gamma,
            sde_vp=True, cfg_scale=1.0, seed=qid)
        same_tokens = torch.equal(ro.response_ids, expected[0, prompt.numel():].cpu())

        steps = list(range(args.num_steps - 1))
        start = time.perf_counter()
        with torch.no_grad():
            means, _ = packed_step_means(model, ro, steps)
        torch.cuda.synchronize()
        packed_s = time.perf_counter() - start
        rel_err, log_ratio = [], []
        for i, k in enumerate(steps):
            old = torch.stack([blk.means[k] for blk in ro.blocks]).to(means.device)
            std = torch.stack([blk.coef[k + 1] for blk in ro.blocks]).to(means.device) * ro.sigma
            rel_err.append(((means[i] - old).abs() / std[..., None]).flatten())
            log_ratio.append(transition_logprob(ro, k, means[i]) - transition_logprob(ro, k, old))
        rel_err, log_ratio = torch.cat(rel_err), torch.cat(log_ratio)
        row = dict(
            qid=qid, same_tokens=same_tokens, response_tokens=ro.response_ids.numel(),
            blocks=len(ro.blocks), transitions=log_ratio.numel(),
            mean_err_over_std=dict(median=rel_err.median().item(), max=rel_err.max().item()),
            log_ratio=dict(mean=log_ratio.mean().item(), abs_max=log_ratio.abs().max().item()),
            rollout_s=round(rollout_s, 2), packed_forward_s=round(packed_s, 3),
            response=tokenizer.decode(ro.response_ids, skip_special_tokens=True))
        report.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'response'}), flush=True)

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(dict(settings=vars(args), questions=report), f, indent=1)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
