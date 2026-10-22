"""Generate chat responses from a Lafa model and report tokens per forward.

Prompts come from --prompt (repeatable) or --prompt-file (one per line, or
JSONL rows with a "prompt" field). Results are printed and optionally written
as JSONL.
"""
import argparse
import json
import os
import time


def _str2bool(value):
    return value.lower() in ('1', 'true', 'yes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='lafa-hf/lafa-0.5B', help='HF repo ID or local directory')
    parser.add_argument('--revision')
    parser.add_argument('--subfolder', default='', help='"raw" for non-EMA weights')
    parser.add_argument('--prompt', action='append', default=[])
    parser.add_argument('--prompt-file')
    parser.add_argument('--gen-mode', default='flow', choices=['flow', 'ar', 'speculative'])
    parser.add_argument('--num-steps', type=int, default=4)
    parser.add_argument('--cfg-scale', type=float, default=1.0)
    parser.add_argument('--noise-scale-override', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--self-cond', type=_str2bool, help='Default keeps the checkpoint setting')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', help='Optional JSONL output path')
    args = parser.parse_args()

    prompts = list(args.prompt)
    if args.prompt_file:
        with open(args.prompt_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line)['prompt'] if line.startswith('{') else line)
    if not prompts:
        parser.error('Pass --prompt or --prompt-file')

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load = dict(trust_remote_code=True, revision=args.revision, subfolder=args.subfolder)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, **load).to('cuda').eval()

    gen_kwargs = dict(gen_mode=args.gen_mode, max_new_tokens=args.max_new_tokens,
                      noise_scale_override=args.noise_scale_override, seed=args.seed,
                      return_dict_in_generate=True)
    if args.gen_mode == 'flow':
        gen_kwargs.update(num_steps=args.num_steps, cfg_scale=args.cfg_scale)
        if args.threshold:
            gen_kwargs['threshold'] = args.threshold
    if args.self_cond is not None:
        gen_kwargs['self_cond'] = args.self_cond

    out = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        out = open(args.output, 'w')
    total_tokens = total_forwards = 0
    for prompt in prompts:
        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt').to('cuda')
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        response = tokenizer.decode(
            result.sequences[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        stats = result.stats[0]
        gen_len = stats.get('gen_len', stats.get('generated', 0))
        forwards = stats.get('total_steps', 0)
        total_tokens += gen_len
        total_forwards += forwards
        print(f'### Prompt\n{prompt}\n### Response\n{response}\n'
              f'[{gen_len} tokens, {forwards} forwards, '
              f'TPF {gen_len / max(forwards, 1):.2f}, {seconds:.2f}s]\n', flush=True)
        if out:
            out.write(json.dumps(dict(prompt=prompt, response=response, seconds=seconds,
                                      stats=stats, settings=gen_kwargs), default=str) + '\n')
    if out:
        out.close()
    print(f'Overall TPF: {total_tokens / max(total_forwards, 1):.2f} '
          f'({total_tokens} tokens / {total_forwards} forwards)')


if __name__ == '__main__':
    main()
