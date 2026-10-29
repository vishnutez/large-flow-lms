"""Lafa-NFT on GSM8K train questions (forward-process RL, see src/rl/nft.py).

Same launch, validation split, logging, checkpointing and resume as
scripts/train_grpo.py. Each iteration: θ_old rolls out ``group_size`` ODE
answers per question, groups with mixed correctness are kept, the policy takes
``updates`` × ``minibatches`` optimizer steps on Lafa's training loss with the
NFT contrast (``updates`` passes over the answers), and θ_old moves toward θ by EMA.

    RL_SCRIPT=scripts/train_nft.py sbatch slurm/grpo-flex.sbatch --model lafa-hf/lafa-7B \\
        --run-name nft-7b-k2
"""
import argparse
import datetime
import functools
import glob
import json
import os
import random
import shutil
import socket
import time

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer

from src.lafa.modeling_lafa import LafaForCausalLM
from src.rl.grpo import MasterAdamW, evaluate, gsm8k_reward
from src.rl.lora import ADAPTER_CONFIG, add_lora, merge_lora, save_adapters, unmerge_lora
from src.rl.nft import OldPolicyEMA, eta_schedule, nft_step, nft_terms
from src.rl.nft_dn import dn_terms
from src.rl.nft_v2 import nft_terms_v2
from src.rl.ppo_lafa import ppo_lafa_terms
from src.rl.tilt import tilt_terms
from train_grpo import (init_wandb, log_wandb, other_live_job, reduce_sums, slurm_job_end,
                        slurm_to_torch_env)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='lafa-hf/lafa-7B')
    p.add_argument('--run-name', default='nft')
    p.add_argument('--train-steps', type=int, default=200, help='rollout iterations')
    p.add_argument('--questions-per-rank', type=int, default=2)
    p.add_argument('--group-size', type=int, default=8)
    p.add_argument('--num-steps', type=int, default=2, help='denoising steps K per block')
    p.add_argument('--noise-scale', type=float, default=0.1, help='decoder start noise')
    p.add_argument('--forward-noise-scale', type=float, default=4.0,
                   help="noise scale of the update's forward process (Lafa trains at 4.0)")
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--minibatches', type=int, default=4,
                   help='optimizer steps per pass over the rollouts')
    p.add_argument('--rollout-batch', type=int, default=0,
                   help='decode each rank\'s answers this many at a time (0: one at a time, '
                        'the earlier runs\' random stream)')
    p.add_argument('--updates', type=int, default=1,
                   help='passes over the rollouts per iteration (PPO epochs; 1 = each '
                        'answer trains once)')
    p.add_argument('--loss', default='v1', choices=['v1', 'v2', 'dn', 'ppo-lafa', 'tilt'],
                   help='v1: fixed beta in both denoiser branches (run nft-7b-k2); '
                        'v2: negative branch beta = r/(1-r), group-relative (nft_v2.py); '
                        'dn: DiffusionNFT, soft r and one beta (nft_dn.py); '
                        'ppo-lafa: PPO on the Lafa loss, A = R - r, one ratio per slot '
                        '(ppo_lafa.py; needs --ppo-clip, uses --dual-clip); '
                        'tilt: sigmoid-tilt p+/p- of p_old, always valid, weight 1 per answer '
                        '(tilt.py, our_work/sigmoid-tilt.md)')
    p.add_argument('--beta', type=float, default=1.0,
                   help='positive-branch beta (v2), both (v1, dn), temperature (tilt)')
    p.add_argument('--tilt-b', default='solve', choices=['solve', 'fixed'],
                   help='tilt: per-slot offset b; solve: alpha = r (root find), '
                        'fixed: b = F^-1(r)')
    p.add_argument('--tilt-kappa', type=float, default=1.0,
                   help='tilt: link F(s) = sigmoid(kappa s)^(1/kappa); 1 = sigmoid, large = '
                        "v2's pair inside the feasible region (a smooth dual clip)")
    p.add_argument('--eps', type=float, default=1.0, help='negative ratio clip (NFT)')
    # Defaults below depend on --loss: v1/v2 = all three terms, nft weighting,
    # per-token averaging, plain MSE; dn = embedding term only, soft r,
    # per-answer averaging, adaptive weight. Override one to ablate it.
    p.add_argument('--token-weight', type=float, default=None, help='denoiser token CE (0 drops it)')
    p.add_argument('--embed-weight', type=float, default=None, help='denoiser embedding MSE')
    p.add_argument('--ar-weight', type=float, default=None, help='AR head term')
    p.add_argument('--weighting', default=None, choices=['nft', 'dn', 'adv'],
                   help='nft: binary branch, omega = 1 - r; dn: soft r over both branches')
    p.add_argument('--normalize', default=None, choices=['token', 'answer'])
    p.add_argument('--adaptive-weight', type=int, default=None, choices=[0, 1],
                   help="DiffusionNFT's self-normalized MSE: mean / sg(mean |x - x0|)")
    p.add_argument('--mix', default='prob', choices=['prob', 'logit', 'geo'],
                   help='token-level +/- policies (token CE, AR): mix probabilities (NFT, '
                        'negative needs the eps clip) or logits (no clip, bounded gradient); '
                        'geo: logits, and the embedding term uses the same mixed p '
                        '(our_work/geometric-mixing.md)')
    p.add_argument('--ppo-clip', type=float, default=0.0,
                   help="PPO trust region on the token CE and AR terms: no gradient once "
                        "p/p_old moved past 1 +- this in the branch's direction (0: off)")
    p.add_argument('--dual-clip', type=float, default=0.0,
                   help='no gradient for a negative token with p/p_old above this, which bounds '
                        'the straight-through gradient by beta * this (0: off)')
    p.add_argument('--keep-uniform', type=int, default=0, choices=[0, 1],
                   help='dn: also train groups where every answer has the same reward '
                        '(DiffusionNFT does; r = 0.5, they only anchor to theta_old)')
    p.add_argument('--adv-clip', type=float, default=5.0,
                   help='dn: advantage clip; r = 0.5 + 0.5 clip(A)/adv_clip (DiffusionNFT: 5)')
    p.add_argument('--self-cond-prob', type=float, default=0.5)
    p.add_argument('--eta-rate', type=float, default=0.001)
    p.add_argument('--eta-max', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--train-embeddings', action='store_true')
    p.add_argument('--val-size', type=int, default=500)
    p.add_argument('--val-every', type=int, default=25)
    p.add_argument('--val-seed', type=int, default=1234)
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--export-at-val', type=int, default=1,
                   help='also save the policy as a bf16 HF model at every Nth validation '
                        'step (output/rl/<run>/step-NNNN, ~15 GB for 7B); 0 = never')
    p.add_argument('--lora-rank', type=int, default=0,
                   help='LoRA rank on the backbone linear layers (q,k,v,o,gate,up,down); '
                        'everything else frozen. 0: full fine-tuning')
    p.add_argument('--lora-alpha', type=float, default=None, help='LoRA alpha (default 2 x rank)')
    p.add_argument('--stop-margin-min', type=float, default=15)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--wandb-project', default='lafa-rl', help="'' disables W&B logging")
    p.add_argument('--wandb-entity', default=None)
    p.add_argument('--wandb-key-file', default=os.path.join(REPO, '..', 'wandb-auth.txt'))
    args = p.parse_args()
    dn = args.loss == 'dn'
    if args.loss in ('ppo-lafa', 'tilt') and args.weighting is None:
        args.weighting = 'adv'
    for name, v12, dn_value in [('token_weight', 1.0, 0.0), ('embed_weight', 1.0, 1.0),
                                ('ar_weight', 1.0, 0.0), ('weighting', 'nft', 'dn'),
                                ('normalize', 'token', 'answer'), ('adaptive_weight', 0, 1)]:
        if getattr(args, name) is None:
            setattr(args, name, dn_value if dn else v12)
    if args.loss == 'ppo-lafa':
        if args.weighting != 'adv' or args.ppo_clip <= 0.0:
            p.error('ppo-lafa: needs --ppo-clip > 0 and weighting adv')
    if args.loss == 'tilt' and (args.weighting != 'adv' or args.mix != 'prob'
                                or args.ppo_clip or args.dual_clip):
        p.error('tilt: uses weighting adv (equal branch weights) and no mix/ppo/dual clip')
    if dn and args.ar_weight and args.beta != 1.0:
        p.error('dn: the AR positive branch is the beta = 1 mixture; use --beta 1 or --ar-weight 0')
    args.sde_gamma = 0.0  # rollouts and the SDE validation column use the ODE
    if args.lora_rank and args.lora_alpha is None:
        args.lora_alpha = 2.0 * args.lora_rank
    return args


def exported(path):
    """A model export (full model, or LoRA adapters) exists at ``path``."""
    return any(os.path.exists(f'{path}/{name}') for name in ('config.json', ADAPTER_CONFIG))


def main():
    args = parse_args()
    distributed = slurm_to_torch_env(os.environ)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank if torch.cuda.device_count() > 1 else 0)
    if torch.cuda.device_count() > 1 and local_rank >= torch.cuda.device_count():
        raise RuntimeError(f'local rank {local_rank} but only {torch.cuda.device_count()} GPUs visible')
    torch.cuda.set_device(device)
    print(f"[nft] rank={os.environ.get('RANK', 0)} host={socket.gethostname().split('.')[0]} "
          f"device={device} pci_bus={torch.cuda.get_device_properties(device).pci_bus_id}",
          flush=True)
    if distributed:
        dist.init_process_group('nccl', device_id=device)
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    out_dir = f'output/rl/{args.run_name}'
    ckpt_dir = f'{out_dir}/checkpoint'

    if (exported(f'{out_dir}/final')
            and not os.path.exists(f'{ckpt_dir}/state.json')):
        # Finished, and its optimizer checkpoint was deleted: a chained
        # follow-up job must not start the run over.
        if rank == 0:
            print(f'[nft] {out_dir} is finished (final/ exists); exiting', flush=True)
        if distributed:
            dist.destroy_process_group()
        return

    clash = [None]
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        clash = [other_live_job(f'{out_dir}/RUNNING')]
        if clash[0] is None:
            with open(f'{out_dir}/RUNNING', 'w') as f:
                f.write(os.environ.get('SLURM_JOB_ID', 'local') + '\n')
    if distributed:
        dist.broadcast_object_list(clash, src=0)
    if clash[0] is not None:
        if rank == 0:
            print(f'[nft] job {clash[0]} is already training {out_dir}; exiting', flush=True)
        if distributed:
            dist.destroy_process_group()
        return

    resume = None
    if os.path.exists(f'{ckpt_dir}/state.json'):
        with open(f'{ckpt_dir}/state.json') as f:
            resume = json.load(f)
    start_step = resume['step'] + 1 if resume else 1
    if start_step > args.train_steps:
        if rank == 0:
            print(f'[nft] {out_dir} already finished {resume["step"]} steps', flush=True)
            os.remove(f'{out_dir}/RUNNING')
        if distributed:
            dist.destroy_process_group()
        return
    if rank == 0:
        with open(f'{out_dir}/args.json', 'w') as f:
            json.dump(dict(vars(args), world_size=world, algorithm='nft'), f, indent=1)
        if resume and os.path.exists(f'{out_dir}/log.jsonl'):
            with open(f'{out_dir}/log.jsonl') as f:
                rows = [line for line in f if json.loads(line)['step'] <= resume['step']]
            with open(f'{out_dir}/log.jsonl', 'w') as f:
                f.writelines(rows)
    wb = init_wandb(args, world) if rank == 0 else None

    # Rank 0 resolves the hub first; the others read the cache (HTTP 429 otherwise).
    if distributed and rank != 0:
        dist.barrier()
    local = dict(local_files_only=rank != 0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, **local)
    model = LafaForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, **local)
    old_model = LafaForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, **local)
    if distributed and rank == 0:
        dist.barrier()
    im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    for m in (model, old_model):
        m.to(device, torch.bfloat16).eval()
        m.noise_scale_mult = args.noise_scale
        eos = m.backbone.config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        m.backbone.config.eos_token_id = eos + [im_end] * (im_end not in eos)
    if args.lora_rank:
        # Same seed for both, so θ_old starts with the policy's A (B = 0).
        for m in (model, old_model):
            torch.manual_seed(args.seed)
            add_lora(m, args.lora_rank, args.lora_alpha)
    elif not args.train_embeddings:
        model.embed_tokens.weight.requires_grad_(False)
        model.lm_head.weight.requires_grad_(False)
    optimizer = MasterAdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = OldPolicyEMA(optimizer, model, old_model)
    if resume:
        optimizer.load_shards(sorted(glob.glob(f'{ckpt_dir}/optim-*.pt')))
        ema.load_shards(sorted(glob.glob(f'{ckpt_dir}/ema-*.pt')))
    if rank == 0:
        n = sum(p.numel() for p in optimizer.params)
        print(f'[nft] loss={args.loss} world={world} trainable={n / 1e9:.2f}B params'
              + (f', resumed after step {resume["step"]}' if resume else ''), flush=True)

    data = [None]
    if rank == 0:
        data = [[dict(question=r['question'], answer=r['answer'])
                 for r in load_dataset('openai/gsm8k', 'main', split='train')]]
    if distributed:
        dist.broadcast_object_list(data, src=0)
    data = data[0]

    def encode(i):
        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': f"Q: {data[i]['question']}\nA:"}],
            tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids[0]
        return dict(prompt_ids=ids, answer=data[i]['answer'], qid=i)

    ids = list(range(len(data)))
    random.Random(args.val_seed).shuffle(ids)
    val_ids, train_ids = sorted(ids[:args.val_size]), ids[args.val_size:]
    random.Random(args.seed).shuffle(train_ids)
    if rank == 0:
        with open(f'{out_dir}/val_ids.json', 'w') as f:
            json.dump(val_ids, f)
        print(f'[nft] train={len(train_ids)} val={len(val_ids)} questions', flush=True)
    val_questions = [encode(i) for i in val_ids[rank::world]]
    per_step = args.questions_per_rank * world
    log = open(f'{out_dir}/log.jsonl', 'a') if rank == 0 else None

    def write(row):
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + '\n')
        log.flush()
        if wb:
            log_wandb(wb, row, row['step'] * per_step / len(train_ids))

    def run_eval(step):
        # Validation decodes with θ (ODE). The SDE column of GRPO runs is not
        # meaningful here, so only the ODE is scored.
        if args.lora_rank:
            merge_lora(model)  # decode with one matmul per adapted layer
        sums = reduce_sums(evaluate(model, tokenizer, val_questions, num_steps=args.num_steps,
                                    max_new_tokens=args.max_new_tokens, sde_gamma=0.0), device)
        if args.lora_rank:
            unmerge_lora(model)
        if rank == 0:
            write(dict(kind='val', step=step, accuracy_ode=sums['correct'] / sums['count'],
                       tokens_ode=sums['tokens'] / sums['count'],
                       tokens_per_forward_ode=sums['tokens'] / sums['forwards']))
        if args.export_at_val and step > 0 and (step // args.val_every) % args.export_at_val == 0:
            if rank == 0:
                path = f'{out_dir}/step-{step:04d}'
                save_model(path)
                tokenizer.save_pretrained(path)
                print(f'[nft] exported {path}', flush=True)
            if distributed:
                dist.barrier()

    def save_model(path):
        # LoRA: only the adapters (scripts/lora_merge.py builds the full model).
        if args.lora_rank:
            save_adapters(model, path, base_model=args.model, rank=args.lora_rank,
                          alpha=args.lora_alpha)
        else:
            model.save_pretrained(path)

    def save(step):
        tmp = f'{ckpt_dir}.tmp'
        if rank == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
        if distributed:
            dist.barrier()
        torch.save(optimizer.shard_state(), f'{tmp}/optim-rank{rank:03d}-of-{world:03d}.pt')
        torch.save(ema.shard_state(), f'{tmp}/ema-rank{rank:03d}-of-{world:03d}.pt')
        if distributed:
            dist.barrier()
        if rank == 0:
            with open(f'{tmp}/state.json', 'w') as f:
                json.dump(dict(step=step, world_size=world,
                               slurm_job_id=os.environ.get('SLURM_JOB_ID')), f)
            shutil.rmtree(f'{ckpt_dir}.old', ignore_errors=True)
            if os.path.exists(ckpt_dir):
                os.rename(ckpt_dir, f'{ckpt_dir}.old')
            os.rename(tmp, ckpt_dir)
            shutil.rmtree(f'{ckpt_dir}.old', ignore_errors=True)
            print(f'[nft] saved checkpoint at step {step}', flush=True)
        if distributed:
            dist.barrier()

    job_end = slurm_job_end() if rank == 0 else None
    if not resume:
        run_eval(0)
    step = start_step - 1
    for step in range(start_step, args.train_steps + 1):
        offset = ((step - 1) * per_step + rank * args.questions_per_rank) % len(train_ids)
        qids = [train_ids[(offset + j) % len(train_ids)] for j in range(args.questions_per_rank)]
        start = time.perf_counter()
        sums = nft_step(
            model, old_model, ema, optimizer, tokenizer, [encode(i) for i in qids],
            group_size=args.group_size, num_steps=args.num_steps,
            max_new_tokens=args.max_new_tokens, minibatches=args.minibatches,
            updates=args.updates,
            rollout_batch=args.rollout_batch, beta=args.beta,
            eps=args.eps, embed_weight=args.embed_weight, ar_weight=args.ar_weight,
            noise_scale=args.forward_noise_scale, self_cond_prob=args.self_cond_prob,
            max_grad_norm=args.max_grad_norm, eta=eta_schedule(step, args.eta_rate, args.eta_max),
            seed=args.seed * 100_003 + step * world + rank, reward_fn=gsm8k_reward,
            terms_fn={'v1': nft_terms, 'v2': nft_terms_v2, 'dn': dn_terms,
                      'ppo-lafa': ppo_lafa_terms,
                      'tilt': functools.partial(tilt_terms, b_mode=args.tilt_b,
                                                kappa=args.tilt_kappa)}[args.loss],
            weighting=args.weighting, adv_clip=args.adv_clip, normalize=args.normalize,
            adaptive_weight=bool(args.adaptive_weight), token_weight=args.token_weight,
            keep_uniform=bool(args.keep_uniform), mix=args.mix,
            ppo_clip=args.ppo_clip, dual_clip=args.dual_clip)
        sums = reduce_sums(sums, device)
        if rank == 0:
            n = sums['count']
            nf, na = max(sums['n_flow'], 1), max(sums['n_ar'], 1)
            write(dict(
                kind='train', step=step, reward=sums['reward'] / n,
                mixed_groups=sums['mixed_groups'] / per_step, kept=sums['kept'],
                truncated=sums['truncated'] / n, tokens=sums['tokens'] / n,
                tokens_per_forward=sums['tokens'] / sums['forwards'],
                token_loss=sums['token_loss'] / nf, embed_loss=sums['embed_loss'] / nf,
                ar_loss=sums['ar_loss'] / na, token_clip_frac=sums['token_clip'] / nf,
                ar_clip_frac=sums['ar_clip'] / na, log_ratio_abs_max=sums['log_ratio_abs_max'],
                grad_norm=sums['grad_norm'], skipped_answers=sums['skipped_answers'],
                skipped_steps=sums['skipped_steps'] / world, eta=eta_schedule(step, args.eta_rate, args.eta_max),
                rollout_s=round(sums['rollout_s'], 1), update_s=round(sums['update_s'], 1),
                step_s=round(time.perf_counter() - start, 1),
                max_mem_gb=round(torch.cuda.max_memory_allocated(device) / 2**30, 1)))
        if step % args.val_every == 0 or step == args.train_steps:
            run_eval(step)
        stop = [job_end is not None and (job_end - datetime.datetime.now()).total_seconds()
                < args.stop_margin_min * 60]
        if distributed:
            dist.broadcast_object_list(stop, src=0)
        if step % args.save_every == 0 or step == args.train_steps or stop[0]:
            save(step)
        if stop[0]:
            if rank == 0:
                print(f'[nft] near the time limit; stopping after step {step}', flush=True)
            break

    if rank == 0:
        if step == args.train_steps:
            last = f'step-{step:04d}'
            if exported(f'{out_dir}/{last}'):
                if not os.path.lexists(f'{out_dir}/final'):
                    os.symlink(last, f'{out_dir}/final')
            else:
                save_model(f'{out_dir}/final')
                tokenizer.save_pretrained(f'{out_dir}/final')
            print(f'[nft] final model at {out_dir}/final', flush=True)
        os.remove(f'{out_dir}/RUNNING')
    if wb:
        wb.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
