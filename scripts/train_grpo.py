"""GRPO on GSM8K train questions with SDE flow rollouts.

Each rank (one per GPU, on one or more nodes) rolls out its own questions
(whole groups, so advantages stay local) and gradients are averaged across
ranks. A seeded random validation set
of ``val_size`` train-split questions is held out (indices saved to
val_ids.json) and scored every ``val_every`` steps with both the deterministic
ODE sampler (sde_gamma=0, what decoding uses) and the training SDE sampler (one
sample per question); training samples from the rest.

    sbatch slurm/grpo.sbatch --model lafa-hf/lafa-7B --run-name grpo-7b-k2
    sbatch --nodes=2 slurm/grpo.sbatch --run-name grpo-7b-k2-16gpu    # 2 x 8 GPUs
"""
import argparse
import datetime
import glob
import json
import os
import random
import secrets
import shutil
import socket
import subprocess
import time

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer

from src.lafa.modeling_lafa import LafaForCausalLM
from src.rl.grpo import MasterAdamW, evaluate, grpo_step

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='lafa-hf/lafa-7B')
    p.add_argument('--run-name', default='grpo')
    p.add_argument('--train-steps', type=int, default=200)
    p.add_argument('--questions-per-rank', type=int, default=2)
    p.add_argument('--group-size', type=int, default=8)
    p.add_argument('--num-steps', type=int, default=2, help='denoising steps K per block')
    p.add_argument('--sde-gamma', type=float, default=0.5)
    p.add_argument('--noise-scale', type=float, default=0.1)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--minibatches', type=int, default=1,
                   help='optimizer steps per iteration on disjoint parts of the rollouts '
                        '(1 = one on-policy step; >1 makes the clip a trust region)')
    p.add_argument('--ar-weight', type=float, default=1.0)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--train-embeddings', action='store_true',
                   help='also train embed_tokens and lm_head (frozen by default, as in training)')
    p.add_argument('--val-size', type=int, default=500)
    p.add_argument('--val-every', type=int, default=25)
    p.add_argument('--val-seed', type=int, default=1234,
                   help='fixed apart from --seed so runs share a validation set')
    p.add_argument('--save-every', type=int, default=25,
                   help='checkpoint (weights + optimizer) every N steps; runs resume from it')
    p.add_argument('--export-at-val', type=int, default=0,
                   help='also save the policy as a bf16 HF model at every Nth validation '
                        'step (output/rl/<run>/step-NNNN, ~15 GB for 7B); 0 = never')
    p.add_argument('--stop-margin-min', type=float, default=15,
                   help='save and exit this many minutes before the Slurm time limit')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--wandb-project', default='lafa-rl', help="'' disables W&B logging")
    p.add_argument('--wandb-entity', default=None)
    p.add_argument('--wandb-key-file', default=os.path.join(REPO, '..', 'wandb-auth.txt'),
                   help='read when WANDB_API_KEY is unset')
    return p.parse_args()


def reduce_sums(sums, device):
    """Sum counters over ranks; take the max of *_abs_max entries."""
    if not dist.is_initialized():
        return sums
    out = {}
    for key, value in sums.items():
        t = torch.tensor(float(value), device=device, dtype=torch.float64)
        if key.endswith('_abs_max') or key.endswith('_s'):
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        elif key != 'grad_norm':
            dist.all_reduce(t)
        out[key] = t.item()
    return out


def init_wandb(args, world):
    """Start (or, after a restart, continue) a W&B run on rank 0; returns the
    module, or None when disabled."""
    if not args.wandb_project:
        return None
    import wandb
    if not os.environ.get('WANDB_API_KEY') and os.path.exists(args.wandb_key_file):
        with open(args.wandb_key_file) as f:
            os.environ['WANDB_API_KEY'] = f.read().strip()
    out_dir = f'output/rl/{args.run_name}'
    id_file = f'{out_dir}/wandb_id.txt'
    if os.path.exists(id_file):
        with open(id_file) as f:
            run_id = f.read().strip()
    else:
        run_id = secrets.token_hex(4)  # W&B ids: lowercase alphanumeric
        with open(id_file, 'w') as f:
            f.write(run_id + '\n')
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
               id=run_id, resume='allow', dir=out_dir,
               config=dict(vars(args), world_size=world,
                           slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    # Every row is plotted against the step; the reward and validation
    # metrics are repeated under by_epoch/ against epochs of the training set
    # (also defined by exact name in log_wandb). The patterns do not overlap.
    wandb.define_metric('step')
    wandb.define_metric('epoch')
    for prefix in ('train', 'val'):
        wandb.define_metric(f'{prefix}/*', step_metric='step')
    wandb.define_metric('by_epoch/*', step_metric='epoch')
    return wandb


def log_wandb(wb, row, epoch):
    """Log a row to W&B (see ``init_wandb``). Each by_epoch/ metric is also
    defined by its exact name when first logged: the web UI plotted these nested
    keys against the step despite the by_epoch/* rule."""
    out = {'step': row['step'], 'epoch': epoch}
    for k, v in row.items():
        if k in ('kind', 'step'):
            continue
        out[f"{row['kind']}/{k}"] = v
        if row['kind'] == 'val' or k == 'reward':
            name = f"by_epoch/{row['kind']}/{k}"
            if name not in _epoch_metrics:
                wb.define_metric(name, step_metric='epoch')
                _epoch_metrics.add(name)
            out[name] = v
    wb.log(out)


_epoch_metrics = set()


def slurm_job_end():
    """End time of this Slurm job (from its time limit), or None outside Slurm."""
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        return None
    try:
        out = subprocess.run(['squeue', '-h', '-j', job, '-o', '%e'], capture_output=True,
                             text=True, timeout=30).stdout.strip()
        return datetime.datetime.fromisoformat(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def other_live_job(lock_file):
    """Job id recorded in ``lock_file`` if it is another job that is still running."""
    if not os.path.exists(lock_file):
        return None
    with open(lock_file) as f:
        job = f.read().strip()
    if not job or job == os.environ.get('SLURM_JOB_ID'):
        return None
    out = subprocess.run(['squeue', '-h', '-j', job, '-o', '%T'], capture_output=True,
                         text=True).stdout.strip()
    return job if out in ('RUNNING', 'COMPLETING') else None


def slurm_to_torch_env(env):
    """Fill RANK/WORLD_SIZE/LOCAL_RANK from Slurm when launched by srun with one
    task per GPU (possibly across nodes). torchrun sets them itself."""
    if 'RANK' not in env and int(env.get('SLURM_NTASKS', 1)) > 1 and 'SLURM_PROCID' in env:
        env['RANK'] = env['SLURM_PROCID']
        env['WORLD_SIZE'] = env['SLURM_NTASKS']
        env['LOCAL_RANK'] = env['SLURM_LOCALID']
    return 'RANK' in env


def main():
    args = parse_args()
    distributed = slurm_to_torch_env(os.environ)
    # Tasks see either all of their node's GPUs or (with --gpus-per-task) only their own.
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank if torch.cuda.device_count() > 1 else 0)
    if torch.cuda.device_count() > 1 and local_rank >= torch.cuda.device_count():
        raise RuntimeError(f'local rank {local_rank} but only {torch.cuda.device_count()} GPUs visible')
    torch.cuda.set_device(device)
    print(f"[grpo] rank={os.environ.get('RANK', 0)} host={socket.gethostname().split('.')[0]} "
          f"local_rank={local_rank} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
          f"visible={torch.cuda.device_count()} device={device} "
          f"pci_bus={torch.cuda.get_device_properties(device).pci_bus_id}", flush=True)
    if distributed:
        dist.init_process_group('nccl', device_id=device)
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    out_dir = f'output/rl/{args.run_name}'
    ckpt_dir = f'{out_dir}/checkpoint'
    if (os.path.exists(f'{out_dir}/final/config.json')
            and not os.path.exists(f'{ckpt_dir}/state.json')):
        # Finished, and its optimizer checkpoint was deleted: a chained
        # follow-up job must not start the run over.
        if rank == 0:
            print(f'[grpo] {out_dir} is finished (final/ exists); exiting', flush=True)
        if distributed:
            dist.destroy_process_group()
        return
    # Refuse to start while another job is training the same run directory.
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
            print(f'[grpo] job {clash[0]} is already training {out_dir}; exiting', flush=True)
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
            print(f'[grpo] {out_dir} already finished {resume["step"]} steps', flush=True)
        if distributed:
            dist.destroy_process_group()
        return
    if rank == 0:
        with open(f'{out_dir}/args.json', 'w') as f:
            json.dump(dict(vars(args), world_size=world), f, indent=1)
        if resume and os.path.exists(f'{out_dir}/log.jsonl'):
            # Drop rows logged after the checkpoint; those steps are redone.
            with open(f'{out_dir}/log.jsonl') as f:
                rows = [line for line in f if json.loads(line)['step'] <= resume['step']]
            with open(f'{out_dir}/log.jsonl', 'w') as f:
                f.writelines(rows)
    wb = init_wandb(args, world) if rank == 0 else None

    # Rank 0 resolves the checkpoint against the hub; the others then read the
    # cache only (8 ranks querying the hub at once get rate-limited, HTTP 429).
    if distributed and rank != 0:
        dist.barrier()
    local = dict(local_files_only=rank != 0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, **local)
    model = LafaForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, **local)
    if distributed and rank == 0:
        dist.barrier()
    model = model.to(device, torch.bfloat16).eval()  # no dropout; eval mode still trains
    model.noise_scale_mult = args.noise_scale
    im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    eos = model.backbone.config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    model.backbone.config.eos_token_id = eos + [im_end] * (im_end not in eos)
    if not args.train_embeddings:
        model.embed_tokens.weight.requires_grad_(False)
        model.lm_head.weight.requires_grad_(False)
    optimizer = MasterAdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if resume:
        optimizer.load_shards(sorted(glob.glob(f'{ckpt_dir}/optim-*.pt')))
    if rank == 0:
        n = sum(p.numel() for p in optimizer.params)
        print(f'[grpo] world={world} trainable={n / 1e9:.2f}B params'
              + (f', resumed after step {resume["step"]}' if resume else ''), flush=True)

    # Rank 0 loads the dataset and shares it, again to keep hub queries to one rank.
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
        print(f'[grpo] train={len(train_ids)} val={len(val_ids)} questions', flush=True)
    val_questions = [encode(i) for i in val_ids[rank::world]]
    per_step = args.questions_per_rank * world
    log = open(f'{out_dir}/log.jsonl', 'a') if rank == 0 else None

    def run_eval(step):
        row = dict(kind='val', step=step)
        for name, gamma in (('ode', 0.0), ('sde', args.sde_gamma)):
            sums = reduce_sums(evaluate(model, tokenizer, val_questions, num_steps=args.num_steps,
                                        max_new_tokens=args.max_new_tokens, sde_gamma=gamma),
                               device)
            row[f'accuracy_{name}'] = sums['correct'] / sums['count']
            row[f'tokens_{name}'] = sums['tokens'] / sums['count']
            row[f'tokens_per_forward_{name}'] = sums['tokens'] / sums['forwards']
        if rank == 0:
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + '\n')
            log.flush()
            if wb:
                log_wandb(wb, row, row['step'] * per_step / len(train_ids))
        if args.export_at_val and step > 0 and (step // args.val_every) % args.export_at_val == 0:
            if rank == 0:
                path = f'{out_dir}/step-{step:04d}'
                model.save_pretrained(path)
                tokenizer.save_pretrained(path)
                print(f'[grpo] exported {path}', flush=True)
            if distributed:
                dist.barrier()

    def save(step):
        """Write weights, optimizer shards and the step to a temporary directory,
        then swap it in, so an interrupted save leaves the previous checkpoint."""
        tmp = f'{ckpt_dir}.tmp'
        if rank == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp)
        if distributed:
            dist.barrier()
        # The float32 masters are the weights; the bf16 model is exported once, at
        # the end (scratch space is tight: shards are ~80 GB for 7B).
        torch.save(optimizer.shard_state(), f'{tmp}/optim-rank{rank:03d}-of-{world:03d}.pt')
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
            print(f'[grpo] saved checkpoint at step {step}', flush=True)
        if distributed:
            dist.barrier()

    job_end = slurm_job_end() if rank == 0 else None
    if not resume:
        run_eval(0)
    for step in range(start_step, args.train_steps + 1):
        offset = ((step - 1) * per_step + rank * args.questions_per_rank) % len(train_ids)
        qids = [train_ids[(offset + j) % len(train_ids)] for j in range(args.questions_per_rank)]
        start = time.perf_counter()
        sums = grpo_step(
            model, optimizer, tokenizer, [encode(i) for i in qids],
            group_size=args.group_size, num_steps=args.num_steps, sde_gamma=args.sde_gamma,
            max_new_tokens=args.max_new_tokens, clip=args.clip, ar_weight=args.ar_weight,
            max_grad_norm=args.max_grad_norm, seed=args.seed * 100_003 + step * world + rank,
            minibatches=args.minibatches)
        sums = reduce_sums(sums, device)
        if rank == 0:
            n, trained = sums['count'], max(sums['trained'], 1)
            row = dict(
                kind='train', step=step, reward=sums['reward'] / n,
                mixed_groups=sums['mixed_groups'] / (per_step),
                tokens=sums['tokens'] / n, tokens_per_forward=sums['tokens'] / sums['forwards'],
                flow_loss=sums['flow_loss'] / trained, ar_loss=sums['ar_loss'] / trained,
                flow_clip_frac=sums['flow_clip_frac'] / trained,
                ar_clip_frac=sums['ar_clip_frac'] / trained,
                grad_norm=sums['grad_norm'],
                mismatch_mean=sums['mismatch_mean'] / trained,
                mismatch_frac_gt1=sums['mismatch_frac_gt1'] / trained,
                mismatch_abs_max=sums['mismatch_abs_max'],
                rollout_s=round(sums['rollout_s'], 1), update_s=round(sums['update_s'], 1),
                step_s=round(time.perf_counter() - start, 1),
                max_mem_gb=round(torch.cuda.max_memory_allocated(device) / 2**30, 1))
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + '\n')
            log.flush()
            if wb:
                log_wandb(wb, row, row['step'] * per_step / len(train_ids))
        if step % args.val_every == 0 or step == args.train_steps:
            run_eval(step)
        # Stop before the time limit (decided on rank 0, shared with all ranks).
        stop = [job_end is not None and (job_end - datetime.datetime.now()).total_seconds()
                < args.stop_margin_min * 60]
        if distributed:
            dist.broadcast_object_list(stop, src=0)
        if step % args.save_every == 0 or step == args.train_steps or stop[0]:
            save(step)
        if stop[0]:
            if rank == 0:
                print(f'[grpo] near the time limit; stopping after step {step}', flush=True)
            break

    if rank == 0:
        if step == args.train_steps:
            last = f'step-{step:04d}'
            if os.path.exists(f'{out_dir}/{last}/config.json'):
                if not os.path.lexists(f'{out_dir}/final'):
                    os.symlink(last, f'{out_dir}/final')
            else:
                model.save_pretrained(f'{out_dir}/final')
                tokenizer.save_pretrained(f'{out_dir}/final')
            print(f'[grpo] final model at {out_dir}/final', flush=True)
        os.remove(f'{out_dir}/RUNNING')
    if wb:
        wb.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
