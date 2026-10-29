"""GRPO on flow rollouts: rewards, group advantages, one update step.

One step: for each question, sample a group of SDE rollouts, score them,
normalize rewards within the group, then accumulate ``sequence_loss`` over
the rollouts with nonzero advantage and take one optimizer step. With one
step per batch the update is on-policy (ratios are 1, the clip is inactive).
"""

import re
import time

import torch
import torch.distributed as dist

from src.rl.flow_policy import rollout
from src.rl.loss import old_logprobs, sequence_loss

# lm-eval gsm8k_cot "flexible-extract": the last number in the response,
# compared after the metric's regexes_to_ignore.
_NUMBER = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def _normalize(s):
    s = re.sub(r"(?s).*#### ", "", s)
    s = s.replace(",", "").replace("$", "")
    return re.sub(r"\.$", "", s.strip()).lower()


def gsm8k_reward(response, answer):
    """1 if the response's last number equals the gold answer, else 0."""
    response = response.split("Q:")[0]
    matches = _NUMBER.findall(response)
    if not matches:
        return 0.0
    pred = next(group for group in matches[-1] if group)
    return float(_normalize(pred) == _normalize(answer.split("####")[-1]))


def group_advantages(rewards, eps=1e-4):
    rewards = torch.as_tensor(rewards, dtype=torch.float64)
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)


class MasterAdamW:
    """AdamW on float32 master copies of bf16 parameters, sharded over ranks.

    bf16 weights cannot absorb RL-sized updates (a 1e-6 step is below bf16
    resolution), so the optimizer keeps float32 masters and copies them back
    after each step. Gradients are accumulated in float32 on every rank; the
    masters and Adam state are split across ranks by whole tensors (ZeRO-1),
    and each rank broadcasts the parameters it updated. For 7B, replicated
    state (16 bytes/param) does not fit next to the model on one H200.
    """

    def __init__(self, params, **kwargs):
        self.params = [p for p in params if p.requires_grad]
        self.grads = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        # Greedy balance: largest tensors first, each to the least-loaded rank.
        load, self.owner = [0] * world, [0] * len(self.params)
        for i in sorted(range(len(self.params)), key=lambda i: -self.params[i].numel()):
            self.owner[i] = min(range(world), key=load.__getitem__)
            load[self.owner[i]] += self.params[i].numel()
        self.owned = [i for i in range(len(self.params)) if self.owner[i] == rank]
        self.masters = {i: self.params[i].detach().float().clone() for i in self.owned}
        for i, m in self.masters.items():
            m.grad = self.grads[i]
        self.opt = torch.optim.AdamW(list(self.masters.values()), **kwargs)

    def accumulate(self):
        for p, g in zip(self.params, self.grads):
            if p.grad is not None:
                g.add_(p.grad.float())
                p.grad = None

    def step(self, max_grad_norm):
        if dist.is_initialized():
            for g in self.grads:
                dist.all_reduce(g, op=dist.ReduceOp.AVG)
        norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in self.grads]))
        if not torch.isfinite(norm):
            # Skip the update: one inf/NaN gradient would otherwise turn every
            # parameter into NaN (the norm is identical on every rank).
            for g in self.grads:
                g.zero_()
            return norm.item()
        scale = (max_grad_norm / (norm + 1e-6)).clamp(max=1.0)
        for g in self.grads:
            g.mul_(scale)
        if self.masters:
            self.opt.step()
        for i, m in self.masters.items():
            self.params[i].data.copy_(m)
        if dist.is_initialized():
            for p, owner in zip(self.params, self.owner):
                dist.broadcast(p.data, src=owner)
        for g in self.grads:
            g.zero_()
        return norm.item()

    def shard_state(self):
        """This rank's masters and Adam state, keyed by parameter index (CPU)."""
        out = {}
        for i, m in self.masters.items():
            state = self.opt.state.get(m, {})
            out[i] = dict(master=m.detach().cpu(),
                          **{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in state.items()})
        return out

    def load_shards(self, paths):
        """Restore owned parameters from shard files saved under any rank layout,
        then copy the masters into the bf16 parameters on every rank."""
        found = set()
        for path in paths:
            shard = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            for i in self.owned:
                if i in shard:
                    entry = shard[i]
                    m = self.masters[i]
                    m.data.copy_(entry["master"])
                    state = {k: v for k, v in entry.items() if k != "master"}
                    if state:
                        self.opt.state[m] = {
                            k: ((v.to(m.device) if v.ndim else v.clone()) if torch.is_tensor(v) else v)
                            for k, v in state.items()}
                    found.add(i)
        if found != set(self.owned):
            raise RuntimeError(f"checkpoint misses {len(set(self.owned) - found)} owned tensors")
        for i, m in self.masters.items():
            self.params[i].data.copy_(m)
        if dist.is_initialized():
            for p, owner in zip(self.params, self.owner):
                dist.broadcast(p.data, src=owner)


def grpo_step(model, optimizer, tokenizer, questions, *, group_size, num_steps, sde_gamma,
              max_new_tokens, clip, ar_weight, max_grad_norm, seed, minibatches=1):
    """One GRPO step on ``questions`` (dicts with 'prompt_ids' and 'answer').

    With ``minibatches`` > 1 the rollouts are split into that many disjoint
    parts and the optimizer steps once per part (each rollout is used once).
    Old log-probs are computed before the first step, so the ratio is 1 in the
    first part and the PPO clip is a trust region in the later ones; with one
    minibatch the update is exactly on-policy (ratio 1).

    Returns a dict of sums over this rank's rollouts (``count`` is the number
    of rollouts) so callers can all-reduce and average.
    """
    steps = list(range(num_steps - 1))
    sums = dict(count=0, reward=0.0, tokens=0, forwards=0, mixed_groups=0, trained=0,
                flow_loss=0.0, ar_loss=0.0, flow_clip_frac=0.0, ar_clip_frac=0.0,
                mismatch_mean=0.0, mismatch_abs_max=0.0,
                mismatch_frac_gt1=0.0, rollout_s=0.0, update_s=0.0)
    batch = []
    start = time.perf_counter()
    for qi, q in enumerate(questions):
        group = [rollout(model, q["prompt_ids"], max_new_tokens=max_new_tokens,
                         num_steps=num_steps, sde_gamma=sde_gamma,
                         seed=seed * 1_000_003 + qi * group_size + g)
                 for g in range(group_size)]
        rewards = [gsm8k_reward(tokenizer.decode(ro.response_ids, skip_special_tokens=True),
                                q["answer"]) for ro in group]
        adv = group_advantages(rewards)
        sums["mixed_groups"] += int(len(set(rewards)) > 1)
        for ro, r, a in zip(group, rewards, adv.tolist()):
            sums["count"] += 1
            sums["reward"] += r
            sums["tokens"] += ro.response_ids.numel()
            sums["forwards"] += ro.num_forwards
            batch.append((ro, a))
    sums["rollout_s"] = time.perf_counter() - start

    start = time.perf_counter()
    olds = [old_logprobs(model, ro, steps) if minibatches > 1 and a != 0.0 else None
            for ro, a in batch]
    order = torch.randperm(len(batch), generator=torch.Generator().manual_seed(seed)).tolist() \
        if minibatches > 1 else list(range(len(batch)))
    norms = []
    for mb in range(minibatches):
        part = order[mb::minibatches]
        for j in part:
            ro, a = batch[j]
            if a == 0.0:  # no learning signal; skip the forward
                continue
            loss, stats = sequence_loss(model, ro, a, olds[j], steps, clip=clip,
                                        ar_weight=ar_weight)
            (loss / len(part)).backward()
            optimizer.accumulate()
            sums["trained"] += 1
            for key in ("flow_loss", "ar_loss", "flow_clip_frac", "ar_clip_frac",
                        "mismatch_mean", "mismatch_frac_gt1"):
                sums[key] += stats[key]
            sums["mismatch_abs_max"] = max(sums["mismatch_abs_max"], stats["mismatch_abs_max"])
        norms.append(optimizer.step(max_grad_norm))
    sums["grad_norm"] = sum(norms) / len(norms)
    sums["update_s"] = time.perf_counter() - start
    return sums


@torch.no_grad()
def evaluate(model, tokenizer, questions, *, num_steps, max_new_tokens, sde_gamma=0.0):
    """Accuracy sums on ``questions``: (correct, tokens, forwards, count).

    sde_gamma=0 is the deterministic (ODE) sampler; with sde_gamma > 0 each
    question gets one SDE sample seeded by its ``qid``, so runs are comparable.
    """
    correct = tokens = forwards = 0
    for q in questions:
        ro = rollout(model, q["prompt_ids"], max_new_tokens=max_new_tokens,
                     num_steps=num_steps, sde_gamma=sde_gamma, seed=q.get("qid", 0))
        correct += gsm8k_reward(tokenizer.decode(ro.response_ids, skip_special_tokens=True),
                                q["answer"])
        tokens += ro.response_ids.numel()
        forwards += ro.num_forwards
    return dict(correct=correct, tokens=tokens, forwards=forwards, count=len(questions))
