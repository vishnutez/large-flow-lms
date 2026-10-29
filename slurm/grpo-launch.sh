#!/bin/bash
# Shared tail of slurm/grpo*.sbatch (sourced after env.sh): rendezvous on the
# first node, then one task per GPU.

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))
# NCCL's bootstrap sockets must use the routable Ethernet bond (what hostnames
# resolve to); the per-rail IPoIB addresses (100.12x.x.x) do not route between
# nodes. Bulk traffic still goes over InfiniBand verbs.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^ib,lo,docker}"
echo "nodes=$SLURM_JOB_NODELIST tasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT"

# --gpu-bind=none: with --gpus-per-task, Slurm otherwise confines each task to
# its own GPU and NCCL cannot reach same-node peers ("invalid device ordinal");
# every task sees its node's GPUs and picks one by local rank.
# Nodes mix InfiniBand and RoCE NICs; NCCL fails pairing one with the other
# across nodes ("Try selecting NICs of only one link type"). Each task restricts
# NCCL to its node's InfiniBand ports, read from /sys.
srun --kill-on-bad-exit=1 --gpu-bind=none bash -c '
    hcas=$(for d in /sys/class/infiniband/*; do
        [[ $(cat "$d/ports/1/link_layer" 2>/dev/null) == InfiniBand ]] && printf "%s," "${d##*/}"
    done)
    export NCCL_IB_HCA="${NCCL_IB_HCA:-=${hcas%,}}"
    # Slurm sometimes sets CUDA_VISIBLE_DEVICES to one GPU for every task on a
    # node even with --gpu-bind=none; clear it so CUDA sees what the cgroup
    # allows (all of the job'"'"'s GPUs on the node) and pick by local rank.
    unset CUDA_VISIBLE_DEVICES
    [[ $SLURM_PROCID == 0 ]] && echo "NCCL_IB_HCA=$NCCL_IB_HCA"
    exec python "${RL_SCRIPT:-scripts/train_grpo.py}" "$@"' bash "$@"
