#!/bin/bash
#SBATCH --job-name=lafa-train
#SBATCH --partition=def
#SBATCH --account=prj-02-llm-reasoning-kalathil
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=112
#SBATCH --mem=900G
#SBATCH --time=2-00:00:00
#SBATCH --output=output/slurm/%x-%j.out

# Train on one node. The first argument is the training script, the second its
# config, and the rest are config overrides (see docs/training.md):
#
#   sbatch slurm/train.sbatch train_flow.py config/lafa_0.5b.json
#   sbatch --gres=gpu:H200:4 slurm/train.sbatch train_flow.py config/finetune_7b.json
#   sbatch slurm/train.sbatch train_ar.py config/ar_0.5b.json
#   sbatch slurm/train.sbatch train_distill.py config/distill_0.5b.json
#
# The global batch is per-device batch x accumulation x GPUs; adjust
# --gradient_accumulation_steps when changing the GPU count. Every script
# resumes from the latest checkpoint-* in output_dir, so resubmitting the same
# command continues a run that hit the time limit.

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch slurm/train.sbatch <train_flow.py|train_ar.py|train_distill.py> <config.json> [overrides]" >&2
    exit 1
fi
SCRIPT="$1"
shift

PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))
echo "script=$SCRIPT args=$*"

if [[ "$SCRIPT" == train_distill.py ]]; then
    # Distillation uses plain DDP with replicated frozen teachers.
    torchrun --standalone --nproc_per_node="$NUM_GPUS" "$SCRIPT" "$@"
else
    accelerate launch --num_machines 1 --num_processes "$NUM_GPUS" \
        --main_process_port "$PORT" "$SCRIPT" "$@"
fi
