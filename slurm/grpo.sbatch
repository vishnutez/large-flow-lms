#!/bin/bash
#SBATCH --job-name=lafa-grpo
#SBATCH --partition=def
#SBATCH --account=prj-02-llm-reasoning-kalathil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=14
#SBATCH --mem=900G
#SBATCH --time=12:00:00
#SBATCH --output=output/slurm/%x-%j.out

# GRPO with flow rollouts, one task per GPU (scripts/train_grpo.py). Tasks read
# their rank from Slurm, so the same script runs on one or several nodes:
#
#   sbatch slurm/grpo.sbatch --model lafa-hf/lafa-7B --run-name grpo-7b-k2
#   sbatch --nodes=2 slurm/grpo.sbatch --run-name grpo-7b-k2-16gpu           # 2 x 8 GPUs
#   sbatch --nodes=2 --ntasks-per-node=4 --gres=gpu:H200:4 --mem=450G \
#       slurm/grpo.sbatch --run-name grpo-7b-k2                              # 2 x 4 GPUs
#   sbatch --nodes=1 --ntasks-per-node=1 --gres=gpu:H200:1 --mem=128G --time=1:00:00 \
#       slurm/grpo.sbatch --model lafa-hf/lafa-0.5B --run-name smoke --train-steps 3
#
# Keep --ntasks-per-node equal to the GPUs per node. Questions per step scale
# with the task count (--questions-per-rank each).

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"

source "$REPO/slurm/grpo-launch.sh"
