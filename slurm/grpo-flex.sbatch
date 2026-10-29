#!/bin/bash
#SBATCH --job-name=lafa-grpo
#SBATCH --partition=def
#SBATCH --account=prj-02-llm-reasoning-kalathil
#SBATCH --ntasks=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-gpu=110G
#SBATCH --time=4:00:00
#SBATCH --output=output/slurm/%x-%j.out

# GRPO on 8 GPUs taken from wherever they are free (any number of nodes), for a
# busy queue where whole nodes are scarce. Same arguments as slurm/grpo.sbatch:
#
#   sbatch slurm/grpo-flex.sbatch --model lafa-hf/lafa-7B --run-name grpo-7b-k2
#   sbatch --ntasks=4 slurm/grpo-flex.sbatch ...                             # 4 GPUs
#
# Runs resume from output/rl/<run-name>/checkpoint, so a chain of short jobs
# (sbatch --dependency=afterany:<id> ...) finishes a long run.

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
source "$REPO/slurm/grpo-launch.sh"
