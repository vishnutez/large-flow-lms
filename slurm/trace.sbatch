#!/bin/bash
#SBATCH --job-name=lafa-trace
#SBATCH --partition=def
#SBATCH --account=prj-02-llm-reasoning-kalathil
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=output/slurm/%x-%j.out

# Record per-forward decoding trajectories on GSM8K questions for the
# dashboard. Arguments go to scripts/trace_generation.py, for example:
#
#   sbatch slurm/trace.sbatch --model lafa-hf/lafa-8B --modes flow,speculative --qids 0,1,2,3,4,5
#   sbatch slurm/trace.sbatch --model lafa-hf/lafa-0.5B --modes flow --qids 0 --num-steps 8

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"

python scripts/trace_generation.py "$@"
