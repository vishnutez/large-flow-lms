#!/bin/bash
#SBATCH --job-name=lafa-run
#SBATCH --partition=def
#SBATCH --account=prj-02-llm-reasoning-kalathil
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm/%x-%j.out

# Run any Python script from the repo on one GPU, for example:
#
#   sbatch slurm/run.sbatch scripts/rl_check_packing.py --model lafa-hf/lafa-7B

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"

python "$@"
