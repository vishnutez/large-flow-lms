#!/bin/bash
#SBATCH --job-name=lafa-infer
#SBATCH --partition=def
#SBATCH --account=prj-02-llm-reasoning-kalathil
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm/%x-%j.out

# Generate responses with a Lafa model on one GPU. Arguments go to
# scripts/generate.py, for example:
#
#   sbatch slurm/infer.sbatch --model lafa-hf/lafa-7B --gen-mode speculative \
#       --prompt "What is 17 + 25?"
#   sbatch slurm/infer.sbatch --model lafa-hf/lafa-0.5B --gen-mode flow --num-steps 4 \
#       --prompt-file prompts.txt --output output/infer/flow.jsonl

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"

python scripts/generate.py "$@"
