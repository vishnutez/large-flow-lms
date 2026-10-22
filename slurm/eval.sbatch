#!/bin/bash
#SBATCH --job-name=lafa-eval
#SBATCH --partition=def
#SBATCH --account=prj-02-llm-reasoning-kalathil
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=output/slurm/%x-%j.out

# Run an lm-evaluation-harness benchmark on one GPU (see docs/evaluation.md).
# Settings come from environment variables, passed with --export:
#
#   MODEL       HF repo ID or local checkpoint      (default lafa-hf/lafa-0.5B)
#   TASK        lm-eval task                         (default ifeval)
#   FEWSHOT     few-shot count                       (default 0)
#   CHAT        1 to apply the chat template         (default 1)
#   GEN_ARGS    decoding part of --model_args        (default flow, 4 steps, cfg 8)
#   MAX_TOKENS  generation budget                    (default 1024)
#   EXTRA       extra lm-eval flags, e.g. "--limit 50 --confirm_run_unsafe_code"
#
#   sbatch --export=ALL,MODEL=lafa-hf/lafa-7B,TASK=gsm8k_cot,GEN_ARGS=gen_mode=speculative \
#       slurm/eval.sbatch

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"

MODEL="${MODEL:-lafa-hf/lafa-0.5B}"
TASK="${TASK:-ifeval}"
FEWSHOT="${FEWSHOT:-0}"
CHAT="${CHAT:-1}"
GEN_ARGS="${GEN_ARGS:-gen_mode=flow,num_steps=4,cfg_scale=8,noise_scale_override=0.1}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
EXTRA="${EXTRA:-}"

TAG="$(basename "$MODEL")-$TASK-$(echo "$GEN_ARGS" | tr ',=' '_-')"
OUT="output/eval/$TAG/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
echo "model=$MODEL task=$TASK gen=$GEN_ARGS -> $OUT"

CHAT_FLAG=()
[[ "$CHAT" == 1 ]] && CHAT_FLAG=(--apply_chat_template)

# shellcheck disable=SC2086
python -m src.evaluation.harness \
    --model lafa --tasks "$TASK" --num_fewshot "$FEWSHOT" "${CHAT_FLAG[@]}" \
    --gen_kwargs "max_gen_toks=$MAX_TOKENS,do_sample=false,temperature=0.0" \
    --output_path "$OUT" \
    --model_args "pretrained=$MODEL,$GEN_ARGS,max_new_tokens=$MAX_TOKENS,max_length=4096" \
    $EXTRA
