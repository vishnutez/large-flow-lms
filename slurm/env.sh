# Shared setup for the sbatch scripts in this directory. Submit jobs from the
# repository root so that SLURM_SUBMIT_DIR points at it.

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ ! -f "$REPO/train_flow.py" ]]; then
    echo "Submit from the large-flow-lms repository root (got $REPO)" >&2
    exit 1
fi
cd "$REPO"

PY="$REPO/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "Missing $PY; create it with: uv venv --python 3.12 --managed-python .venv &&" \
         "uv pip install --python .venv/bin/python -r requirements.txt -c constraints-tested.txt" >&2
    exit 1
fi
export PATH="$REPO/.venv/bin:$PATH"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Model and dataset caches on scratch; the released Lafa checkpoints live here.
export HF_HOME="${HF_HOME:-/scratch/user/$USER/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/user/$USER/.cache/triton}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

NUM_GPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"

echo "[$(date '+%F %T')] job=${SLURM_JOB_ID:-local} node=$(hostname) gpus=$NUM_GPUS repo=$REPO"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
