#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_offline_disagg_multinode.sh [options]

Run this from inside a 2-node SLURM allocation. It starts a Ray head on the
first node, joins the second node to the same cluster, and launches offline
disaggregated inference with configurable tensor and pipeline parallelism.

Options:
  --repo-dir PATH              DistServe repo path
  --model-dir PATH             Local model path
  --tokenizer-dir PATH         Local tokenizer path (defaults to --model-dir)
  --conda-env NAME             Conda env to activate (default: distserve)
  --ray-port PORT              Ray head port (default: 6379)
  --gpus-per-node N            GPUs per node to expose to Ray (default: 4)
  --context-tp N               Context stage tensor parallel size (default: 2)
  --context-pp N               Context stage pipeline parallel size (default: 2)
  --decoding-tp N              Decoding stage tensor parallel size (default: 2)
  --decoding-pp N              Decoding stage pipeline parallel size (default: 2)
  --sync-root PATH             Shared filesystem root for coordination files
  -h, --help                   Show this help
EOF
}

REPO_DIR=/pscratch/sd/h/hmuki/DistServe
MODEL_DIR=/pscratch/sd/h/hmuki/models/facebook/opt-1.3b
TOKENIZER_DIR=""
CONDA_ENV=distserve
RAY_PORT=6379
GPUS_PER_NODE=4
CONTEXT_TP=2
CONTEXT_PP=2
DECODING_TP=2
DECODING_PP=2
SYNC_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --tokenizer-dir)
      TOKENIZER_DIR="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --ray-port)
      RAY_PORT="$2"
      shift 2
      ;;
    --gpus-per-node)
      GPUS_PER_NODE="$2"
      shift 2
      ;;
    --context-tp)
      CONTEXT_TP="$2"
      shift 2
      ;;
    --context-pp)
      CONTEXT_PP="$2"
      shift 2
      ;;
    --decoding-tp)
      DECODING_TP="$2"
      shift 2
      ;;
    --decoding-pp)
      DECODING_PP="$2"
      shift 2
      ;;
    --sync-root)
      SYNC_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "This script must be run inside a SLURM allocation." >&2
  exit 1
fi

if [[ -z "$TOKENIZER_DIR" ]]; then
  TOKENIZER_DIR="$MODEL_DIR"
fi

if [[ -z "$SYNC_ROOT" ]]; then
  SYNC_ROOT="$REPO_DIR/.ray-sync"
fi

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${#NODES[@]} -ne 2 ]]; then
  echo "This launcher expects exactly 2 nodes, but found ${#NODES[@]}." >&2
  exit 1
fi

head_node="${NODES[0]}"
worker_node="${NODES[1]}"
SYNC_DIR="$SYNC_ROOT/$SLURM_JOB_ID"
mkdir -p "$SYNC_DIR"

HEAD_IP_FILE="$SYNC_DIR/head_ip"
HEAD_READY="$SYNC_DIR/head_ready"
WORKER_READY="$SYNC_DIR/worker_ready"
DONE_FILE="$SYNC_DIR/done"
REMOTE_SCRIPT="$SYNC_DIR/run_remote.sh"

export REPO_DIR MODEL_DIR TOKENIZER_DIR CONDA_ENV RAY_PORT GPUS_PER_NODE
export CONTEXT_TP CONTEXT_PP DECODING_TP DECODING_PP
export HEAD_IP_FILE HEAD_READY WORKER_READY DONE_FILE

cat > "$REMOTE_SCRIPT" <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV}"
fi

if [[ "${SLURM_PROCID}" -eq 0 ]]; then
  head_ip=$(hostname -I | awk '{print $1}')
  echo "$head_ip" > "$HEAD_IP_FILE"

  ray stop -f >/dev/null 2>&1 || true
  ray start --head --node-ip-address="$head_ip" --port="$RAY_PORT" --num-gpus="$GPUS_PER_NODE" --disable-usage-stats

  touch "$HEAD_READY"
  until [[ -f "$WORKER_READY" ]]; do sleep 1; done

  cd "$REPO_DIR"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python examples/offline_disagg.py \
    --ray-address auto \
    --model "$MODEL_DIR" \
    --tokenizer "$TOKENIZER_DIR" \
    --context-tensor-parallel-size "$CONTEXT_TP" \
    --context-pipeline-parallel-size "$CONTEXT_PP" \
    --decoding-tensor-parallel-size "$DECODING_TP" \
    --decoding-pipeline-parallel-size "$DECODING_PP"

  touch "$DONE_FILE"
  ray stop -f >/dev/null 2>&1 || true
else
  until [[ -f "$HEAD_IP_FILE" ]]; do sleep 1; done
  head_ip=$(cat "$HEAD_IP_FILE")

  until [[ -f "$HEAD_READY" ]]; do sleep 1; done

  ray stop -f >/dev/null 2>&1 || true
  ray start --address="$head_ip:$RAY_PORT" --num-gpus="$GPUS_PER_NODE" --disable-usage-stats

  touch "$WORKER_READY"
  until [[ -f "$DONE_FILE" ]]; do sleep 2; done

  ray stop -f >/dev/null 2>&1 || true
fi
EOF

chmod +x "$REMOTE_SCRIPT"

echo "Launching from head node: $head_node"
echo "Joining worker node:     $worker_node"
echo "Sync directory:          $SYNC_DIR"

srun --label \
  --nodes=2 \
  --ntasks=2 \
  --ntasks-per-node=1 \
  --gpus-per-task="$GPUS_PER_NODE" \
  --exclusive \
  bash "$REMOTE_SCRIPT"