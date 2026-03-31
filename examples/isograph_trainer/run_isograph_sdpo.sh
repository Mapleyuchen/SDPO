#!/usr/bin/env bash
# =============================================================================
# IsoGraph SDPO — Training Launch Script
#
# Usage:
#   # With local model (recommended for this machine):
#   MODEL_NAME=/home/mail-robo/IsoGraph/SDPO/QwenVL25/Qwen/Qwen2.5-VL-7B-Instruct \
#       bash examples/isograph_trainer/run_isograph_sdpo.sh
#
#   # With HuggingFace model (requires network):
#   MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
#       bash examples/isograph_trainer/run_isograph_sdpo.sh
#
#   # Override any parameter via env or Hydra override:
#   MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
#   N_GPUS=4 \
#   DATA_DIR=/path/to/my/parquets \
#       bash examples/isograph_trainer/run_isograph_sdpo.sh
# =============================================================================

set -xeuo pipefail

# =============================================================================
# Default Configuration
# =============================================================================

# Model
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"

# Training
LR="${LR:-1e-6}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"

# Data
DATA_DIR="${DATA_DIR:-}"   # If set, must contain train.parquet and val.parquet

# Resources
N_GPUS="${N_GPUS:-1}"
N_NODES="${N_NODES:-1}"

# Checkpointing / Logging
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-5}"
PROJECT_NAME="${PROJECT_NAME:-isograph_sdpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2_5_vl_7b_isograph}"
USE_WANDB="${USE_WANDB:-false}"

# IsoGraph specific
ORACLE_GRAPH_PATH="${ORACLE_GRAPH_PATH:-}"

# =============================================================================
# SDPO Root (the SDPO package root)
# =============================================================================

# SDPO package root: script is at .../SDPO/examples/isograph_trainer/run_isograph_sdpo.sh
# dirname/../.. gives us /home/mail-robo/IsoGraph/SDPO
SDPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SDPO_ROOT

# =============================================================================
# Resolve Data Files
# =============================================================================

if [ -n "$DATA_DIR" ] && [ -f "${DATA_DIR}/train.parquet" ] && [ -f "${DATA_DIR}/val.parquet" ]; then
    TRAIN_FILES="${DATA_DIR}/train.parquet"
    VAL_FILES="${DATA_DIR}/val.parquet"
else
    TRAIN_FILES="${SDPO_ROOT}/dummy_train.parquet"
    VAL_FILES="${SDPO_ROOT}/dummy_val.parquet"

    if [ ! -f "$TRAIN_FILES" ] || [ ! -f "$VAL_FILES" ]; then
        echo "[INFO] Generating dummy parquet files for pipeline testing..."
        python -c "
import json
import os
import pandas as pd

sdpo_root = os.environ.get('SDPO_ROOT', '.')

train_data = []
val_data = []
for i in range(32):
    row = {
        'prompt': json.dumps([{'role': 'user', 'content': f'Question {i}: What is 1+1? Answer with just the number.'}]),
        'reward_model': 1.0,
        'ground_truth': '2',
    }
    if i < 24:
        train_data.append(row)
    else:
        val_data.append(row)

train_df = pd.DataFrame(train_data)
val_df = pd.DataFrame(val_data)
train_df.to_parquet(os.path.join(sdpo_root, 'dummy_train.parquet'), engine='pyarrow', index=False)
val_df.to_parquet(os.path.join(sdpo_root, 'dummy_val.parquet'), engine='pyarrow', index=False)
print('[INFO] Dummy parquet files generated successfully.')
"
    else
        echo "[INFO] Using existing dummy parquet files."
    fi
fi

# =============================================================================
# Resolve Oracle Graph Path
# =============================================================================

if [ -z "$ORACLE_GRAPH_PATH" ]; then
    ORACLE_GRAPH_PATH="${SDPO_ROOT}/global_oracle_graph_demo.json"
fi

# =============================================================================
# Detect CUDA / NPU availability
# =============================================================================

if command -v nvidia-smi &> /dev/null; then
    ACCELERATOR="cuda"
    echo "[INFO] Detected CUDA GPUs:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null | head -n ${N_GPUS}
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    ACCELERATOR="npu"
    echo "[INFO] Detected NPU (Ascend)."
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
else
    ACCELERATOR="cuda"
    echo "[INFO] Assuming CUDA (nvidia-smi not found)."
fi

# =============================================================================
# Python path setup
# =============================================================================

LOCAL_LIB_PATH="${SDPO_ROOT}/local_lib/python3.11"
PYTHONPATH="${PYTHONPATH:-}"   # initialize so set -u doesn't break
if [ -d "$LOCAL_LIB_PATH" ]; then
    export PYTHONPATH="${LOCAL_LIB_PATH}:${SDPO_ROOT}:${SDPO_ROOT}/verl:${PYTHONPATH}"
else
    export PYTHONPATH="${SDPO_ROOT}:${SDPO_ROOT}/verl:${PYTHONPATH}"
fi

# =============================================================================
# Logger configuration
# =============================================================================

if [ "$USE_WANDB" = "true" ]; then
    if [ -z "$WANDB_API_KEY" ]; then
        echo "[WARNING] USE_WANDB=true but WANDB_API_KEY is not set. Falling back to console."
        LOGGER="console"
    else
        LOGGER="['console', 'wandb']"
    fi
else
    LOGGER="console"
fi

# =============================================================================
# Print Configuration
# =============================================================================

echo "========================================================================"
echo "IsoGraph SDPO Training Launch"
echo "========================================================================"
echo "  MODEL_NAME:        ${MODEL_NAME}"
echo "  N_GPUS:           ${N_GPUS}"
echo "  N_NODES:          ${N_NODES}"
if [ -n "$DATA_DIR" ]; then
    echo "  DATA_DIR:         ${DATA_DIR}"
else
    echo "  DATA_DIR:         [using demo json]"
fi
echo "  ORACLE_GRAPH:      ${ORACLE_GRAPH_PATH}"
echo "  TOTAL_EPOCHS:     ${TOTAL_EPOCHS}"
echo "  ROLLOUT_N:        ${ROLLOUT_N}"
echo "  LR:               ${LR}"
echo "  PROJECT_NAME:     ${PROJECT_NAME}"
echo "  EXPERIMENT_NAME:  ${EXPERIMENT_NAME}"
echo "  USE_WANDB:        ${USE_WANDB}"
echo "========================================================================"
if [ -z "$DATA_DIR" ]; then
    echo "[WARNING] DATA_DIR not set. Training will use DummyEnvironment with demo json."
    echo "[WARNING] To train on real data, set DATA_DIR to a directory containing:"
    echo "           - train.parquet (training samples)"
    echo "           - val.parquet   (validation samples)"
fi

# =============================================================================
# Build Hydra command
# =============================================================================

set -x

python -m verl.trainer.train_isograph_sdpo \
    --config-name=isograph_sdpo \
    actor_rollout_ref.model.path="${MODEL_NAME}" \
    +actor_rollout_ref.model.override_config.attn_implementation="sdpa" \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.actor.policy_loss.isograph.ema_decay=0.99 \
    actor_rollout_ref.actor.policy_loss.isograph.beta=0.01 \
    actor_rollout_ref.actor.policy_loss.isograph.clip_ratio=0.2 \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes="${N_NODES}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="${LOGGER}" \
    isograph.oracle_graph_path="${ORACLE_GRAPH_PATH}" \
    isograph.use_dummy_env=true \
    trainer.device="${ACCELERATOR}" \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.ref.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=true \
    "$@"

set +x

echo "========================================================================"
echo "Training finished (or exited with error)."
echo "Logs: ${SDPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/"
echo "========================================================================"
