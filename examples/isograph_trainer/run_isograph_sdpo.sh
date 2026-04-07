#!/usr/bin/env bash
# =============================================================================
# IsoGraph SDPO — Training Launch Script
#
# Integration: Members A + B + C
#
# Usage:
#
#   [A] Dummy environment (no Member C needed):
#     MODEL_NAME=/path/to/model bash examples/isograph_trainer/run_isograph_sdpo.sh
#
#   [B] Member C's real VE-MDP environment (recommended for full integration):
#     MODEL_NAME=/path/to/model \
#     ISOGRAPH_C_ROOT=/path/to/ISOGraph-C \
#     ISOGRAPH_ORACLE_GRAPH_DIR=/path/to/data-B \
#     USE_DUMMY_ENV=false \
#         bash examples/isograph_trainer/run_isograph_sdpo.sh
#
#   [C] Member C with single oracle graph:
#     MODEL_NAME=/path/to/model \
#     ISOGRAPH_C_ROOT=/path/to/ISOGraph-C \
#     ORACLE_GRAPH_PATH=/path/to/global_oracle_graph_demo.json \
#     USE_DUMMY_ENV=false \
#         bash examples/isograph_trainer/run_isograph_sdpo.sh
#
#   Override parameters via environment variables:
#     MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
#     N_GPUS=4 \
#     DATA_DIR=/path/to/my/parquets \
#         bash examples/isograph_trainer/run_isograph_sdpo.sh
#
# Environment Variables:
#   MODEL_NAME                  — HuggingFace model path or local directory
#   LR                         — Learning rate (default: 1e-6)
#   ROLLOUT_N                  — Number of rollouts per update (default: 4)
#   TOTAL_EPOCHS              — Training epochs (default: 3)
#   DATA_DIR                   — Directory containing train.parquet + val.parquet
#   N_GPUS / N_NODES          — GPU/node configuration
#   SAVE_FREQ / TEST_FREQ     — Checkpoint and validation frequency
#   USE_WANDB                 — Use Weights & Biases logging (default: false)
#   ISOGRAPH_C_ROOT            — Path to ISOGraph-C package (for Member C)
#   ISOGRAPH_ORACLE_GRAPH_DIR — Path to Member B's data-B directory (page_*.json)
#   ORACLE_GRAPH_PATH         — Single oracle graph JSON file
#   IMAGE_PATH                — Source image (for Member C zoom action)
#   USE_DUMMY_ENV             — "true" (DummyEnvironment) or "false" (Member C VE-MDP)
#   SVM_BACKEND               — "dummy" or "onnx" (for Member C)
# =============================================================================

set -xeuo pipefail

# 【修改1：防强杀护盾】关闭 Ray 的内存监控强杀，保住系统 RAM 的命！
export RAY_memory_usage_threshold=1.0
export RAY_memory_monitor_refresh_ms=0

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_P2P_DISABLE=1

# =============================================================================
# Default Configuration
# =============================================================================

# Model
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"

# Training
LR="${LR:-1e-6}"
ROLLOUT_N="${ROLLOUT_N:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"

# Data (parquet files)
DATA_DIR="${DATA_DIR:-/home/aisuan/SDPO/data}"

# Resources
N_GPUS="${N_GPUS:-1}"
N_NODES="${N_NODES:-1}"

# Checkpointing / Logging
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-5}"
PROJECT_NAME="${PROJECT_NAME:-isograph_sdpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2_5_vl_7b_isograph}"
USE_WANDB="${USE_WANDB:-false}"

# IsoGraph environment configuration
USE_DUMMY_ENV="${USE_DUMMY_ENV:-false}"

# 【修改2：Member C 路径】必须指向父目录 /home/aisuan，这样 Python 才能 import ISOGraph_C
ISOGRAPH_C_ROOT="${ISOGRAPH_C_ROOT:-/home/aisuan}"
ISOGRAPH_ORACLE_GRAPH_DIR="${ISOGRAPH_ORACLE_GRAPH_DIR:-/home/aisuan/data-B}"
ORACLE_GRAPH_PATH="${ORACLE_GRAPH_PATH:-}"
IMAGE_PATH="${IMAGE_PATH:-}"
SVM_BACKEND="${SVM_BACKEND:-dummy}"
SVM_MODEL_PATH="${SVM_MODEL_PATH:-}"

# =============================================================================
# SDPO Root
# =============================================================================

# Script is at .../SDPO/examples/isograph_trainer/run_isograph_sdpo.sh
SDPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SDPO_ROOT

# =============================================================================
# Resolve Data Files
# =============================================================================

if [ -n "$DATA_DIR" ] && [ -f "${DATA_DIR}/train.parquet" ] && [ -f "${DATA_DIR}/val.parquet" ]; then
    TRAIN_FILES="${DATA_DIR}/train.parquet"
    VAL_FILES="${DATA_DIR}/val.parquet"
    echo "[INFO] Using data from: ${DATA_DIR}"
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
# Resolve Oracle Graph
# =============================================================================

if [ "$USE_DUMMY_ENV" = "false" ]; then
    if [ -n "$ISOGRAPH_ORACLE_GRAPH_DIR" ]; then
        echo "[INFO] Using Member B's oracle graph directory: ${ISOGRAPH_ORACLE_GRAPH_DIR}"
        ORACLE_GRAPH_ARG="isograph.oracle_graph_dir=${ISOGRAPH_ORACLE_GRAPH_DIR}"
        ORACLE_GRAPH_PATH=""
    elif [ -n "$ORACLE_GRAPH_PATH" ]; then
        echo "[INFO] Using single oracle graph: ${ORACLE_GRAPH_PATH}"
        ORACLE_GRAPH_ARG="isograph.oracle_graph_path=${ORACLE_GRAPH_PATH}"
    else
        echo "[WARNING] USE_DUMMY_ENV=false but no oracle graph specified."
        echo "[WARNING] Falling back to DummyEnvironment."
        USE_DUMMY_ENV="true"
        ORACLE_GRAPH_ARG="isograph.use_dummy_env=true"
    fi
else
    ORACLE_GRAPH_ARG=""
    if [ -z "$ORACLE_GRAPH_PATH" ]; then
        ORACLE_GRAPH_PATH="${SDPO_ROOT}/global_oracle_graph_demo.json"
    fi
fi

# =============================================================================
# Resolve Member C ISOGraph-C package
# =============================================================================

if [ "$USE_DUMMY_ENV" = "false" ]; then
    if [ -z "$ISOGRAPH_C_ROOT" ]; then
        ISOGRAPH_C_ROOT="${SDPO_ROOT}/../ISOGraph_C"
        if [ -d "$ISOGRAPH_C_ROOT" ]; then
            echo "[INFO] Auto-detected ISOGraph_C at: ${ISOGRAPH_C_ROOT}"
        else
            echo "[WARNING] USE_DUMMY_ENV=false but ISOGRAPH_C_ROOT is not set."
        fi
    else
        echo "[INFO] Using ISOGraph_C from: ${ISOGRAPH_C_ROOT}"
    fi

    if [ -n "$ISOGRAPH_C_ROOT" ]; then
        export ISOGRAPH_C_ROOT
        export PYTHONPATH="${ISOGRAPH_C_ROOT}:${PYTHONPATH:-}"
    fi
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
PYTHONPATH="${PYTHONPATH:-}"
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
echo "IsoGraph SDPO Training Launch (Members A + B + C)"
echo "========================================================================"
echo "  MODEL_NAME:              ${MODEL_NAME}"
echo "  N_GPUS:                 ${N_GPUS}"
echo "  N_NODES:                ${N_NODES}"
if [ -n "$DATA_DIR" ]; then
    echo "  DATA_DIR:               ${DATA_DIR}"
else
    echo "  DATA_DIR:               [demo parquet files]"
fi
if [ "$USE_DUMMY_ENV" = "true" ]; then
    echo "  ENV BACKEND:            DummyEnvironment"
    echo "  ORACLE_GRAPH:          ${ORACLE_GRAPH_PATH}"
else
    echo "  ENV BACKEND:            Member C IsoGraphEnvironment"
    if [ -n "$ISOGRAPH_ORACLE_GRAPH_DIR" ]; then
        echo "  ORACLE_GRAPH_DIR:       ${ISOGRAPH_ORACLE_GRAPH_DIR}"
    fi
    if [ -n "$ISOGRAPH_C_ROOT" ]; then
        echo "  ISOGRAPH_C_ROOT:        ${ISOGRAPH_C_ROOT}"
    fi
    echo "  SVM_BACKEND:            ${SVM_BACKEND}"
fi
if [ -n "$IMAGE_PATH" ]; then
    echo "  IMAGE_PATH:             ${IMAGE_PATH}"
fi
echo "  TOTAL_EPOCHS:           ${TOTAL_EPOCHS}"
echo "  ROLLOUT_N:              ${ROLLOUT_N}"
echo "  LR:                    ${LR}"
echo "  PROJECT_NAME:          ${PROJECT_NAME}"
echo "  EXPERIMENT_NAME:       ${EXPERIMENT_NAME}"
echo "  USE_WANDB:             ${USE_WANDB}"
echo "========================================================================"

# =============================================================================
# Build Hydra command
# =============================================================================

set -x

HYDRA_ARGS=(
    # Config
    --config-name=isograph_sdpo

    # Model
    actor_rollout_ref.model.path="${MODEL_NAME}"
    +actor_rollout_ref.model.override_config.attn_implementation="sdpa"

    # Training
    actor_rollout_ref.actor.optim.lr="${LR}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.n_gpus_per_node="${N_GPUS}"
    trainer.nnodes="${N_NODES}"
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.logger="${LOGGER}"

    # Data
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    
    # 扩大 Pipeline 管道
    ++data.max_prompt_length=8192
    ++data.max_response_length=2048
    ++data.filter_overlong_prompts=false

    # IsoGraph environment
    isograph.use_dummy_env="${USE_DUMMY_ENV}"
    isograph.svm_backend="${SVM_BACKEND}"

    # Hardware
    trainer.device="${ACCELERATOR}"
    actor_rollout_ref.rollout.name=hf
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.actor.use_torch_compile=false
    actor_rollout_ref.ref.use_torch_compile=false
    actor_rollout_ref.actor.fsdp_config.param_offload=false
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
    actor_rollout_ref.ref.fsdp_config.param_offload=false
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=false

    actor_rollout_ref.actor.use_kl_loss=false

    # IsoGraph SDPO hyperparameters
    actor_rollout_ref.actor.policy_loss.isograph.ema_decay=0.99
    actor_rollout_ref.actor.policy_loss.isograph.beta=0.01
    actor_rollout_ref.actor.policy_loss.isograph.clip_ratio=0.2

    actor_rollout_ref.actor.strategy=fsdp2

    # 【修改3：极限榨干显存的 4 板斧】防止单卡 VRAM 爆炸
    ++actor_rollout_ref.model.enable_gradient_checkpointing=true
    ++actor_rollout_ref.model.override_config.max_pixels=921600
    ++actor_rollout_ref.model.target_modules="all-linear"
    ++actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048
    ++actor_rollout_ref.actor.ppo_mini_batch_size=16
)

# Conditionally add oracle graph path
if [ -n "$ORACLE_GRAPH_PATH" ]; then
    HYDRA_ARGS+=(isograph.oracle_graph_path="${ORACLE_GRAPH_PATH}")
fi

# Conditionally add oracle graph directory
if [ -n "$ISOGRAPH_ORACLE_GRAPH_DIR" ]; then
    HYDRA_ARGS+=(isograph.oracle_graph_dir="${ISOGRAPH_ORACLE_GRAPH_DIR}")
fi

# Conditionally add image path
if [ -n "$IMAGE_PATH" ]; then
    HYDRA_ARGS+=(isograph.image_path="${IMAGE_PATH}")
fi

# Conditionally add SVM model path
if [ -n "$SVM_MODEL_PATH" ]; then
    HYDRA_ARGS+=(isograph.svm_model_path="${SVM_MODEL_PATH}")
fi

python -m verl.trainer.train_isograph_sdpo "${HYDRA_ARGS[@]}" "$@"

set +x

echo "========================================================================"
echo "Training finished (or exited with error)."
echo "Logs: ${SDPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/"
echo "========================================================================"