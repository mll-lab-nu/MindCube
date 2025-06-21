#!/bin/bash

# ==============================================================================
# Qwen2.5-VL SFT Training Script
# Independent training script for different MindCube tasks
# ==============================================================================

# Check if a config file is provided
CONFIG_FILE=${1:-""}
if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
    echo "Loading configuration from: $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "Using default configuration"
    # Default configuration
    TASK_NAME="cog_reasoning"
    DATASET_NAME="cog_reasoning_sft"
    MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
    LEARNING_RATE=1e-5
    BATCH_SIZE=4
    GRAD_ACCUM_STEPS=32
    NUM_EPOCHS=3
    GPU_DEVICES="0,1,2,3"
    NUM_PROCESSES=4
    OUTPUT_BASE_DIR="experiments/sft/results"
    RUN_NAME="qwen2vl-baseline-${TASK_NAME}_sft"
    MAX_PIXELS=90000
    MIN_PIXELS=784
    MODEL_MAX_LENGTH=8192
    SAVE_STEPS=5
    SAVE_TOTAL_LIMIT=12
fi

# Change to MindCube project root directory (07_MindCube_new)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$NUM_PROCESSES}
export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

# DeepSpeed configuration (using absolute path from project root)
deepspeed=${PROJECT_ROOT}/Qwen2.5-VL/qwen-vl-finetune/scripts/zero3.json

# Training entry point (using absolute path from project root)
entry_file=${PROJECT_ROOT}/Qwen2.5-VL/qwen-vl-finetune/qwenvl/train/train_qwen.py

# Output directory
output_dir=${PROJECT_ROOT}/${OUTPUT_BASE_DIR}/${TASK_NAME}/

# Ensure output directory exists
mkdir -p ${output_dir}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${MODEL_NAME} \
    --dataset_use ${DATASET_NAME} \
    --data_flatten True \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${NUM_EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size $((BATCH_SIZE*2)) \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --max_pixels ${MAX_PIXELS} \
    --min_pixels ${MIN_PIXELS} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${RUN_NAME} \
    --report_to none"

# Print configuration for verification
echo "==============================================================================
Training Configuration:
- Task: ${TASK_NAME}
- Model: ${MODEL_NAME}
- Dataset: ${DATASET_NAME}
- Run name: ${RUN_NAME}
- Output directory: ${output_dir}
- Learning rate: ${LEARNING_RATE}
- Batch size: ${BATCH_SIZE}
- Gradient accumulation steps: ${GRAD_ACCUM_STEPS}
- Number of epochs: ${NUM_EPOCHS}
- GPUs: ${CUDA_VISIBLE_DEVICES}
- Number of processes per node: ${NPROC_PER_NODE}
=============================================================================="

# Verify that Qwen's data configuration has been patched with MindCube datasets
echo "Verifying MindCube datasets are available..."
cd ${PROJECT_ROOT}/experiments/sft
python patch_qwen_data.py verify
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ MindCube datasets not found in Qwen's data configuration!"
    echo "Please run the following command first to setup the environment:"
    echo "  cd ${PROJECT_ROOT}/experiments/sft"
    echo "  python patch_qwen_data.py patch"
    echo ""
    exit 1
fi

# Return to project root
cd ${PROJECT_ROOT}

# Launch training
echo "Starting training..."
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed!"
    exit 1
fi 