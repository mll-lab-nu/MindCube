#!/bin/bash

# ==============================================================================
# Free-Form Reasoning Task Configuration
# Configuration for free-form reasoning fine-tuning task
# ==============================================================================

# Task-specific configuration
TASK_NAME="ff_rsn"
DATASET_NAME="ff_rsn"

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Training hyperparameters
LEARNING_RATE=1e-5
BATCH_SIZE=4
GRAD_ACCUM_STEPS=32
NUM_EPOCHS=3

# Hardware configuration
GPU_DEVICES="0,1,2,3"
NUM_PROCESSES=4

# Output configuration
OUTPUT_BASE_DIR="checkpoints/sft"
RUN_NAME="qwen2vl-${TASK_NAME}_sft"

# Additional training arguments
MAX_PIXELS=90000
MIN_PIXELS=784
MODEL_MAX_LENGTH=8192
SAVE_STEPS=5
SAVE_TOTAL_LIMIT=12

echo "Configuration loaded for task: ${TASK_NAME}" 