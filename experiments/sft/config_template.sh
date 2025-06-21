#!/bin/bash

# ==============================================================================
# Training Configuration Template
# Copy this file and modify for your specific task
# ==============================================================================

# Task-specific configuration
TASK_NAME="your_task_name"
DATASET_NAME="your_dataset"  # e.g., cog_reasoning_sft, reasoning_sft, full, etc.

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"  # Or other model variants

# Training hyperparameters
LEARNING_RATE=1e-5
BATCH_SIZE=4
GRAD_ACCUM_STEPS=32
NUM_EPOCHS=3

# Hardware configuration
GPU_DEVICES="0,1,2,3"  # Modify based on available GPUs
NUM_PROCESSES=4        # Should match number of GPUs

# Output configuration
OUTPUT_BASE_DIR="experiments/sft/results"
RUN_NAME="qwen2vl-${TASK_NAME}"

# Additional training arguments (optional)
MAX_PIXELS=90000
MIN_PIXELS=784
MODEL_MAX_LENGTH=8192
SAVE_STEPS=5
SAVE_TOTAL_LIMIT=12

# ==============================================================================
# Example configurations for different tasks:
# ==============================================================================

# For cognitive reasoning task:
# TASK_NAME="cog_reasoning"
# DATASET_NAME="cog_reasoning_sft"
# LEARNING_RATE=1e-5

# For base SFT task:
# TASK_NAME="base_sft"
# DATASET_NAME="full"
# LEARNING_RATE=2e-5

# For reasoning task:
# TASK_NAME="reasoning"
# DATASET_NAME="reasoning_sft"
# LEARNING_RATE=1e-5

# ==============================================================================
# Usage:
# 1. Copy this file: cp config_template.sh config_my_task.sh
# 2. Edit config_my_task.sh with your specific settings
# 3. Source the config: source config_my_task.sh
# 4. Run training: ./train_qwen_sft.sh
# ============================================================================== 