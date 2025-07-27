#!/bin/bash

# ==============================================================================
# Raw QA Task Configuration
# Configuration for raw question-answering fine-tuning task
# ==============================================================================

# Task-specific configuration
TASK_NAME="raw_qa"
DATASET_NAME="raw_qa"

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Training hyperparameters
LEARNING_RATE=2e-5
NUM_EPOCHS=3

# Output configuration
OUTPUT_BASE_DIR="experiments/sft/results/lr_2-5"
RUN_NAME="qwen2vl-${TASK_NAME}_sft"

# Additional training arguments
MAX_PIXELS=90000
MIN_PIXELS=784
MODEL_MAX_LENGTH=8192
SAVE_STEPS=10000
SAVE_TOTAL_LIMIT=2

echo "Task configuration loaded: ${TASK_NAME}"

# ==============================================================================
# Task Description:
# ==============================================================================
# Raw QA task focuses on basic question-answering capabilities without
# additional cognitive mapping or reasoning structures. This is the baseline
# task for evaluating fundamental VLM understanding.
# ==============================================================================

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