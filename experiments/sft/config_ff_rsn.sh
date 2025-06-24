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
NUM_EPOCHS=3

# Output configuration
OUTPUT_BASE_DIR="experiments/sft/results"
RUN_NAME="qwen2vl-${TASK_NAME}_sft"

# Additional training arguments
MAX_PIXELS=90000
MIN_PIXELS=784
MODEL_MAX_LENGTH=8192
SAVE_STEPS=5
SAVE_TOTAL_LIMIT=12

echo "Task configuration loaded: ${TASK_NAME}"

# ==============================================================================
# Task Description:
# ==============================================================================
# Free-Form Reasoning task focuses on developing flexible reasoning capabilities
# without predefined cognitive structures. Models learn to generate coherent
# reasoning chains in natural language format.
# ============================================================================== 