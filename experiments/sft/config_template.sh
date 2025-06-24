#!/bin/bash

# ==============================================================================
# Training Configuration Template
# Copy this file and modify for your specific task
# ==============================================================================

# Task-specific configuration
TASK_NAME="your_task_name"
DATASET_NAME="your_dataset"  # e.g., raw_qa, aug_cgmap_out, ff_rsn, etc.

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"  # Or other model variants

# Training hyperparameters
LEARNING_RATE=1e-5
NUM_EPOCHS=3

# Output configuration
OUTPUT_BASE_DIR="experiments/sft/results"
RUN_NAME="qwen2vl-${TASK_NAME}_sft"

# Additional training arguments (optional)
MAX_PIXELS=90000
MIN_PIXELS=784
MODEL_MAX_LENGTH=8192
SAVE_STEPS=5
SAVE_TOTAL_LIMIT=12

echo "Task configuration loaded: ${TASK_NAME}"

# ==============================================================================
# Available MindCube Datasets:
# ==============================================================================

# raw_qa              - Raw question-answering task
# aug_cgmap_in         - Augmented cognitive map input task  
# ff_rsn               - Free-form reasoning task
# aug_cgmap_ffr_out    - Augmented cognitive map + free-form reasoning output
# plain_cgmap_ffr_out  - Plain cognitive map + free-form reasoning output
# cgmap_in_ffr_out     - Cognitive map input + free-form reasoning output

# ==============================================================================
# Example Task Configurations:
# ==============================================================================

# For raw QA task:
# TASK_NAME="raw_qa"
# DATASET_NAME="raw_qa"
# LEARNING_RATE=1e-5

# For free-form reasoning task:
# TASK_NAME="ff_rsn"
# DATASET_NAME="ff_rsn"
# LEARNING_RATE=1e-5

# For cognitive map reasoning:
# TASK_NAME="aug_cgmap_ffr_out"
# DATASET_NAME="aug_cgmap_ffr_out"
# LEARNING_RATE=1e-5

# ==============================================================================
# Usage:
# 1. Copy this file: cp config_template.sh config_my_task.sh
# 2. Edit config_my_task.sh with your specific task settings
# 3. Customize hardware settings in config_hardware.sh
# 4. Run training: ./train_qwen_sft.sh config_my_task.sh
# ============================================================================== 