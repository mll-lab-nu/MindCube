#!/bin/bash

# ==============================================================================
# Plain Cognitive Map Output Task Configuration
# Configuration for plain cognitive map output fine-tuning task
# ==============================================================================

# Task-specific configuration
TASK_NAME="plain_cgmap_out"
DATASET_NAME="plain_cgmap_out"

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
# Plain Cognitive Map Output task focuses on generating basic cognitive
# representations from visual inputs. Models learn to create fundamental
# cognitive maps without additional augmentation or complex structures.
# ============================================================================== 