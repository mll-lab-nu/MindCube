#!/bin/bash

# ==============================================================================
# Augmented Cognitive Map Input Task Configuration
# Configuration for augmented cognitive map input fine-tuning task
# ==============================================================================

# Task-specific configuration
TASK_NAME="aug_cgmap_in"
DATASET_NAME="aug_cgmap_in"

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
# Augmented Cognitive Map Input task focuses on processing enhanced cognitive
# map inputs with additional contextual information. This task trains models
# to understand and utilize augmented cognitive structures for better reasoning.
# ============================================================================== 