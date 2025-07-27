#!/bin/bash

# ==============================================================================
# Augmented Cognitive Map + Free-Form Reasoning Output Task Configuration
# Configuration for cognitive map with free-form reasoning output fine-tuning
# ==============================================================================

# Task-specific configuration
TASK_NAME="aug_cgmap_ffr_out"
DATASET_NAME="aug_cgmap_ffr_out"

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"

# Training hyperparameters
LEARNING_RATE=1e-5
NUM_EPOCHS=3

# Output configuration
OUTPUT_BASE_DIR="checkpoints/sft"
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
# Augmented Cognitive Map + Free-Form Reasoning Output combines structured
# cognitive mapping with flexible reasoning output generation. This task trains
# models to understand complex visual relationships and express reasoning in
# natural language format.
# ============================================================================== 