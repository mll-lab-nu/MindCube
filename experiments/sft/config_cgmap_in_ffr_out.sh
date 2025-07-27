#!/bin/bash

# ==============================================================================
# Cognitive Map Input + Free-Form Reasoning Output Task Configuration
# Configuration for cognitive map input with free-form reasoning output fine-tuning
# ==============================================================================

# Task-specific configuration
TASK_NAME="cgmap_in_ffr_out"
DATASET_NAME="cgmap_in_ffr_out"

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
SAVE_STEPS=10000
SAVE_TOTAL_LIMIT=2

echo "Task configuration loaded: ${TASK_NAME}"

# ==============================================================================
# Task Description:
# ==============================================================================
# Cognitive Map Input + Free-Form Reasoning Output combines structured cognitive
# map inputs with flexible reasoning output generation. This task trains models
# to process structured cognitive representations and generate natural language
# reasoning responses.
# ============================================================================== 