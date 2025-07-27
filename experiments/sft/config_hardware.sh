#!/bin/bash

# ==============================================================================
# Hardware Configuration for Qwen2.5-VL SFT Training
# Customize this file based on your GPU setup
# ==============================================================================

# Hardware configuration
GPU_DEVICES="0, 1, 2, 3"          # Modify based on available GPUs (e.g., "0,1,2,3" for 4 GPUs)
NUM_PROCESSES=4          # Should match number of GPUs
BATCH_SIZE=16             # Per-device batch size (adjust based on GPU memory)

# Calculate gradient accumulation steps to maintain total batch size of 512
# Formula: GRAD_ACCUM_STEPS = 512 / NUM_PROCESSES / BATCH_SIZE
GRAD_ACCUM_STEPS=$((1024 / NUM_PROCESSES / BATCH_SIZE))

# ==============================================================================
# Common Hardware Configurations:
# ==============================================================================

# Single H100 (80GB):
# GPU_DEVICES="0"
# NUM_PROCESSES=1
# BATCH_SIZE=4
# -> GRAD_ACCUM_STEPS=128

# Quad H100 (80GB each):
# GPU_DEVICES="0,1,2,3"
# NUM_PROCESSES=4
# BATCH_SIZE=4
# -> GRAD_ACCUM_STEPS=32

# ==============================================================================
# Memory Optimization Tips:
# ==============================================================================

# If you encounter OOM (Out of Memory) errors:
# 1. Reduce BATCH_SIZE (e.g., from 4 to 2 or 1)
# 2. The GRAD_ACCUM_STEPS will automatically adjust to maintain total batch size
# 3. Enable gradient checkpointing (already enabled in train script)

# ==============================================================================
# Validation
# ==============================================================================

# Validate configuration
if [ $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS)) -ne 512 ]; then
    echo "⚠️  Warning: Total batch size is not 512!"
    echo "   Current: BATCH_SIZE($BATCH_SIZE) × NUM_PROCESSES($NUM_PROCESSES) × GRAD_ACCUM_STEPS($GRAD_ACCUM_STEPS) = $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))"
    echo "   Expected: 512"
fi

echo "Hardware configuration loaded:"
echo "  - GPUs: ${GPU_DEVICES}"
echo "  - Processes: ${NUM_PROCESSES}"
echo "  - Batch size per device: ${BATCH_SIZE}"
echo "  - Gradient accumulation steps: ${GRAD_ACCUM_STEPS}"
echo "  - Total effective batch size: $((BATCH_SIZE * NUM_PROCESSES * GRAD_ACCUM_STEPS))" 