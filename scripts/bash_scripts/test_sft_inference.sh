#!/bin/bash

# Test script to verify SFT checkpoint inference setup
# Tests one checkpoint to ensure everything works before running the full batch
# Usage: bash scripts/bash_scripts/test_sft_inference.sh

echo "🧪 Testing SFT checkpoint inference setup..."
echo "📅 Start time: $(date)"

# Configuration
MODEL_TYPE="qwen2.5vl"
INPUT_DIR="./data/prompts/general"
OUTPUT_DIR="./data/results/sft"
LOG_DIR="./logs/sft_inference"
CHECKPOINT_BASE_DIR="./checkpoints/sft"

# Test parameters
TEST_TASK="raw_qa"
TEST_CHECKPOINT="checkpoint-5"

# Create directories
mkdir -p "${OUTPUT_DIR}/${TEST_TASK}"
mkdir -p "${LOG_DIR}"

# Check if test checkpoint exists
TEST_CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${TEST_TASK}/${TEST_CHECKPOINT}"
if [ ! -d "${TEST_CHECKPOINT_PATH}" ]; then
    echo "❌ Test checkpoint not found: ${TEST_CHECKPOINT_PATH}"
    echo "Available checkpoints for ${TEST_TASK}:"
    ls -la "${CHECKPOINT_BASE_DIR}/${TEST_TASK}/" | grep checkpoint
    exit 1
fi

# Check if test input file exists
TEST_INPUT_FILE="${INPUT_DIR}/MindCube_tinybench_${TEST_TASK}.jsonl"
if [ ! -f "${TEST_INPUT_FILE}" ]; then
    echo "❌ Test input file not found: ${TEST_INPUT_FILE}"
    exit 1
fi

echo "✅ Test setup validated:"
echo "  - Model type: ${MODEL_TYPE}"
echo "  - Task: ${TEST_TASK}"
echo "  - Checkpoint: ${TEST_CHECKPOINT}"
echo "  - Checkpoint path: ${TEST_CHECKPOINT_PATH}"
echo "  - Input file: ${TEST_INPUT_FILE}"
echo "  - Output directory: ${OUTPUT_DIR}/${TEST_TASK}"

# Run test inference
echo ""
echo "🚀 Running test inference..."

LOG_FILE="${LOG_DIR}/test_inference_${TEST_TASK}_${TEST_CHECKPOINT}.log"

python scripts/run_inference.py \
    --model-type "${MODEL_TYPE}" \
    --model-path "${TEST_CHECKPOINT_PATH}" \
    --input-file "${TEST_INPUT_FILE}" \
    --output-dir "${OUTPUT_DIR}/${TEST_TASK}" \
    --verbose \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "📋 Test completed. Check results:"
echo "  - Log file: ${LOG_FILE}"
echo "  - Output directory: ${OUTPUT_DIR}/${TEST_TASK}"
echo "  - Expected output pattern: *${TEST_CHECKPOINT}*responses.jsonl"

# Check if output file was created
echo ""
echo "📁 Output files in ${OUTPUT_DIR}/${TEST_TASK}:"
ls -la "${OUTPUT_DIR}/${TEST_TASK}/"

echo ""
echo "✅ Test script completed at: $(date)" 