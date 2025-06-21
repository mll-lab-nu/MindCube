#!/bin/bash

# Script to run inference for all specified tasks on Qwen2.5-VL model
# Uses 4 GPUs (0-3) with nohup for background execution
# Author: MindCube Team
# Usage: bash scripts/bash_scripts/run_frozen_vlm_all_tasks_qwen.sh

echo "🚀 Starting frozen VLM inference for all tasks on Qwen2.5-VL..."
echo "📅 Start time: $(date)"

# Configuration
MODEL_TYPE="qwen2.5vl"
INPUT_DIR="./data/prompts/general"
OUTPUT_DIR="./data/results/frozen_vlm"
LOG_DIR="./logs/inference"

# Task list (6 tasks total)
TASKS=(
    # "raw_qa"
    # "aug_cgmap_in" 
    # "ff_rsn"
    "aug_cgmap_ffr_out"
    # "plain_cgmap_ffr_out"
    # "cgmap_in_ffr_out"
)

# Create necessary directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "📁 Input directory: ${INPUT_DIR}"
echo "📁 Output directory: ${OUTPUT_DIR}"
echo "📁 Log directory: ${LOG_DIR}"
echo "🎯 Model type: ${MODEL_TYPE}"
echo "📋 Tasks to run: ${#TASKS[@]} tasks"

# Display task list
for i in "${!TASKS[@]}"; do
    gpu_id=$((i % 4))
    echo "  ${TASKS[i]} -> GPU ${gpu_id}"
done

echo ""

# Function to run inference for a single task
run_task_inference() {
    local task_name=$1
    local gpu_id=$2
    local input_file="${INPUT_DIR}/MindCube_tinybench_${task_name}.jsonl"
    local log_file="${LOG_DIR}/inference_${task_name}_gpu${gpu_id}.log"
    
    echo "🔧 [GPU ${gpu_id}] Starting inference for task: ${task_name}"
    echo "📝 [GPU ${gpu_id}] Input file: ${input_file}"
    echo "📋 [GPU ${gpu_id}] Log file: ${log_file}"
    
    # Check if input file exists
    if [ ! -f "${input_file}" ]; then
        echo "❌ [GPU ${gpu_id}] Error: Input file not found: ${input_file}"
        return 1
    fi
    
    # Run inference with nohup in background
    nohup env CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/run_inference.py \
        --model-type "${MODEL_TYPE}" \
        --input-file "${input_file}" \
        --output-dir "${OUTPUT_DIR}" \
        > "${log_file}" 2>&1 &
    
    local pid=$!
    echo "✅ [GPU ${gpu_id}] Task ${task_name} started with PID: ${pid}"
    
    # Save PID for monitoring
    echo "${pid}" > "${LOG_DIR}/pid_${task_name}_gpu${gpu_id}.txt"
    
    return 0
}

# Start all tasks
echo "🎬 Starting all inference tasks..."
echo ""

for i in "${!TASKS[@]}"; do
    task_name="${TASKS[i]}"
    gpu_id=$((i % 4))
    
    run_task_inference "${task_name}" "${gpu_id}"
    
    # Small delay between task starts to avoid resource conflicts
    sleep 2
done

echo ""
echo "🎯 All tasks have been launched!"
echo "📊 Summary:"
echo "  - Total tasks: ${#TASKS[@]}"
echo "  - GPUs used: 4 (GPU 0-3)"
echo "  - Model: ${MODEL_TYPE}"
echo "  - Input directory: ${INPUT_DIR}"
echo "  - Output directory: ${OUTPUT_DIR}"
echo "  - Log directory: ${LOG_DIR}"

echo ""
echo "📋 To monitor progress:"
echo "  # Check running processes:"
echo "  ps aux | grep run_inference.py"
echo ""
echo "  # Check logs for specific task:"
echo "  tail -f ${LOG_DIR}/inference_<task_name>_gpu<gpu_id>.log"
echo ""
echo "  # Example log files:"
for i in "${!TASKS[@]}"; do
    task_name="${TASKS[i]}"
    gpu_id=$((i % 4))
    echo "  tail -f ${LOG_DIR}/inference_${task_name}_gpu${gpu_id}.log"
done

echo ""
echo "📈 To check completion status:"
echo "  # Count completed tasks:"
echo "  ls ${OUTPUT_DIR}/*_responses.jsonl | wc -l"
echo ""
echo "  # List completed output files:"
echo "  ls -la ${OUTPUT_DIR}/"

echo ""
echo "⏰ Script started at: $(date)"
echo "🔄 All tasks are now running in background with nohup..."
echo "✅ Script execution completed. Check logs for progress updates."
