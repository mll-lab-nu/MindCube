#!/bin/bash

# Script to run vLLM inference for all specified tasks on Qwen2.5-VL model (Frozen VLM)
# Uses GPU queue management - one task per GPU, sequential execution
# Uses default pre-trained Qwen2.5-VL model (not fine-tuned checkpoint)
# Author: MindCube Team
# Usage: bash scripts/bash_scripts/run_vllm_frozen_vlm_all_tasks_qwen.sh

echo "🚀 Starting vLLM inference for all tasks on Frozen Qwen2.5-VL..."
echo "📅 Start time: $(date)"

# Configuration
MODEL_TYPE="qwen2.5vl"
BACKEND="vllm"
INPUT_DIR="./data/prompts/general"
OUTPUT_DIR="./data/results/frozen_vlm_vllm"
LOG_DIR="./logs/vllm_inference_frozen_vlm"
CONFIG_FILE="./configs/vllm_optimized.json"
BATCH_SIZE=1
# NOTE: No checkpoint path - using default pre-trained Qwen2.5-VL model

# Task list (6 tasks total)
TASKS=(
    "raw_qa"
    "aug_cgmap_in" 
    "ff_rsn"
    "aug_cgmap_ffr_out"
    "plain_cgmap_ffr_out"
    "cgmap_in_ffr_out"
)

# GPU configuration
AVAILABLE_GPUS=(0 1 2 3)  # Available GPU IDs
declare -A GPU_STATUS      # Track GPU status: 0=free, 1=busy
declare -A GPU_PIDS        # Track PIDs running on each GPU
declare -A TASK_STATUS     # Track task completion status

# Initialize GPU status
for gpu in "${AVAILABLE_GPUS[@]}"; do
    GPU_STATUS[$gpu]=0
    GPU_PIDS[$gpu]=""
done

# Initialize task status
for task in "${TASKS[@]}"; do
    TASK_STATUS[$task]="pending"
done

# Create necessary directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "📁 Input directory: ${INPUT_DIR}"
echo "📁 Output directory: ${OUTPUT_DIR}"
echo "📁 Log directory: ${LOG_DIR}"
echo "🎯 Model type: ${MODEL_TYPE}"
echo "🔧 Backend: ${BACKEND}"
echo "📋 Tasks to run: ${#TASKS[@]} tasks"
echo "🖥️  Available GPUs: ${AVAILABLE_GPUS[*]}"
echo "📊 Batch size: ${BATCH_SIZE}"
echo "🏷️  Model: Default pre-trained Qwen2.5-VL (Frozen VLM)"

echo ""

# Function to check if a process is still running
is_process_running() {
    local pid=$1
    if [ -z "$pid" ]; then
        return 1
    fi
    kill -0 "$pid" 2>/dev/null
    return $?
}

# Function to find a free GPU
find_free_gpu() {
    for gpu in "${AVAILABLE_GPUS[@]}"; do
        if [ "${GPU_STATUS[$gpu]}" -eq 0 ]; then
            echo $gpu
            return 0
        fi
    done
    echo ""
    return 1
}

# Function to update GPU status
update_gpu_status() {
    for gpu in "${AVAILABLE_GPUS[@]}"; do
        if [ "${GPU_STATUS[$gpu]}" -eq 1 ]; then
            local pid="${GPU_PIDS[$gpu]}"
            if ! is_process_running "$pid"; then
                echo "✅ [GPU $gpu] Task completed (PID: $pid)"
                GPU_STATUS[$gpu]=0
                GPU_PIDS[$gpu]=""
            fi
        fi
    done
}

# Function to run inference for a single task
run_task_inference() {
    local task_name=$1
    local gpu_id=$2
    local input_file="${INPUT_DIR}/MindCube_tinybench_${task_name}.jsonl"
    local log_file="${LOG_DIR}/vllm_frozen_vlm_${task_name}_gpu${gpu_id}.log"
    
    echo "🔧 [GPU ${gpu_id}] Starting vLLM inference for task: ${task_name}"
    echo "📝 [GPU ${gpu_id}] Input file: ${input_file}"
    echo "📋 [GPU ${gpu_id}] Log file: ${log_file}"
    
    # Check if input file exists
    if [ ! -f "${input_file}" ]; then
        echo "❌ [GPU ${gpu_id}] Error: Input file not found: ${input_file}"
        TASK_STATUS[$task_name]="failed"
        return 1
    fi
    
    # Run vLLM inference with nohup in background (NO --model-path, uses default)
    nohup env CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/run_inference.py \
        --model-type "${MODEL_TYPE}" \
        --backend "${BACKEND}" \
        --input-file "${input_file}" \
        --output-dir "${OUTPUT_DIR}" \
        --config "${CONFIG_FILE}" \
        --batch-size "${BATCH_SIZE}" \
        --verbose \
        > "${log_file}" 2>&1 &
    
    local pid=$!
    echo "✅ [GPU ${gpu_id}] Task ${task_name} started with PID: ${pid}"
    
    # Update GPU status
    GPU_STATUS[$gpu_id]=1
    GPU_PIDS[$gpu_id]=$pid
    TASK_STATUS[$task_name]="running"
    
    # Save PID for monitoring
    echo "${pid}" > "${LOG_DIR}/pid_${task_name}_gpu${gpu_id}.txt"
    
    return 0
}

# Function to get pending tasks
get_pending_tasks() {
    local pending_tasks=()
    for task in "${TASKS[@]}"; do
        if [ "${TASK_STATUS[$task]}" == "pending" ]; then
            pending_tasks+=("$task")
        fi
    done
    echo "${pending_tasks[@]}"
}

# Function to count running tasks
count_running_tasks() {
    local count=0
    for task in "${TASKS[@]}"; do
        if [ "${TASK_STATUS[$task]}" == "running" ]; then
            ((count++))
        fi
    done
    echo $count
}

# Function to count completed tasks
count_completed_tasks() {
    local count=0
    for task in "${TASKS[@]}"; do
        if [ "${TASK_STATUS[$task]}" == "completed" ]; then
            ((count++))
        fi
    done
    echo $count
}

# Function to mark completed tasks
mark_completed_tasks() {
    for task in "${TASKS[@]}"; do
        if [ "${TASK_STATUS[$task]}" == "running" ]; then
            # Updated output file pattern for default model
            local output_file="${OUTPUT_DIR}/MindCube_tinybench_${task}_qwen2.5vl_responses.jsonl"
            if [ -f "$output_file" ] && [ -s "$output_file" ]; then
                TASK_STATUS[$task]="completed"
                echo "🎉 Task ${task} completed successfully!"
            fi
        fi
    done
}

# Main execution loop
echo "🎬 Starting vLLM frozen VLM inference queue management..."
echo ""

# Initial task assignment
initial_assignments=0
for task in "${TASKS[@]}"; do
    if [ $initial_assignments -lt ${#AVAILABLE_GPUS[@]} ]; then
        gpu_id="${AVAILABLE_GPUS[$initial_assignments]}"
        run_task_inference "$task" "$gpu_id"
        ((initial_assignments++))
        sleep 3  # Small delay to avoid resource conflicts
    fi
done

echo ""
echo "🔄 Entering queue management loop..."

# Main monitoring loop
while true; do
    # Update GPU status
    update_gpu_status
    
    # Mark completed tasks
    mark_completed_tasks
    
    # Get pending tasks
    pending_tasks=($(get_pending_tasks))
    
    # Check if all tasks are completed
    completed_count=$(count_completed_tasks)
    if [ $completed_count -eq ${#TASKS[@]} ]; then
        echo "🎉 All tasks completed successfully!"
        break
    fi
    
    # Assign new tasks to free GPUs
    if [ ${#pending_tasks[@]} -gt 0 ]; then
        free_gpu=$(find_free_gpu)
        if [ -n "$free_gpu" ]; then
            next_task="${pending_tasks[0]}"
            echo "🔄 Assigning task ${next_task} to GPU ${free_gpu}"
            run_task_inference "$next_task" "$free_gpu"
            sleep 3
        fi
    fi
    
    # Status update
    running_count=$(count_running_tasks)
    pending_count=${#pending_tasks[@]}
    
    echo "📊 Status: Completed: ${completed_count}/${#TASKS[@]}, Running: ${running_count}, Pending: ${pending_count}"
    
    # Sleep before next check
    sleep 30
done

echo ""
echo "🎯 All vLLM frozen VLM inference tasks completed!"
echo "📊 Final Summary:"
echo "  - Total tasks: ${#TASKS[@]}"
echo "  - Completed tasks: $(count_completed_tasks)"
echo "  - GPUs used: ${#AVAILABLE_GPUS[@]} (GPU ${AVAILABLE_GPUS[*]})"
echo "  - Model: ${MODEL_TYPE} (Default pre-trained)"
echo "  - Backend: ${BACKEND}"
echo "  - Input directory: ${INPUT_DIR}"
echo "  - Output directory: ${OUTPUT_DIR}"
echo "  - Log directory: ${LOG_DIR}"

echo ""
echo "📋 Output files:"
ls -la "${OUTPUT_DIR}"/*_responses.jsonl 2>/dev/null || echo "  No output files found"

echo ""
echo "📈 Log files:"
ls -la "${LOG_DIR}"/vllm_frozen_vlm_*.log 2>/dev/null || echo "  No log files found"

echo ""
echo "⏰ Script completed at: $(date)"
echo "✅ All vLLM frozen VLM inference tasks finished successfully!" 