#!/bin/bash

# Script to run inference for all SFT checkpoints on all tasks
# Dynamically detects available GPUs and manages task queues
# Each GPU runs maximum N tasks concurrently using round robin algorithm
# Author: MindCube Team
# Usage: bash scripts/bash_scripts/run_sft_ckpt_inference_qwen.sh [--max-tasks-per-gpu N]

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --max-tasks-per-gpu N    Maximum number of tasks per GPU (default: 1)"
    echo "  --help, -h               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                            # Use default (1 task per GPU)"
    echo "  $0 --max-tasks-per-gpu 2      # Allow 2 tasks per GPU"
    echo "  $0 --max-tasks-per-gpu 3      # Allow 3 tasks per GPU"
    echo ""
    echo "Note: This script processes all checkpoints for each task."
}

# Parse command line arguments
MAX_TASKS_PER_GPU=1  # Default value for better memory management
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-tasks-per-gpu)
            MAX_TASKS_PER_GPU="$2"
            if ! [[ "$MAX_TASKS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
                echo "❌ Error: --max-tasks-per-gpu must be a positive integer"
                exit 1
            fi
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Error: Unknown option $1"
            show_usage
            exit 1
            ;;
    esac
done

echo "🚀 Starting SFT checkpoint inference for all tasks (Enhanced Version)..."
echo "📅 Start time: $(date)"

# Configuration
MODEL_TYPE="qwen2.5vl"
INPUT_DIR="./data/prompts/general"
OUTPUT_DIR="./data/results/sft"
LOG_DIR="./logs/sft_inference"
CHECKPOINT_BASE_DIR="./checkpoints/sft"
MONITOR_INTERVAL=30  # seconds between status checks

# Detect available GPUs
echo "🔍 Detecting available GPUs..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ ${GPU_COUNT} -eq 0 ]; then
        echo "❌ No NVIDIA GPUs detected!"
        exit 1
    fi
else
    echo "⚠️  nvidia-smi not found, defaulting to 1 GPU"
    GPU_COUNT=1
fi

echo "🎮 Detected ${GPU_COUNT} GPU(s)"

# All available tasks (based on checkpoint directories)
TASKS=(
    "raw_qa"
    "plain_cgmap_out"
    "aug_cgmap_out"
    "ff_rsn"
    "aug_cgmap_ffr_out"
    "plain_cgmap_ffr_out"
)

# Create necessary directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "📁 Input directory: ${INPUT_DIR}"
echo "📁 Output directory: ${OUTPUT_DIR}"
echo "📁 Log directory: ${LOG_DIR}"
echo "📁 Checkpoint base directory: ${CHECKPOINT_BASE_DIR}"
echo "🎯 Model type: ${MODEL_TYPE}"
echo "📋 Tasks to process: ${#TASKS[@]} tasks"
echo "⚙️  Max tasks per GPU: ${MAX_TASKS_PER_GPU}"
echo "💾 Max concurrent tasks: $((GPU_COUNT * MAX_TASKS_PER_GPU))"

# Initialize GPU task counters
declare -A gpu_task_count
for ((i=0; i<GPU_COUNT; i++)); do
    gpu_task_count[$i]=0
done

# Task queue management
declare -a job_queue=()
declare -a running_jobs=()
declare -A job_gpu_mapping
declare -A job_pids

# Build job queue (all task-checkpoint combinations)
echo ""
echo "🔍 Discovering checkpoints and building job queue..."
total_jobs=0

for task in "${TASKS[@]}"; do
    echo "📋 Processing task: ${task}"
    
    # Get all checkpoints for this task
    task_checkpoint_dir="${CHECKPOINT_BASE_DIR}/${task}"
    if [ ! -d "${task_checkpoint_dir}" ]; then
        echo "⚠️  Warning: No checkpoint directory found for task ${task}"
        continue
    fi
    
    # Find and sort checkpoints
    checkpoints=($(find "${task_checkpoint_dir}" -name "checkpoint-*" -type d | sort -t'-' -k2 -n))
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "⚠️  Warning: No checkpoints found for task ${task}"
        continue
    fi
    
    echo "📁 Found ${#checkpoints[@]} checkpoints for task ${task}"
    
    for checkpoint_path in "${checkpoints[@]}"; do
        checkpoint_id=$(basename "${checkpoint_path}" | sed 's/checkpoint-//')
        job_id="${task}_checkpoint-${checkpoint_id}"
        job_queue+=("${job_id}:${task}:${checkpoint_path}:${checkpoint_id}")
        ((total_jobs++))
    done
done

echo "✅ Job queue built: ${total_jobs} total jobs"
echo ""

# Function to get next available GPU using round robin
get_next_gpu() {
    local min_tasks=999
    local selected_gpu=-1
    
    # Find GPU with minimum running tasks
    for ((i=0; i<GPU_COUNT; i++)); do
        if [ ${gpu_task_count[$i]} -lt ${MAX_TASKS_PER_GPU} ]; then
            if [ ${gpu_task_count[$i]} -lt ${min_tasks} ]; then
                min_tasks=${gpu_task_count[$i]}
                selected_gpu=$i
            fi
        fi
    done
    
    echo "${selected_gpu}"
}

# Function to run inference for a single checkpoint
run_checkpoint_inference() {
    local job_id=$1
    local task_name=$2
    local checkpoint_path=$3
    local checkpoint_id=$4
    local gpu_id=$5
    
    local input_file="${INPUT_DIR}/MindCube_tinybench_${task_name}.jsonl"
    local output_subdir="${OUTPUT_DIR}/${task_name}"
    local log_file="${LOG_DIR}/inference_${task_name}_checkpoint-${checkpoint_id}_gpu${gpu_id}.log"
    local pid_file="${LOG_DIR}/pid_${task_name}_checkpoint-${checkpoint_id}_gpu${gpu_id}.txt"
    
    # Create task-specific output directory
    mkdir -p "${output_subdir}"
    
    echo "🔧 [GPU ${gpu_id}] Starting inference for ${task_name} checkpoint-${checkpoint_id}"
    echo "📝 [GPU ${gpu_id}] Input file: ${input_file}"
    echo "📋 [GPU ${gpu_id}] Checkpoint: ${checkpoint_path}"
    echo "📤 [GPU ${gpu_id}] Output directory: ${output_subdir}"
    echo "📋 [GPU ${gpu_id}] Log file: ${log_file}"
    
    # Check if input file exists
    if [ ! -f "${input_file}" ]; then
        echo "❌ [GPU ${gpu_id}] Error: Input file not found: ${input_file}"
        return 1
    fi
    
    # Check if checkpoint exists
    if [ ! -d "${checkpoint_path}" ]; then
        echo "❌ [GPU ${gpu_id}] Error: Checkpoint not found: ${checkpoint_path}"
        return 1
    fi
    
    # Run inference with nohup in background
    nohup env CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/run_inference.py \
        --model-type "${MODEL_TYPE}" \
        --model-path "${checkpoint_path}" \
        --input-file "${input_file}" \
        --output-dir "${output_subdir}" \
        > "${log_file}" 2>&1 &
    
    local pid=$!
    echo "✅ [GPU ${gpu_id}] Job ${job_id} started with PID: ${pid}"
    
    # Save PID and update tracking
    echo "${pid}" > "${pid_file}"
    job_pids["${job_id}"]=${pid}
    job_gpu_mapping["${job_id}"]=${gpu_id}
    running_jobs+=("${job_id}")
    ((gpu_task_count[${gpu_id}]++))
    
    return 0
}

# Function to check task completion and cleanup
check_and_cleanup_completed_jobs() {
    local updated_running_jobs=()
    
    for job_id in "${running_jobs[@]}"; do
        local pid=${job_pids["${job_id}"]}
        local gpu_id=${job_gpu_mapping["${job_id}"]}
        
        # Extract task info for PID file cleanup
        IFS=':' read -r task_part task_name checkpoint_path checkpoint_id <<< "${job_id}:dummy:dummy:dummy"
        local pid_file="${LOG_DIR}/pid_${task_name}_checkpoint-${checkpoint_id}_gpu${gpu_id}.txt"
        
        # Check if process is still running
        if ! kill -0 ${pid} 2>/dev/null; then
            echo "✅ [GPU ${gpu_id}] Job ${job_id} completed (PID: ${pid})"
            
            # Cleanup PID file
            if [ -f "${pid_file}" ]; then
                rm "${pid_file}"
                echo "🗑️  [GPU ${gpu_id}] Cleaned up PID file: ${pid_file}"
            fi
            
            # Update counters
            ((gpu_task_count[${gpu_id}]--))
            unset job_pids["${job_id}"]
            unset job_gpu_mapping["${job_id}"]
        else
            # Job still running, keep it in the list
            updated_running_jobs+=("${job_id}")
        fi
    done
    
    running_jobs=("${updated_running_jobs[@]}")
}

# Function to display current status
show_status() {
    echo ""
    echo "📊 Current Status:"
    echo "  - Total jobs: ${total_jobs}"
    echo "  - Jobs in queue: ${#job_queue[@]}"
    echo "  - Jobs running: ${#running_jobs[@]}"
    echo "  - Jobs completed: $((${total_jobs} - ${#job_queue[@]} - ${#running_jobs[@]}))"
    
    echo "  - GPU utilization:"
    for ((i=0; i<GPU_COUNT; i++)); do
        echo "    GPU ${i}: ${gpu_task_count[$i]}/${MAX_TASKS_PER_GPU} tasks"
    done
    
    echo "  - Progress by task:"
    for task in "${TASKS[@]}"; do
        local completed=$(ls "${OUTPUT_DIR}/${task}"/*_responses.jsonl 2>/dev/null | wc -l)
        local total_checkpoints=$(find "${CHECKPOINT_BASE_DIR}/${task}" -name "checkpoint-*" -type d 2>/dev/null | wc -l)
        if [ ${total_checkpoints} -gt 0 ]; then
            echo "    ${task}: ${completed}/${total_checkpoints} checkpoints completed"
        fi
    done
    echo ""
}

# Main execution loop
echo "🎬 Starting intelligent checkpoint inference execution..."

while [ ${#job_queue[@]} -gt 0 ] || [ ${#running_jobs[@]} -gt 0 ]; do
    # Check and cleanup completed jobs
    check_and_cleanup_completed_jobs
    
    # Try to start new jobs from queue
    updated_queue=()
    for job_spec in "${job_queue[@]}"; do
        IFS=':' read -r job_id task_name checkpoint_path checkpoint_id <<< "${job_spec}"
        gpu_id=$(get_next_gpu)
        
        if [ "${gpu_id}" -ge 0 ] 2>/dev/null; then
            # Found available GPU, start the job
            run_checkpoint_inference "${job_id}" "${task_name}" "${checkpoint_path}" "${checkpoint_id}" "${gpu_id}"
            sleep 2  # Small delay to avoid resource conflicts
        else
            # No available GPU, keep job in queue
            updated_queue+=("${job_spec}")
        fi
    done
    job_queue=("${updated_queue[@]}")
    
    # Show current status
    show_status
    
    # Wait before next check if there are still jobs running or in queue
    if [ ${#running_jobs[@]} -gt 0 ] || [ ${#job_queue[@]} -gt 0 ]; then
        echo "⏳ Waiting ${MONITOR_INTERVAL}s before next status check..."
        sleep ${MONITOR_INTERVAL}
    fi
done

echo ""
echo "🎉 All SFT checkpoint inference jobs completed successfully!"
echo "📊 Final Summary:"
echo "  - Total jobs completed: ${total_jobs}"
echo "  - GPUs used: ${GPU_COUNT}"
echo "  - Max tasks per GPU: ${MAX_TASKS_PER_GPU}"
echo "  - Model: ${MODEL_TYPE}"
echo "  - Tasks processed: ${#TASKS[@]}"
echo "  - Input directory: ${INPUT_DIR}"
echo "  - Output directory: ${OUTPUT_DIR}"
echo "  - Log directory: ${LOG_DIR}"

echo ""
echo "📋 Monitoring commands for future reference:"
echo "  # Check running processes:"
echo "  ps aux | grep run_inference.py"
echo ""
echo "  # Check logs for specific task and checkpoint:"
echo "  tail -f ${LOG_DIR}/inference_<task_name>_checkpoint-<checkpoint_id>_gpu<gpu_id>.log"
echo ""
echo "  # Count completed results by task:"
echo "  for task in ${TASKS[@]}; do"
echo "    echo \"Task \$task: \$(ls ${OUTPUT_DIR}/\$task/*_responses.jsonl 2>/dev/null | wc -l) completed\""
echo "  done"
echo ""
echo "  # List all completed output files:"
echo "  find ${OUTPUT_DIR} -name '*_responses.jsonl' | sort"

echo ""
echo "⏰ Script started at: $(date)"
echo "⏰ Script completed at: $(date)"
echo "✅ All jobs finished and PID files cleaned up automatically.z