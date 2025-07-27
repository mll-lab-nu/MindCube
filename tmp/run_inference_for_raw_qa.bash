#!/bin/bash

# Script to run inference for all hyperparameter ablation checkpoints (raw_qa task only)
# Dynamically detects available GPUs and manages task queues
# Each GPU runs maximum N tasks concurrently using round robin algorithm
# Author: MindCube Team
# Usage: bash tmp/run_inference_for_raw_qa.bash [--max-tasks-per-gpu N]

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --max-tasks-per-gpu N    Maximum number of tasks per GPU (default: 2)"
    echo "  --help, -h               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                            # Use default (2 tasks per GPU)"
    echo "  $0 --max-tasks-per-gpu 1      # Allow 1 task per GPU"
    echo "  $0 --max-tasks-per-gpu 3      # Allow 3 tasks per GPU"
    echo ""
    echo "Note: This script processes all checkpoints from hyperparameter ablation experiments."
}

# Parse command line arguments
MAX_TASKS_PER_GPU=2  # Default value for hyperparameter ablation experiments
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

echo "🚀 Starting hyperparameter ablation checkpoint inference for raw_qa task..."
echo "📅 Start time: $(date)"

# Configuration
MODEL_TYPE="qwen2.5vl"
INPUT_FILE="./data/prompts/general/MindCube_tinybench_raw_qa.jsonl"
OUTPUT_BASE_DIR="/projects/b1222/userdata/qineng/01_projects/07_MindCube_new/tmp_results/hp_ablation"
LOG_DIR="./logs/hp_ablation_inference"
EXPERIMENTS_BASE_DIR="/projects/b1222/userdata/qineng/01_projects/07_MindCube_new/experiments/sft/tmp"
MONITOR_INTERVAL=30  # seconds between status checks

# Force GPU count to 4 as specified
GPU_COUNT=4
echo "🎮 Using ${GPU_COUNT} GPUs as specified"

# Create necessary directories
mkdir -p "${OUTPUT_BASE_DIR}"
mkdir -p "${LOG_DIR}"

echo "📁 Input file: ${INPUT_FILE}"
echo "📁 Output base directory: ${OUTPUT_BASE_DIR}"
echo "📁 Log directory: ${LOG_DIR}"
echo "📁 Experiments base directory: ${EXPERIMENTS_BASE_DIR}"
echo "🎯 Model type: ${MODEL_TYPE}"
echo "⚙️  Max tasks per GPU: ${MAX_TASKS_PER_GPU}"
echo "💾 Max concurrent tasks: $((GPU_COUNT * MAX_TASKS_PER_GPU))"

# Check if input file exists
if [ ! -f "${INPUT_FILE}" ]; then
    echo "❌ Error: Input file not found: ${INPUT_FILE}"
    exit 1
fi

# Check if experiments directory exists
if [ ! -d "${EXPERIMENTS_BASE_DIR}" ]; then
    echo "❌ Error: Experiments directory not found: ${EXPERIMENTS_BASE_DIR}"
    exit 1
fi

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

# Build job queue (all setting-checkpoint combinations)
echo ""
echo "🔍 Discovering hyperparameter settings and checkpoints..."
total_jobs=0

# Find all setting directories
settings=($(find "${EXPERIMENTS_BASE_DIR}" -maxdepth 1 -type d -not -path "${EXPERIMENTS_BASE_DIR}"))

if [ ${#settings[@]} -eq 0 ]; then
    echo "❌ Error: No setting directories found in ${EXPERIMENTS_BASE_DIR}"
    exit 1
fi

echo "📋 Found ${#settings[@]} hyperparameter settings"

for setting_path in "${settings[@]}"; do
    setting_name=$(basename "${setting_path}")
    echo "📋 Processing setting: ${setting_name}"
    
    # Look for raw_qa directory
    raw_qa_dir="${setting_path}/raw_qa"
    if [ ! -d "${raw_qa_dir}" ]; then
        echo "⚠️  Warning: No raw_qa directory found for setting ${setting_name}"
        continue
    fi
    
    # Find and sort checkpoints
    checkpoints=($(find "${raw_qa_dir}" -name "checkpoint-*" -type d | sort -t'-' -k2 -n))
    
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo "⚠️  Warning: No checkpoints found for setting ${setting_name}"
        continue
    fi
    
    echo "📁 Found ${#checkpoints[@]} checkpoints for setting ${setting_name}"
    
    for checkpoint_path in "${checkpoints[@]}"; do
        checkpoint_id=$(basename "${checkpoint_path}" | sed 's/checkpoint-//')
        job_id="${setting_name}_checkpoint-${checkpoint_id}"
        job_queue+=("${job_id}:${setting_name}:${checkpoint_path}:${checkpoint_id}")
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
    local setting_name=$2
    local checkpoint_path=$3
    local checkpoint_id=$4
    local gpu_id=$5
    
    local output_subdir="${OUTPUT_BASE_DIR}/${setting_name}"
    local log_file="${LOG_DIR}/inference_${setting_name}_checkpoint-${checkpoint_id}_gpu${gpu_id}.log"
    local pid_file="${LOG_DIR}/pid_${setting_name}_checkpoint-${checkpoint_id}_gpu${gpu_id}.txt"
    
    # Create setting-specific output directory
    mkdir -p "${output_subdir}"
    
    echo "🔧 [GPU ${gpu_id}] Starting inference for ${setting_name} checkpoint-${checkpoint_id}"
    echo "📝 [GPU ${gpu_id}] Input file: ${INPUT_FILE}"
    echo "📋 [GPU ${gpu_id}] Checkpoint: ${checkpoint_path}"
    echo "📤 [GPU ${gpu_id}] Output directory: ${output_subdir}"
    echo "📋 [GPU ${gpu_id}] Log file: ${log_file}"
    
    # Check if checkpoint exists
    if [ ! -d "${checkpoint_path}" ]; then
        echo "❌ [GPU ${gpu_id}] Error: Checkpoint not found: ${checkpoint_path}"
        return 1
    fi
    
    # Run inference with nohup in background
    nohup env CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/run_inference.py \
        --model-type "${MODEL_TYPE}" \
        --model-path "${checkpoint_path}" \
        --input-file "${INPUT_FILE}" \
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
        
        # Extract setting info for PID file cleanup
        IFS=':' read -r setting_part setting_name checkpoint_path checkpoint_id <<< "${job_id}:dummy:dummy:dummy"
        local pid_file="${LOG_DIR}/pid_${setting_name}_checkpoint-${checkpoint_id}_gpu${gpu_id}.txt"
        
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
    
    echo "  - Progress by setting:"
    for setting_path in "${settings[@]}"; do
        if [ -d "${setting_path}" ]; then
            local setting_name=$(basename "${setting_path}")
            local completed=$(ls "${OUTPUT_BASE_DIR}/${setting_name}"/*_responses.jsonl 2>/dev/null | wc -l)
            local total_checkpoints=$(find "${setting_path}/raw_qa" -name "checkpoint-*" -type d 2>/dev/null | wc -l)
            if [ ${total_checkpoints} -gt 0 ]; then
                echo "    ${setting_name}: ${completed}/${total_checkpoints} checkpoints completed"
            fi
        fi
    done
    echo ""
}

# Main execution loop
echo "🎬 Starting intelligent hyperparameter ablation inference execution..."

while [ ${#job_queue[@]} -gt 0 ] || [ ${#running_jobs[@]} -gt 0 ]; do
    # Check and cleanup completed jobs
    check_and_cleanup_completed_jobs
    
    # Try to start new jobs from queue
    updated_queue=()
    for job_spec in "${job_queue[@]}"; do
        IFS=':' read -r job_id setting_name checkpoint_path checkpoint_id <<< "${job_spec}"
        gpu_id=$(get_next_gpu)
        
        if [ "${gpu_id}" -ge 0 ] 2>/dev/null; then
            # Found available GPU, start the job
            run_checkpoint_inference "${job_id}" "${setting_name}" "${checkpoint_path}" "${checkpoint_id}" "${gpu_id}"
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
echo "🎉 All hyperparameter ablation inference jobs completed successfully!"
echo "📊 Final Summary:"
echo "  - Total jobs completed: ${total_jobs}"
echo "  - GPUs used: ${GPU_COUNT}"
echo "  - Max tasks per GPU: ${MAX_TASKS_PER_GPU}"
echo "  - Model: ${MODEL_TYPE}"
echo "  - Task: raw_qa"
echo "  - Input file: ${INPUT_FILE}"
echo "  - Output directory: ${OUTPUT_BASE_DIR}"
echo "  - Log directory: ${LOG_DIR}"

echo ""
echo "📋 Monitoring commands for future reference:"
echo "  # Check running processes:"
echo "  ps aux | grep run_inference.py"
echo ""
echo "  # Check logs for specific setting and checkpoint:"
echo "  tail -f ${LOG_DIR}/inference_<setting_name>_checkpoint-<checkpoint_id>_gpu<gpu_id>.log"
echo ""
echo "  # Count completed results by setting:"
echo "  for setting in \$(ls ${OUTPUT_BASE_DIR}); do"
echo "    echo \"Setting \$setting: \$(ls ${OUTPUT_BASE_DIR}/\$setting/*_responses.jsonl 2>/dev/null | wc -l) completed\""
echo "  done"
echo ""
echo "  # List all completed output files:"
echo "  find ${OUTPUT_BASE_DIR} -name '*_responses.jsonl' | sort"

echo ""
echo "⏰ Script started at: $(date)"
echo "⏰ Script completed at: $(date)"
echo "✅ All jobs finished and PID files cleaned up automatically."
