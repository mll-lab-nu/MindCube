#!/bin/bash

# Script to run inference for SFT trained checkpoints on Qwen2.5-VL model with multiple seeds
# For rebuttal: supports multiple seeds to compute p-values and error bars
# Dynamically detects available GPUs and manages task queues
# Each GPU runs maximum N tasks concurrently using round robin algorithm
# Author: MindCube Team
# Usage: bash tmp/run_sft_inference_all_seeds.bash [--max-tasks-per-gpu N]

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
}

# Parse command line arguments
MAX_TASKS_PER_GPU=1  # Default value
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

echo "🚀 Starting SFT checkpoint inference for all settings on Qwen2.5-VL with multiple seeds..."
echo "📅 Start time: $(date)"

# Configuration
MODEL_TYPE="qwen2.5vl"
INPUT_DIR="./data/prompts/general"
CHECKPOINT_DIR="/projects/b1222/userdata/qineng/01_projects/07_MindCube_new/checkpoints/sft"
OUTPUT_DIR="/projects/b1222/userdata/qineng/01_projects/07_MindCube_new/tmp_results/sft_multi_seeds"
LOG_DIR="./logs/sft_inference_multi_seeds"
MONITOR_INTERVAL=30  # seconds between status checks

# SFT settings list (excluding cgmap_in_ffr_out_llm_only as requested)
SFT_SETTINGS=(
    "raw_qa"
    "ff_rsn" 
    "aug_cgmap_ffr_out"
    "plain_cgmap_ffr_out"
    "cgmap_in_ffr_out_all_parts"
    "plain_cgmap_out"
    "aug_cgmap_out"
)

# Mapping from SFT setting names to their corresponding input file names
declare -A SETTING_TO_INPUT_MAP
SETTING_TO_INPUT_MAP["raw_qa"]="raw_qa"
SETTING_TO_INPUT_MAP["ff_rsn"]="ff_rsn"
SETTING_TO_INPUT_MAP["aug_cgmap_ffr_out"]="aug_cgmap_ffr_out"
SETTING_TO_INPUT_MAP["plain_cgmap_ffr_out"]="plain_cgmap_ffr_out"
SETTING_TO_INPUT_MAP["cgmap_in_ffr_out_all_parts"]="cgmap_in_ffr_out"
SETTING_TO_INPUT_MAP["plain_cgmap_out"]="plain_cgmap_out"
SETTING_TO_INPUT_MAP["aug_cgmap_out"]="aug_cgmap_out"

# Seeds for multi-seed experiments (3 seeds as requested)
SEEDS=(42 123 456)

echo "🎯 Running with ${#SEEDS[@]} seeds: ${SEEDS[*]}"

# Create necessary directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Verify checkpoint directories exist
echo "🔍 Verifying SFT checkpoint directories..."
for setting in "${SFT_SETTINGS[@]}"; do
    ckpt_dir="${CHECKPOINT_DIR}/${setting}/checkpoint-57"
    if [ ! -d "${ckpt_dir}" ]; then
        echo "❌ Error: Checkpoint directory not found: ${ckpt_dir}"
        exit 1
    else
        echo "  ✅ ${setting}/checkpoint-57"
    fi
done

# Detect available GPUs
echo "🔍 Detecting available GPUs..."
if command -v nvidia-smi &> /dev/null; then
    # Get GPU count using nvidia-smi
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
echo "📁 Input directory: ${INPUT_DIR}"
echo "📁 Checkpoint directory: ${CHECKPOINT_DIR}"
echo "📁 Output directory: ${OUTPUT_DIR}"
echo "📁 Log directory: ${LOG_DIR}"
echo "🎯 Model type: ${MODEL_TYPE}"
echo "📋 SFT settings to run: ${#SFT_SETTINGS[@]} settings"
echo "🎲 Seeds to run: ${#SEEDS[@]} seeds per setting"
echo "📊 Total experiments: $((${#SFT_SETTINGS[@]} * ${#SEEDS[@]}))"
echo "⚙️  Max tasks per GPU: ${MAX_TASKS_PER_GPU}"

# Initialize GPU task counters
declare -A gpu_task_count
for ((i=0; i<GPU_COUNT; i++)); do
    gpu_task_count[$i]=0
done

# Create task queue with all setting-seed combinations
declare -a task_queue=()
for setting in "${SFT_SETTINGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        task_queue+=("${setting}_seed${seed}")
    done
done

declare -a running_tasks=()
declare -A task_gpu_mapping
declare -A task_pids

echo ""
echo "🎬 Starting task execution with intelligent GPU allocation..."

# Function to get next available GPU using round robin
get_next_gpu() {
    local min_tasks=999
    local selected_gpu=-1
    
    # Find GPU with minimum running tasks (round robin effect)
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

# Function to run inference for a single setting-seed combination
run_task_inference() {
    local setting_seed_combo=$1
    local gpu_id=$2
    
    # Parse setting name and seed from combination
    local setting_name="${setting_seed_combo%_seed*}"
    local seed="${setting_seed_combo##*_seed}"
    
    # Get corresponding input file name
    local input_task_name="${SETTING_TO_INPUT_MAP[${setting_name}]}"
    if [ -z "${input_task_name}" ]; then
        echo "❌ [GPU ${gpu_id}] Error: No input mapping found for setting: ${setting_name}"
        return 1
    fi
    
    local input_file="${INPUT_DIR}/MindCube_tinybench_${input_task_name}.jsonl"
    local checkpoint_path="${CHECKPOINT_DIR}/${setting_name}/checkpoint-57"
    local log_file="${LOG_DIR}/inference_${setting_name}_seed${seed}_gpu${gpu_id}.log"
    local pid_file="${LOG_DIR}/pid_${setting_name}_seed${seed}_gpu${gpu_id}.txt"
    
    echo "🔧 [GPU ${gpu_id}] Starting inference for setting: ${setting_name} with seed: ${seed}"
    echo "📝 [GPU ${gpu_id}] Input file: ${input_file}"
    echo "🏗️  [GPU ${gpu_id}] Checkpoint: ${checkpoint_path}"
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
    
    # Run inference with nohup in background, using checkpoint path as model-path
    nohup env CUDA_VISIBLE_DEVICES=${gpu_id} python scripts/run_inference.py \
        --model-type "${MODEL_TYPE}" \
        --model-path "${checkpoint_path}" \
        --input-file "${input_file}" \
        --output-dir "${OUTPUT_DIR}" \
        --seed "${seed}" \
        --max-new-tokens 1024 \
        --temperature 0.7 \
        > "${log_file}" 2>&1 &
    
    local pid=$!
    echo "✅ [GPU ${gpu_id}] Setting ${setting_name} (seed ${seed}) started with PID: ${pid}"
    
    # Save PID and update tracking
    echo "${pid}" > "${pid_file}"
    task_pids["${setting_seed_combo}"]=${pid}
    task_gpu_mapping["${setting_seed_combo}"]=${gpu_id}
    running_tasks+=("${setting_seed_combo}")
    ((gpu_task_count[${gpu_id}]++))
    
    return 0
}

# Function to check task completion and cleanup
check_and_cleanup_completed_tasks() {
    local updated_running_tasks=()
    
    for setting_seed_combo in "${running_tasks[@]}"; do
        local pid=${task_pids["${setting_seed_combo}"]}
        local gpu_id=${task_gpu_mapping["${setting_seed_combo}"]}
        
        # Parse setting name and seed from combination
        local setting_name="${setting_seed_combo%_seed*}"
        local seed="${setting_seed_combo##*_seed}"
        local pid_file="${LOG_DIR}/pid_${setting_name}_seed${seed}_gpu${gpu_id}.txt"
        
        # Check if process is still running
        if ! kill -0 ${pid} 2>/dev/null; then
            echo "✅ [GPU ${gpu_id}] Setting ${setting_name} (seed ${seed}) completed (PID: ${pid})"
            
            # Cleanup PID file
            if [ -f "${pid_file}" ]; then
                rm "${pid_file}"
                echo "🗑️  [GPU ${gpu_id}] Cleaned up PID file: ${pid_file}"
            fi
            
            # Update counters
            ((gpu_task_count[${gpu_id}]--))
            unset task_pids["${setting_seed_combo}"]
            unset task_gpu_mapping["${setting_seed_combo}"]
        else
            # Task still running, keep it in the list
            updated_running_tasks+=("${setting_seed_combo}")
        fi
    done
    
    running_tasks=("${updated_running_tasks[@]}")
}

# Function to display current status
show_status() {
    echo ""
    echo "📊 Current Status:"
    echo "  - Total setting-seed combinations: $((${#SFT_SETTINGS[@]} * ${#SEEDS[@]}))"
    echo "  - Combinations in queue: ${#task_queue[@]}"
    echo "  - Combinations running: ${#running_tasks[@]}"
    echo "  - Combinations completed: $(($((${#SFT_SETTINGS[@]} * ${#SEEDS[@]})) - ${#task_queue[@]} - ${#running_tasks[@]}))"
    
    echo "  - GPU utilization:"
    for ((i=0; i<GPU_COUNT; i++)); do
        echo "    GPU ${i}: ${gpu_task_count[$i]}/${MAX_TASKS_PER_GPU} tasks"
    done
    echo ""
}

# Main execution loop
while [ ${#task_queue[@]} -gt 0 ] || [ ${#running_tasks[@]} -gt 0 ]; do
    # Check and cleanup completed tasks
    check_and_cleanup_completed_tasks
    
    # Try to start new tasks from queue
    updated_queue=()
    for setting_seed_combo in "${task_queue[@]}"; do
        gpu_id=$(get_next_gpu)
        
        if [ "${gpu_id}" -ge 0 ] 2>/dev/null; then
            # Found available GPU, start the task
            run_task_inference "${setting_seed_combo}" "${gpu_id}"
            sleep 2  # Small delay to avoid resource conflicts
        else
            # No available GPU, keep task in queue
            updated_queue+=("${setting_seed_combo}")
        fi
    done
    task_queue=("${updated_queue[@]}")
    
    # Show current status
    show_status
    
    # Wait before next check if there are still tasks running or in queue
    if [ ${#running_tasks[@]} -gt 0 ] || [ ${#task_queue[@]} -gt 0 ]; then
        echo "⏳ Waiting ${MONITOR_INTERVAL}s before next status check..."
        sleep ${MONITOR_INTERVAL}
    fi
done

echo ""
echo "🎉 All SFT checkpoint inference completed successfully!"
echo "📊 Final Summary:"
echo "  - Total experiments completed: $((${#SFT_SETTINGS[@]} * ${#SEEDS[@]}))"
echo "  - SFT settings: ${#SFT_SETTINGS[@]} (${SFT_SETTINGS[*]})"
echo "  - Seeds per setting: ${#SEEDS[@]} (${SEEDS[*]})"
echo "  - GPUs used: ${GPU_COUNT}"
echo "  - Model: ${MODEL_TYPE}"
echo "  - Max tokens: 1024"
echo "  - Checkpoint directory: ${CHECKPOINT_DIR}"
echo "  - Input directory: ${INPUT_DIR}"
echo "  - Output directory: ${OUTPUT_DIR}"
echo "  - Log directory: ${LOG_DIR}"

echo ""
echo "📋 Monitoring commands for future reference:"
echo "  # Check running processes:"
echo "  ps aux | grep run_inference.py"
echo ""
echo "  # Check logs for specific setting and seed:"
echo "  tail -f ${LOG_DIR}/inference_<setting_name>_seed<seed>_gpu<gpu_id>.log"
echo ""
echo "  # Count completed experiments:"
echo "  ls ${OUTPUT_DIR}/*_seed*_responses.jsonl | wc -l"
echo ""
echo "  # List completed output files:"
echo "  ls -la ${OUTPUT_DIR}/"
echo ""
echo "  # Group results by setting:"
echo "  for setting in ${SFT_SETTINGS[*]}; do echo \"=== \$setting ===\"; ls ${OUTPUT_DIR}/*\${setting}*seed*.jsonl 2>/dev/null || echo \"No results yet\"; done"

echo ""
echo "⏰ Script started at: $(date)"
echo "⏰ Script completed at: $(date)"
echo "✅ All SFT checkpoint experiments finished and PID files cleaned up automatically."
echo ""
echo "🔬 For statistical analysis, you now have ${#SEEDS[@]} independent runs per SFT setting for computing p-values and error bars."
