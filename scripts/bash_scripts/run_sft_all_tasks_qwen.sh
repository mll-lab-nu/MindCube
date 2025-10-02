#!/bin/bash

# Script to run SFT training for all specified tasks on Qwen2.5-VL model
# Uses 4 GPUs (0-3) for each task, runs tasks sequentially
# Author: MindCube Team
# Usage: bash scripts/bash_scripts/run_sft_all_tasks_qwen.sh

echo "🚀 Starting SFT training for all tasks on Qwen2.5-VL..."
echo "📅 Start time: $(date)"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SFT_DIR="${PROJECT_ROOT}/checkpoints/sft"
LOG_DIR="${PROJECT_ROOT}/logs/sft_training"
TRAINING_SCRIPT="${SFT_DIR}/train_qwen_sft.sh"

# Task list - EASILY CONFIGURABLE
# Add or remove tasks here as needed
TASKS=(
    "raw_qa"
    "aug_cgmap_out"
    "plain_cgmap_out"
    "ff_rsn"
    "aug_cgmap_ffr_out"
    "plain_cgmap_ffr_out"
)

# Optional: Add cog_reasoning if needed
# Uncomment the line below to include cognitive reasoning task
# TASKS+=("cog_reasoning")

# Create necessary directories
mkdir -p "${LOG_DIR}"

echo "📁 Project root: ${PROJECT_ROOT}"
echo "📁 SFT directory: ${SFT_DIR}"
echo "📁 Log directory: ${LOG_DIR}"
echo "🎯 Training script: ${TRAINING_SCRIPT}"
echo "📋 Tasks to run: ${#TASKS[@]} tasks"

# Display task list
echo ""
echo "📋 Task execution order:"
for i in "${!TASKS[@]}"; do
    echo "  $((i+1)). ${TASKS[i]}"
done

echo ""

# Verify training script exists
if [ ! -f "${TRAINING_SCRIPT}" ]; then
    echo "❌ Error: Training script not found: ${TRAINING_SCRIPT}"
    echo "Please ensure you're running this script from the correct location."
    exit 1
fi

# Verify all config files exist
echo "🔍 Verifying configuration files..."
for task in "${TASKS[@]}"; do
    config_file="${SFT_DIR}/config_${task}.sh"
    if [ ! -f "${config_file}" ]; then
        echo "❌ Error: Configuration file not found: ${config_file}"
        exit 1
    else
        echo "  ✅ ${config_file}"
    fi
done

echo ""

# Function to run SFT training for a single task
run_task_training() {
    local task_name=$1
    local task_number=$2
    local total_tasks=$3
    local config_file="config_${task_name}.sh"
    local log_file="${LOG_DIR}/sft_training_${task_name}.log"
    local pid_file="${LOG_DIR}/sft_training_${task_name}.pid"
    local start_time=$(date)
    
    echo "🎯 [$task_number/$total_tasks] Starting SFT training for task: ${task_name}"
    echo "📝 Config file: ${config_file}"
    echo "📋 Log file: ${log_file}"
    echo "⏰ Start time: ${start_time}"
    echo ""
    
    # Stay in project root directory (following new convention)
    cd "${PROJECT_ROOT}" || {
        echo "❌ Error: Cannot change to project root directory: ${PROJECT_ROOT}"
        return 1
    }
    
    # Use full path for config file to work from project root
    local full_config_path="${SFT_DIR}/${config_file}"
    
    # Run training and capture both stdout and stderr
    echo "🔧 Executing: ${TRAINING_SCRIPT} ${full_config_path}"
    echo "📊 This will block until training completes..."
    
    # Start the training process and get its PID
    "${TRAINING_SCRIPT}" "${full_config_path}" > "${log_file}" 2>&1 &
    local train_pid=$!
    echo ${train_pid} > "${pid_file}"
    
    echo "🔍 Training process started with PID: ${train_pid}"
    echo "⏳ Waiting for training to complete..."
    
    # Wait for the training process to complete
    wait ${train_pid}
    local exit_code=$?
    
    # Clean up PID file
    rm -f "${pid_file}"
    
    local end_time=$(date)
    
    if [ ${exit_code} -eq 0 ]; then
        echo "✅ [$task_number/$total_tasks] Task ${task_name} completed successfully!"
        echo "   Start time: ${start_time}"
        echo "   End time: ${end_time}"
        echo "   Log file: ${log_file}"
        echo ""
        return 0
    else
        echo "❌ [$task_number/$total_tasks] Task ${task_name} failed with exit code: ${exit_code}"
        echo "   Start time: ${start_time}"
        echo "   End time: ${end_time}"
        echo "   Log file: ${log_file}"
        echo "   Check the log file for details."
        echo ""
        return 1
    fi
}

# Start training all tasks sequentially
echo "🎬 Starting sequential SFT training for all tasks..."
echo ""

# Check for existing training processes
echo "🔍 Checking for existing training processes..."
existing_processes=$(ps aux | grep -E "(train_qwen|torchrun)" | grep -v grep | wc -l)
if [ ${existing_processes} -gt 0 ]; then
    echo "⚠️  Warning: Found ${existing_processes} existing training process(es):"
    ps aux | grep -E "(train_qwen|torchrun)" | grep -v grep
    echo ""
    echo "❌ Please stop existing training processes before starting new ones to avoid GPU conflicts."
    echo "   Use: pkill -f train_qwen  or  pkill -f torchrun"
    exit 1
else
    echo "✅ No existing training processes found."
fi
echo ""

# Check GPU availability
echo "🎮 Checking GPU status..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
        echo "  GPU: $line"
    done
else
    echo "  ⚠️  nvidia-smi not available, cannot check GPU status"
fi
echo ""

SUCCESSFUL_TASKS=0
FAILED_TASKS=0
OVERALL_START_TIME=$(date)

for i in "${!TASKS[@]}"; do
    task_name="${TASKS[i]}"
    task_number=$((i+1))
    total_tasks=${#TASKS[@]}
    
    if run_task_training "${task_name}" "${task_number}" "${total_tasks}"; then
        ((SUCCESSFUL_TASKS++))
    else
        ((FAILED_TASKS++))
        echo "⚠️  Task ${task_name} failed, but continuing with remaining tasks..."
        echo ""
    fi
    
    # Add a small delay between tasks to ensure clean separation
    if [ $task_number -lt $total_tasks ]; then
        echo "⏳ Waiting 10 seconds before starting next task..."
        sleep 10
        echo ""
    fi
done

OVERALL_END_TIME=$(date)

echo "🎉 All SFT training tasks completed!"
echo ""
echo "📊 Final Summary:"
echo "  - Total tasks: ${#TASKS[@]}"
echo "  - Successful: ${SUCCESSFUL_TASKS}"
echo "  - Failed: ${FAILED_TASKS}"
echo "  - Overall start time: ${OVERALL_START_TIME}"
echo "  - Overall end time: ${OVERALL_END_TIME}"
echo ""

echo "📋 Task Results:"
for i in "${!TASKS[@]}"; do
    task_name="${TASKS[i]}"
    log_file="${LOG_DIR}/sft_training_${task_name}.log"
    if [ -f "${log_file}" ]; then
        echo "  ${task_name}: ${log_file}"
    else
        echo "  ${task_name}: No log file found"
    fi
done

echo ""
echo "📈 To check training results:"
echo "  # Check checkpoints:"
echo "  ls -la ${PROJECT_ROOT}/experiments/sft/results/"
echo ""
echo "  # Check specific task log:"
echo "  tail -f ${LOG_DIR}/sft_training_<task_name>.log"
echo ""
echo "  # Monitor GPU usage during training:"
echo "  watch -n 1 nvidia-smi"

echo ""
if [ $FAILED_TASKS -eq 0 ]; then
    echo "✅ All tasks completed successfully!"
    exit 0
else
    echo "⚠️  ${FAILED_TASKS} task(s) failed. Check the logs for details."
    exit 1
fi 
