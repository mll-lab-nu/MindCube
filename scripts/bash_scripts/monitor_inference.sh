#!/bin/bash

# Script to monitor the progress of inference tasks
# Author: MindCube Team
# Usage: bash scripts/bash_scripts/monitor_inference.sh

LOG_DIR="./logs/inference"
OUTPUT_DIR="./data/results/frozen_vlm"

TASKS=(
    "raw_qa"
    "aug_cgmap_in" 
    "ff_rsn"
    "aug_cgmap_ffr_out"
    "plain_cgmap_ffr_out"
    "cgmap_in_ffr_out"
)

echo "📊 Inference Task Monitor"
echo "========================="
echo "⏰ Current time: $(date)"
echo ""

# Check running processes
echo "🔄 Running inference processes:"
running_processes=$(ps aux | grep "[r]un_inference.py" | wc -l)
if [ "$running_processes" -gt 0 ]; then
    ps aux | grep "[r]un_inference.py" | awk '{print "  PID: " $2 " | GPU: " $13 " | Status: Running"}'
    echo "  Total running: $running_processes"
else
    echo "  No inference processes currently running"
fi
echo ""

# Check completed tasks
echo "✅ Completed output files:"
if [ -d "$OUTPUT_DIR" ]; then
    completed_files=$(ls "$OUTPUT_DIR"/*_responses.jsonl 2>/dev/null | wc -l)
    if [ "$completed_files" -gt 0 ]; then
        ls -la "$OUTPUT_DIR"/*_responses.jsonl 2>/dev/null | awk '{print "  " $9 " | Size: " $5 " bytes | Date: " $6 " " $7 " " $8}'
        echo "  Total completed: $completed_files"
    else
        echo "  No completed output files found"
    fi
else
    echo "  Output directory not found: $OUTPUT_DIR"
fi
echo ""

# Check task status
echo "📋 Task Status Summary:"
echo "Task                    | GPU | Status      | Log File"
echo "------------------------|-----|-------------|------------------"

for i in "${!TASKS[@]}"; do
    task_name="${TASKS[i]}"
    gpu_id=$((i % 4))
    output_file="$OUTPUT_DIR/MindCube_tinybench_${task_name}_qwen2.5-vl-3b-instruct_responses.jsonl"
    log_file="$LOG_DIR/inference_${task_name}_gpu${gpu_id}.log"
    pid_file="$LOG_DIR/pid_${task_name}_gpu${gpu_id}.txt"
    
    # Check status
    if [ -f "$output_file" ]; then
        status="✅ Completed"
    elif [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            status="🔄 Running"
        else
            status="❌ Failed/Stopped"
        fi
    else
        status="⏳ Not started"
    fi
    
    printf "%-23s | %-3s | %-11s | %s\n" "$task_name" "$gpu_id" "$status" "$(basename "$log_file")"
done

echo ""

# Show recent log activity
echo "📝 Recent log activity (last 3 lines from each active log):"
for i in "${!TASKS[@]}"; do
    task_name="${TASKS[i]}"
    gpu_id=$((i % 4))
    log_file="$LOG_DIR/inference_${task_name}_gpu${gpu_id}.log"
    
    if [ -f "$log_file" ]; then
        echo ""
        echo "--- $task_name (GPU $gpu_id) ---"
        tail -n 3 "$log_file" 2>/dev/null || echo "  Log file empty or unreadable"
    fi
done

echo ""
echo "🔧 Useful commands:"
echo "  # Watch this monitor in real-time:"
echo "  watch -n 10 bash scripts/bash_scripts/monitor_inference.sh"
echo ""
echo "  # Follow specific task log:"
echo "  tail -f $LOG_DIR/inference_<task_name>_gpu<gpu_id>.log"
echo ""
echo "  # Kill all inference processes (if needed):"
echo "  pkill -f run_inference.py" 