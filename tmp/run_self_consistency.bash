#!/bin/bash

# Script to apply self-consistency to all multi-seed results
# Usage: bash tmp/run_self_consistency.bash [OPTIONS]

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --input-dir DIR          Input directory with multi-seed results"
    echo "                          (default: tmp_results/frozen_multi_seeds)"
    echo "  --output-dir DIR         Output directory for consensus results"
    echo "                          (default: tmp_results/self_consistency)"
    echo "  --strategy STRATEGY      Voting strategy: majority, exact, similarity"
    echo "                          (default: majority)"
    echo "  --seeds SEEDS            Comma-separated list of seeds"
    echo "                          (default: 42,123,456)"
    echo "  --help, -h               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use all defaults"
    echo "  $0 --strategy exact                  # Use exact match voting"
    echo "  $0 --output-dir results/consensus    # Custom output directory"
}

# Default values
INPUT_DIR="/workspace/MindCube/tmp_results/frozen_multi_seeds"
OUTPUT_DIR="/workspace/MindCube/tmp_results/self_consistency"
STRATEGY="majority"
SEEDS="42,123,456"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            if [[ ! "$STRATEGY" =~ ^(majority|exact|similarity)$ ]]; then
                echo "❌ Error: Strategy must be one of: majority, exact, similarity"
                exit 1
            fi
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
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

echo "🤖 Starting Self-Consistency Analysis..."
echo "📅 Start time: $(date)"
echo ""

# Configuration
TASKS=(
    "raw_qa"
    "aug_cgmap_in" 
    "ff_rsn"
    "aug_cgmap_ffr_out"
    "plain_cgmap_ffr_out"
    "cgmap_in_ffr_out"
)

echo "🔧 Configuration:"
echo "  📁 Input directory: ${INPUT_DIR}"
echo "  📁 Output directory: ${OUTPUT_DIR}"
echo "  🎯 Strategy: ${STRATEGY}"
echo "  🎲 Seeds: ${SEEDS}"
echo "  📋 Tasks: ${#TASKS[@]} tasks (${TASKS[*]})"
echo ""

# Validate input directory
if [ ! -d "${INPUT_DIR}" ]; then
    echo "❌ Error: Input directory does not exist: ${INPUT_DIR}"
    echo "💡 Make sure you have run the multi-seed inference first using:"
    echo "   bash tmp/run_frozen_all_seeds.bash"
    exit 1
fi

# Check if we have any result files
result_files_count=$(find "${INPUT_DIR}" -name "*seed*responses.jsonl" | wc -l)
if [ ${result_files_count} -eq 0 ]; then
    echo "❌ Error: No multi-seed result files found in ${INPUT_DIR}"
    echo "💡 Expected files with pattern: *seed*responses.jsonl"
    exit 1
fi

echo "✅ Found ${result_files_count} multi-seed result files"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Process each task
echo ""
echo "🎬 Starting self-consistency processing..."

successful_tasks=0
failed_tasks=0

for task in "${TASKS[@]}"; do
    echo ""
    echo "🔍 Processing task: ${task}"
    
    # Check if we have result files for this task
    task_files=$(find "${INPUT_DIR}" -name "*${task}*seed*responses.jsonl" | wc -l)
    if [ ${task_files} -eq 0 ]; then
        echo "⚠️  No result files found for task ${task}, skipping..."
        ((failed_tasks++))
        continue
    fi
    
    echo "📁 Found ${task_files} files for task ${task}"
    
    # Run self-consistency
    echo "🤖 Applying ${STRATEGY} voting strategy..."
    
    if python tmp/self_consistency.py \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --task "${task}" \
        --seeds "${SEEDS}" \
        --strategy "${STRATEGY}"; then
        
        echo "✅ Successfully processed task: ${task}"
        ((successful_tasks++))
    else
        echo "❌ Failed to process task: ${task}"
        ((failed_tasks++))
    fi
done

# Final summary
echo ""
echo "🎉 Self-consistency analysis completed!"
echo "📊 Final Summary:"
echo "  - Tasks processed successfully: ${successful_tasks}"
echo "  - Tasks failed: ${failed_tasks}"
echo "  - Total tasks: ${#TASKS[@]}"
echo "  - Strategy used: ${STRATEGY}"
echo "  - Seeds used: ${SEEDS}"
echo "  - Output directory: ${OUTPUT_DIR}"

echo ""
echo "📋 Results available:"
if [ ${successful_tasks} -gt 0 ]; then
    echo "  # List consensus result files:"
    echo "  ls -la ${OUTPUT_DIR}/*_consensus_${STRATEGY}.jsonl"
    echo ""
    echo "  # Example: View consensus results for raw_qa:"
    echo "  head -1 ${OUTPUT_DIR}/raw_qa_consensus_${STRATEGY}.jsonl | python -m json.tool"
    echo ""
    echo "  # Count consensus results:"
    echo "  wc -l ${OUTPUT_DIR}/*_consensus_${STRATEGY}.jsonl"
else
    echo "  ❌ No consensus results generated"
fi

echo ""
echo "⏰ Script started at: $(date)"
echo "⏰ Script completed at: $(date)"

if [ ${failed_tasks} -eq 0 ]; then
    echo "✅ All tasks completed successfully!"
else
    echo "⚠️  Some tasks failed. Check the logs above for details."
fi

echo ""
echo "🔬 Next steps for your rebuttal:"
echo "  1. Use the consensus results for evaluation"
echo "  2. Compare consensus confidence scores across different strategies"
echo "  3. Analyze agreement patterns between different seeds"
echo "  4. Report improved performance with self-consistency in your response" 