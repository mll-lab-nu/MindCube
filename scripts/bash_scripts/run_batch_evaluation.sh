#!/bin/bash

# Script to run batch evaluation on all JSONL files in a directory
# Author: MindCube Team
# Usage: bash scripts/bash_scripts/run_batch_evaluation.sh <input_directory>

echo "🚀 Starting batch evaluation..."
echo "📅 Start time: $(date)"

# Check if input directory is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide an input directory"
    echo "Usage: bash scripts/bash_scripts/run_batch_evaluation.sh <input_directory>"
    echo ""
    echo "Examples:"
    echo "  bash scripts/bash_scripts/run_batch_evaluation.sh ./data/results/sft/raw_qa"
    echo "  bash scripts/bash_scripts/run_batch_evaluation.sh ./data/results/frozen_vlm"
    exit 1
fi

INPUT_DIR="$1"
EVAL_BASE_DIR="./data/evaluate"

# Check if input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "❌ Error: Input directory '${INPUT_DIR}' does not exist"
    exit 1
fi

echo "📁 Input directory: ${INPUT_DIR}"
echo "📁 Evaluation base directory: ${EVAL_BASE_DIR}"

# Create evaluation base directory
mkdir -p "${EVAL_BASE_DIR}"

# Find all JSONL files recursively
echo "🔍 Searching for JSONL files..."
mapfile -t jsonl_files < <(find "${INPUT_DIR}" -name "*.jsonl" -type f | sort)

if [ ${#jsonl_files[@]} -eq 0 ]; then
    echo "⚠️  No JSONL files found in ${INPUT_DIR}"
    exit 0
fi

echo "📋 Found ${#jsonl_files[@]} JSONL files to evaluate:"
for file in "${jsonl_files[@]}"; do
    echo "  - $(basename ${file})"
done

echo ""

# Arrays to store results
declare -a eval_results=()
declare -a eval_files=()
declare -a eval_accuracies=()

# Function to generate output path
generate_output_path() {
    local input_file="$1"
    
    # Get relative path from input directory
    local rel_path=$(realpath --relative-to="${INPUT_DIR}" "${input_file}")
    
    # Get the directory part and filename
    local file_dir=$(dirname "${rel_path}")
    local filename=$(basename "${rel_path}" .jsonl)
    
    # Replace 'results' with 'evaluate' in the input directory path
    local eval_dir=$(echo "${INPUT_DIR}" | sed 's|/results/|/evaluate/|g')
    
    # If no 'results' found, just use the evaluate base dir with relative structure
    if [[ "${INPUT_DIR}" == "${eval_dir}" ]]; then
        # No 'results' in path, create structure under evaluate base
        eval_dir="${EVAL_BASE_DIR}/$(basename ${INPUT_DIR})"
    fi
    
    # Create the output directory (including subdirectories if they exist)
    if [ "${file_dir}" != "." ]; then
        local output_dir="${eval_dir}/${file_dir}"
    else
        local output_dir="${eval_dir}"
    fi
    
    mkdir -p "${output_dir}"
    
    # Return the output file path with _eval_results.json suffix
    echo "${output_dir}/${filename}_eval_results.json"
}

# Function to extract accuracy from evaluation output
extract_accuracy() {
    local log_content="$1"
    
    # Extract accuracy percentage from the log output
    local accuracy=$(echo "${log_content}" | grep -o "Evaluation completed: [0-9.]*%" | sed 's/.*: \([0-9.]*\)%.*/\1/')
    
    if [ -n "${accuracy}" ]; then
        echo "${accuracy}"
    else
        echo "N/A"
    fi
}

# Main evaluation loop
echo "🎬 Starting evaluation process..."
echo ""

total_files=${#jsonl_files[@]}
current_file=0
successful_evals=0
failed_evals=0

for jsonl_file in "${jsonl_files[@]}"; do
    ((current_file++))
    
    echo "📊 [${current_file}/${total_files}] Evaluating: $(basename ${jsonl_file})"
    
    # Generate output path
    output_path=$(generate_output_path "${jsonl_file}")
    
    echo "📝 Input: ${jsonl_file}"
    echo "📤 Output: ${output_path}"
    
    # Check if input file is not empty
    if [ ! -s "${jsonl_file}" ]; then
        echo "⚠️  Warning: File is empty, skipping..."
        eval_results+=("EMPTY")
        eval_files+=("$(basename ${jsonl_file})")
        eval_accuracies+=("N/A")
        ((failed_evals++))
        echo ""
        continue
    fi
    
    # Run evaluation
    echo "🔄 Running evaluation..."
    eval_output=$(python scripts/run_evaluation.py -i "${jsonl_file}" -o "${output_path}" 2>&1)
    eval_exit_code=$?
    
    # Check if evaluation was successful
    if [ ${eval_exit_code} -eq 0 ]; then
        echo "✅ Evaluation completed successfully"
        
        # Extract accuracy from output
        accuracy=$(extract_accuracy "${eval_output}")
        
        eval_results+=("SUCCESS")
        eval_files+=("$(basename ${jsonl_file})")
        eval_accuracies+=("${accuracy}")
        ((successful_evals++))
        
        echo "📈 Accuracy: ${accuracy}%"
    else
        echo "❌ Evaluation failed"
        echo "Error output:"
        echo "${eval_output}" | head -10
        
        eval_results+=("FAILED")
        eval_files+=("$(basename ${jsonl_file})")
        eval_accuracies+=("N/A")
        ((failed_evals++))
    fi
    
    echo ""
done

# Generate summary report
echo "🎯 Batch Evaluation Summary"
echo "=========================="
echo "📅 Completed at: $(date)"
echo "📁 Input directory: ${INPUT_DIR}"
echo "📁 Output directory: ${EVAL_BASE_DIR}"
echo "📊 Statistics:"
echo "  - Total files: ${total_files}"
echo "  - Successful: ${successful_evals}"
echo "  - Failed: ${failed_evals}"
echo "  - Empty files: $((total_files - successful_evals - failed_evals + $(printf '%s\n' "${eval_results[@]}" | grep -c "EMPTY")))"

echo ""
echo "📋 Detailed Results:"
echo "===================="

# Print results table
printf "%-50s %-10s %-10s\n" "File" "Status" "Accuracy"
printf "%-50s %-10s %-10s\n" "$(printf '%.0s-' {1..50})" "$(printf '%.0s-' {1..10})" "$(printf '%.0s-' {1..10})"

for i in "${!eval_files[@]}"; do
    file="${eval_files[i]}"
    status="${eval_results[i]}"
    accuracy="${eval_accuracies[i]}"
    
    # Truncate filename if too long
    if [ ${#file} -gt 47 ]; then
        display_file="${file:0:44}..."
    else
        display_file="${file}"
    fi
    
    # Color coding for status
    case "${status}" in
        "SUCCESS")
            status_display="✅ OK"
            ;;
        "FAILED")
            status_display="❌ FAIL"
            ;;
        "EMPTY")
            status_display="⚠️  EMPTY"
            ;;
        *)
            status_display="${status}"
            ;;
    esac
    
    printf "%-50s %-10s %-10s\n" "${display_file}" "${status_display}" "${accuracy}%"
done

echo ""

# Calculate average accuracy for successful evaluations
if [ ${successful_evals} -gt 0 ]; then
    total_accuracy=0
    count=0
    
    for acc in "${eval_accuracies[@]}"; do
        if [[ "${acc}" != "N/A" ]] && [[ "${acc}" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            total_accuracy=$(echo "${total_accuracy} + ${acc}" | bc -l)
            ((count++))
        fi
    done
    
    if [ ${count} -gt 0 ]; then
        avg_accuracy=$(echo "scale=2; ${total_accuracy} / ${count}" | bc -l)
        echo "📈 Average Accuracy: ${avg_accuracy}%"
    fi
fi

echo ""
echo "🔍 To view detailed results:"
echo "  find ${EVAL_BASE_DIR} -name '*_eval_results.json' | head -5"
echo ""
echo "📁 Evaluation files saved to:"
echo "  ${EVAL_BASE_DIR}"
echo ""
echo "💡 Output Structure:"
echo "  Input:  ./data/results/sft/raw_qa/file.jsonl"
echo "  Output: ./data/evaluate/sft/raw_qa/file_eval_results.json"

echo ""
echo "⏰ Total execution time: $(date)"
echo "✅ Batch evaluation completed!" 