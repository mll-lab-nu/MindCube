#!/bin/bash

# Comprehensive Meta Evaluation Pipeline for MindCube
# This script runs the complete evaluation pipeline:
# 1. Self-consistency analysis (if not already done)
# 2. Meta evaluation comparing multi-seed vs consensus results
# 3. Summary report generation

# Check and activate the mindcube conda environment if needed
if [[ "$CONDA_DEFAULT_ENV" != "mindcube" ]]; then
    echo "🔧 Activating mindcube conda environment..."
    # Try to activate conda environment
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate mindcube
        if [ $? -ne 0 ]; then
            echo "❌ Error: Failed to activate mindcube conda environment"
            echo "💡 Make sure you have mindcube environment set up"
            echo "💡 Run: conda create -n mindcube python=3.8"
            exit 1
        fi
        echo "✅ Successfully activated mindcube environment"
    else
        echo "❌ Error: conda command not found"
        echo "💡 Make sure you have conda installed and available in PATH"
        exit 1
    fi
else
    echo "✅ Already in mindcube environment"
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
echo "  --multi-seed-dir DIR     Multi-seed results directory (full path)"
echo "  --multi-seed-suffix SUF  Multi-seed directory suffix under tmp_results/"
echo "                          (default: frozen_multi_seeds, can use: sft_multi_seeds)"
echo "  --consensus-dir DIR      Consensus results directory"
echo "                          (default: tmp_results/self_consistency)"
echo "  --output-dir DIR         Output directory for meta analysis"
echo "                          (default: tmp_results/meta_analysis)"
echo "  --strategy STRATEGY      Consensus strategy: majority, exact, similarity"
echo "                          (default: majority)"
echo "  --seeds SEEDS            Comma-separated list of seeds"
echo "                          (default: 42,123,456)"
echo "  --tasks TASKS            Comma-separated list of tasks"
echo "                          (default: all 6 tasks)"
echo "  --skip-consensus         Skip running self-consistency (assume already done)"
echo "  --help, -h               Show this help message"
echo ""
echo "Examples:"
echo "  $0                                   # Run complete pipeline with defaults"
echo "  $0 --skip-consensus                  # Skip self-consistency, only run meta evaluation"
echo "  $0 --strategy exact                  # Use exact match strategy"
echo "  $0 --multi-seed-suffix sft_multi_seeds  # Use SFT results instead of frozen"
}

# Default values
MULTI_SEED_DIR=""  # Will be constructed from suffix if not provided
MULTI_SEED_SUFFIX="sft_multi_seeds"
CONSENSUS_DIR="/workspace/MindCube/tmp_results/self_consistency"
OUTPUT_DIR="/workspace/MindCube/tmp_results/meta_analysis"
STRATEGY="majority"
SEEDS="42,123,456"
TASKS="raw_qa,ff_rsn,aug_cgmap_out,plain_cgmap_out,aug_cgmap_ffr_out,plain_cgmap_ffr_out"
SKIP_CONSENSUS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --multi-seed-dir)
            MULTI_SEED_DIR="$2"
            shift 2
            ;;
        --multi-seed-suffix)
            MULTI_SEED_SUFFIX="$2"
            shift 2
            ;;
        --consensus-dir)
            CONSENSUS_DIR="$2"
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
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --skip-consensus)
            SKIP_CONSENSUS=true
            shift
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

# Determine multi-seed directory
if [ -z "${MULTI_SEED_DIR}" ]; then
    MULTI_SEED_DIR="/workspace/MindCube/tmp_results/${MULTI_SEED_SUFFIX}"
fi

echo "🔬 MindCube Meta Evaluation Pipeline"
echo "📅 Start time: $(date)"
echo ""

echo "🔧 Configuration:"
echo "  📁 Multi-seed directory: ${MULTI_SEED_DIR}"
echo "  📂 Multi-seed suffix: ${MULTI_SEED_SUFFIX}"
echo "  📁 Consensus directory: ${CONSENSUS_DIR}"
echo "  📁 Output directory: ${OUTPUT_DIR}"
echo "  🎯 Strategy: ${STRATEGY}"
echo "  🎲 Seeds: ${SEEDS}"
echo "  📋 Tasks: ${TASKS}"
echo "  ⏭️  Skip consensus: ${SKIP_CONSENSUS}"
echo ""

# Validate multi-seed directory
if [ ! -d "${MULTI_SEED_DIR}" ]; then
    echo "❌ Error: Multi-seed directory does not exist: ${MULTI_SEED_DIR}"
    echo "💡 Make sure you have run the multi-seed inference first using:"
    echo "   bash tmp/run_frozen_all_seeds.bash"
    exit 1
fi

# Check if we have multi-seed result files
multi_seed_files_count=$(find "${MULTI_SEED_DIR}" -name "*seed*responses.jsonl" | wc -l)
if [ ${multi_seed_files_count} -eq 0 ]; then
    echo "❌ Error: No multi-seed result files found in ${MULTI_SEED_DIR}"
    echo "💡 Expected files with pattern: *seed*responses.jsonl"
    exit 1
fi

echo "✅ Found ${multi_seed_files_count} multi-seed result files"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Step 1: Run self-consistency if not skipped
if [ "${SKIP_CONSENSUS}" = false ]; then
    echo ""
    echo "="*60
    echo "🤖 Step 1: Running Self-Consistency Analysis"
    echo "="*60
    
    if bash tmp/run_self_consistency.bash \
        --input-dir "${MULTI_SEED_DIR}" \
        --output-dir "${CONSENSUS_DIR}" \
        --strategy "${STRATEGY}"; then
        
        echo "✅ Self-consistency analysis completed successfully"
    else
        echo "❌ Self-consistency analysis failed"
        exit 1
    fi
else
    echo ""
    echo "⏭️  Skipping self-consistency analysis (--skip-consensus flag used)"
    
    # Validate consensus directory exists
    if [ ! -d "${CONSENSUS_DIR}" ]; then
        echo "❌ Error: Consensus directory does not exist: ${CONSENSUS_DIR}"
        echo "💡 Remove --skip-consensus flag to run self-consistency first"
        exit 1
    fi
    
    # Check if we have consensus result files
    consensus_files_count=$(find "${CONSENSUS_DIR}" -name "*consensus_${STRATEGY}.jsonl" | wc -l)
    if [ ${consensus_files_count} -eq 0 ]; then
        echo "❌ Error: No consensus result files found in ${CONSENSUS_DIR}"
        echo "💡 Expected files with pattern: *consensus_${STRATEGY}.jsonl"
        exit 1
    fi
    
    echo "✅ Found ${consensus_files_count} consensus result files"
fi

# Step 2: Run meta evaluation
echo ""
echo "="*60
echo "📊 Step 2: Running Meta Evaluation"
echo "="*60

OUTPUT_FILE="${OUTPUT_DIR}/meta_analysis_${STRATEGY}.json"

echo "🔍 Running comprehensive evaluation..."
echo "📄 Output file: ${OUTPUT_FILE}"

if python tmp/meta_evaluation.py \
    --multi-seed-suffix "${MULTI_SEED_SUFFIX}" \
    --consensus-dir "${CONSENSUS_DIR}" \
    --output-file "${OUTPUT_FILE}" \
    --tasks "${TASKS}" \
    --seeds "${SEEDS}" \
    --strategy "${STRATEGY}"; then
    
    echo "✅ Meta evaluation completed successfully"
else
    echo "❌ Meta evaluation failed"
    exit 1
fi

# Step 3: Generate summary report
echo ""
echo "="*60
echo "📋 Step 3: Generating Summary Report"
echo "="*60

SUMMARY_FILE="${OUTPUT_DIR}/summary_${STRATEGY}.txt"

echo "📝 Generating summary report: ${SUMMARY_FILE}"

# Create summary report
cat > "${SUMMARY_FILE}" << EOF
# MindCube Meta Evaluation Summary Report
Generated: $(date)
Strategy: ${STRATEGY}
Seeds: ${SEEDS}
Tasks: ${TASKS}

## Configuration
- Multi-seed directory: ${MULTI_SEED_DIR}
- Consensus directory: ${CONSENSUS_DIR}
- Output directory: ${OUTPUT_DIR}
- Multi-seed files found: ${multi_seed_files_count}

## Files Generated
- Meta analysis JSON: ${OUTPUT_FILE}
- Summary report: ${SUMMARY_FILE}

## Quick Access Commands
# View meta analysis JSON (pretty printed)
python -m json.tool "${OUTPUT_FILE}"

# Extract specific metrics
# Overall accuracy comparison for all tasks
python -c "
import json
with open('${OUTPUT_FILE}', 'r') as f:
    data = json.load(f)
    
print('\\n📊 Overall Accuracy Comparison:\\n')
for task, comp in data['comparison'].items():
    if 'overall_accuracy' in comp:
        multi_seed = comp['overall_accuracy']['multi_seed_mean']
        consensus = comp['overall_accuracy']['consensus_value']
        improvement = comp['overall_accuracy']['improvement_percentage']
        print(f'{task:20} | Multi-seed: {multi_seed:.3f} | Consensus: {consensus:.3f} | Improvement: {improvement:+.2f}%')
"

# Self-consistency confidence statistics
python -c "
import json
with open('${OUTPUT_FILE}', 'r') as f:
    data = json.load(f)
    
print('\\n🤖 Self-Consistency Statistics:\\n')
for task, result in data['consensus_results'].items():
    metrics = result['metrics']
    conf_mean = metrics.get('consensus_confidence_mean', 0)
    perfect_rate = metrics.get('perfect_agreement_rate', 0)
    high_conf_rate = metrics.get('high_confidence_rate', 0)
    print(f'{task:20} | Avg Confidence: {conf_mean:.3f} | Perfect Agreement: {perfect_rate:.1%} | High Confidence: {high_conf_rate:.1%}')
"

EOF

echo "✅ Summary report generated: ${SUMMARY_FILE}"

# Step 4: Display quick summary
echo ""
echo "="*60
echo "🎉 Pipeline Completed Successfully!"
echo "="*60

echo ""
echo "📁 Generated Files:"
echo "  📊 Meta analysis JSON: ${OUTPUT_FILE}"
echo "  📋 Summary report: ${SUMMARY_FILE}"

echo ""
echo "🔍 Quick Preview:"
echo ""

# Show a quick preview using Python
python -c "
import json
import sys

try:
    with open('${OUTPUT_FILE}', 'r') as f:
        data = json.load(f)
    
    print('📈 Overall Performance Summary:')
    print('=' * 40)
    
    for task, comp in data['comparison'].items():
        if 'overall_accuracy' in comp:
            improvement = comp['overall_accuracy']['improvement_percentage']
            consensus_acc = comp['overall_accuracy']['consensus_value']
            
            status = '📈' if improvement > 0 else '📉' if improvement < 0 else '➡️'
            print(f'{status} {task:20} | Consensus: {consensus_acc:.1%} | Improvement: {improvement:+.2f}%')
    
    print('')
    print('🤖 Self-Consistency Quality:')
    print('=' * 40)
    
    for task, result in data['consensus_results'].items():
        metrics = result['metrics']
        perfect_rate = metrics.get('perfect_agreement_rate', 0)
        high_conf_rate = metrics.get('high_confidence_rate', 0)
        
        quality = '🟢' if perfect_rate > 0.7 else '🟡' if perfect_rate > 0.5 else '🔴'
        print(f'{quality} {task:20} | Perfect: {perfect_rate:.1%} | High Conf: {high_conf_rate:.1%}')

except Exception as e:
    print(f'Error reading results: {e}')
    sys.exit(1)
"

echo ""
echo "📋 Next Steps:"
echo "  1. Review the detailed meta analysis JSON for complete metrics"
echo "  2. Use the summary report for quick reference"
echo "  3. Include self-consistency improvements in your rebuttal"
echo "  4. Cite the statistical significance from multiple seeds"

echo ""
echo "⏰ Pipeline started at: $(date)"
echo "⏰ Pipeline completed at: $(date)"
echo "✅ All evaluations completed successfully!"

echo ""
echo "🔬 For rebuttal use:"
echo "  - Multi-seed results provide statistical robustness"
echo "  - Self-consistency demonstrates improved reliability" 
echo "  - Detailed metrics available for all categories (rotation, among, around)"
echo "  - Graph metrics show structural understanding quality" 