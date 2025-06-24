#!/bin/bash

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${YELLOW}=== $1 ===${NC}"
}

# Function to run command with error handling
run_with_check() {
    local cmd="$1"
    local description="$2"
    local allow_failure="${3:-false}"
    
    print_info "Running: $description"
    if eval "$cmd"; then
        print_success "$description completed successfully"
        return 0
    else
        if [[ "$allow_failure" == "true" ]]; then
            print_warning "$description failed but continuing..."
            print_warning "Command: $cmd"
            return 1
        else
            print_error "$description failed"
            print_error "Command: $cmd"
            exit 1
        fi
    fi
}

# Assert that current directory contains "MindCube" in its name
current_dir=$(pwd)
if [[ "$current_dir" != *"MindCube"* ]]; then
    print_error "Current directory does not contain 'MindCube' in its path"
    print_error "Current directory: $current_dir"
    exit 1
fi

print_success "Directory assertion passed: Currently in $current_dir"

# Start time
start_time=$(date)
print_info "Starting evaluation data generation at: $start_time"

print_step "Step 1/4: Environment Setup"

# Check if we're already in the mindcube environment
if [[ "$CONDA_DEFAULT_ENV" == "mindcube" ]]; then
    print_success "Already in conda environment 'mindcube'"
elif [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    print_warning "Currently in conda environment: $CONDA_DEFAULT_ENV"
    print_info "Attempting to activate 'mindcube' environment..."
    # Try to source conda and activate
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        if conda activate mindcube 2>/dev/null; then
            print_success "Conda environment 'mindcube' activated"
        else
            print_error "Failed to activate conda environment 'mindcube'"
            print_info "Please manually activate the environment with: conda activate mindcube"
            exit 1
        fi
    else
        print_error "Conda not found in PATH"
        exit 1
    fi
else
    print_warning "No conda environment detected"
    print_info "Attempting to activate 'mindcube' environment..."
    # Try to source conda and activate
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
        if conda activate mindcube 2>/dev/null; then
            print_success "Conda environment 'mindcube' activated"
        else
            print_error "Failed to activate conda environment 'mindcube'"
            print_info "Please manually activate the environment with: conda activate mindcube"
            exit 1
        fi
    else
        print_error "Conda not found in PATH"
        exit 1
    fi
fi

print_step "Step 2/4: Data Scaffold Processing"
print_info "Processing 3 input files with full pipeline..."

datasets=("MindCube_train.jsonl" "MindCube_tinybench.jsonl")
for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    print_info "Processing dataset $((i+1))/3: $dataset"
    run_with_check "python scripts/data_processing.py --input data/raw/$dataset --task full_pipeline" "Data processing for $dataset"
done

print_step "Step 3/4: General Prompts Generation"
print_info "Generating prompts for all task configurations..."

failed_prompts=()
for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    print_info "Generating prompts $((i+1))/3: $dataset"
    if ! run_with_check "python scripts/generate_prompts.py --input data/scaffold/all/$dataset --all_tasks" "Prompt generation for $dataset" "true"; then
        failed_prompts+=("$dataset")
    fi
done

if [[ ${#failed_prompts[@]} -gt 0 ]]; then
    print_warning "Some prompt generation tasks failed:"
    for failed in "${failed_prompts[@]}"; do
        print_warning "  - $failed"
    done
    print_info "Continuing with remaining tasks..."
fi

print_step "Step 4/4: SFT Format Conversion"
print_info "Converting training data to SFT format for Qwen2.5-VL..."
run_with_check "python scripts/convert_to_sft.py --input_dir data/prompts/general/ --model qwen2.5vl" "SFT format conversion"

# End time and summary
end_time=$(date)
print_step "Evaluation Data Generation Complete!"
print_success "Started at: $start_time"
print_success "Completed at: $end_time"
print_success "All tasks completed successfully! 🎉"

print_info "Generated files should be available in:"
print_info "  - data/scaffold/all/ (scaffolded data)"
print_info "  - data/prompts/general/ (generated prompts)"
print_info "  - SFT formatted data (as specified by convert_to_sft.py)"