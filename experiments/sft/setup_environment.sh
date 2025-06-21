#!/bin/bash

# ==============================================================================
# MindCube Environment Setup Script
# This script prepares the Qwen environment for MindCube training
# ==============================================================================

echo "=============================================================================="
echo "MindCube Environment Setup"
echo "=============================================================================="

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Project root: $PROJECT_ROOT"
echo "Setup script location: $SCRIPT_DIR"

# Check if Qwen2.5-VL exists
if [ ! -d "$PROJECT_ROOT/Qwen2.5-VL" ]; then
    echo ""
    echo "❌ Qwen2.5-VL directory not found!"
    echo "Please make sure you have cloned the Qwen2.5-VL repository in the project root."
    echo ""
    echo "To clone Qwen2.5-VL:"
    echo "  cd $PROJECT_ROOT"
    echo "  git clone https://github.com/QwenLM/Qwen2.5-VL.git"
    echo ""
    exit 1
fi

# Check if the target init file exists
QWEN_INIT_FILE="$PROJECT_ROOT/Qwen2.5-VL/qwen-vl-finetune/qwenvl/data/__init__.py"
if [ ! -f "$QWEN_INIT_FILE" ]; then
    echo ""
    echo "❌ Qwen data init file not found at:"
    echo "   $QWEN_INIT_FILE"
    echo "Please check your Qwen2.5-VL installation."
    echo ""
    exit 1
fi

echo ""
echo "✅ Qwen2.5-VL installation found"

# Check if already patched
echo ""
echo "Checking current patch status..."
python "$SCRIPT_DIR/patch_qwen_data.py" verify
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Environment is already set up with MindCube datasets!"
    echo "You can proceed with training."
    exit 0
fi

echo ""
echo "Qwen environment not yet configured for MindCube datasets."
echo ""

# Ask user for confirmation
read -p "Do you want to patch Qwen's data configuration with MindCube datasets? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled by user."
    exit 1
fi

echo ""
echo "Patching Qwen data configuration..."
python "$SCRIPT_DIR/patch_qwen_data.py" patch

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Environment setup completed successfully!"
    echo ""
    echo "Available MindCube datasets:"
    echo "  - raw_qa"
    echo "  - aug_cgmap_out" 
    echo "  - plain_cgmap_out"
    echo "  - ff_rsn"
    echo "  - aug_cgmap_ffr_out"
    echo ""
    echo "You can now run training with any of these datasets:"
    echo "  cd $SCRIPT_DIR"
    echo "  ./train_qwen_sft.sh config_raw_qa.sh"
    echo ""
else
    echo ""
    echo "❌ Setup failed!"
    exit 1
fi 