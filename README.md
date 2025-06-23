# Spatial Mental Modeling from Limited Views

**A comprehensive spatial reasoning dataset generation and evaluation framework for vision-language models.**

## 🌟 Overview

MindCube is a modular framework for generating and evaluating spatial reasoning datasets for multimodal AI models. The project follows a complete 5-step pipeline from raw data to model evaluation, with specialized modules for scaffold data curation, prompt generation, model inference, training, and comprehensive evaluation.

## 🔄 Processing Pipeline

```
Raw Data → Scaffold Data → Model Prompts → SFT Training → Model Inference & Evaluation
    ↓           ↓              ↓             ↓                    ↓
  Step 1      Step 2        Step 3        Step 4              Step 5
 Input       Cogmap +      8 Task        Multi-Model         Performance
Processing   Reasoning     Variants      Training            Metrics
```

### Pipeline Steps

- **Step 1**: Raw Data Processing - Original question-answer pairs with spatial annotations
- **Step 2**: Scaffold Data Generation - Cognitive maps and reasoning chains
- **Step 3**: Model Prompt Generation - 8 task variants for comprehensive training
- **Step 4**: SFT Training Data Generation - Multi-model format support (Qwen2.5-VL, LLaVA, InstructBLIP)
- **Step 5**: Model Operations & Evaluation - Inference and comprehensive evaluation metrics

## 🚀 Quick Start

### Environment Setup

```bash
# Create conda environment
conda create -n mindcube python=3.8
conda activate mindcube

# Install dependencies
pip install torch torchvision transformers
pip install opencv-python pillow datasets accelerate
pip install vllm  # Optional: for faster inference
```

## 📋 Core Usage

### 🏗️ **Step 2: Scaffold Data Generation**

Generate complete scaffold data (cognitive maps + reasoning chains):

```bash
python scripts/data_processing.py \
  --input data/raw/MindCube_tinybench.jsonl \
  --task full_pipeline

python scripts/data_processing.py \
  --input data/raw/MindCube_train.jsonl \
  --task full_pipeline
```

### 💬 **Step 3: Prompt Generation**

Generate all 8 prompt task types from scaffold data:

```bash
python scripts/generate_prompts.py \
  --input data/scaffold/all/MindCube_tinybench.jsonl \
  --all_tasks

python scripts/generate_prompts.py \
  --input data/scaffold/all/MindCube_train.jsonl \
  --all_tasks
```

### 🏋️ **Step 4: SFT Data Conversion**

Convert prompt files to training format:

```bash
# Convert to Qwen2.5-VL format
python scripts/convert_to_sft.py \
  --input_dir data/prompts/general/ \
  --model qwen2.5vl

# Convert to LLaVA format
python scripts/convert_to_sft.py \
  --input_dir data/prompts/general/ \
  --model llava
```

### 🤖 **Step 5: Model Inference**

Run model inference:

```bash
# Basic Qwen2.5-VL inference
python scripts/run_inference.py \
  --model-type qwen2.5vl \
  --input-file data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-dir results/

# Accelerated inference with vLLM
python scripts/run_inference.py \
  --model-type qwen2.5vl \
  --backend vllm \
  --input-file data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-dir results/
```

### 📊 **Model Evaluation**

Evaluate model performance:

```bash
python scripts/run_evaluation.py \
  --predictions results/model_responses.jsonl \
  --ground_truth data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-dir evaluation_results/
```

## 🔄 **Complete Workflow Example**

```bash
# Generate scaffold data
python scripts/data_processing.py \
  --input data/raw/MindCube_tinybench.jsonl \
  --task full_pipeline

# Generate all prompt types
python scripts/generate_prompts.py \
  --input data/scaffold/all/MindCube_tinybench.jsonl \
  --all_tasks

# Convert to SFT training format
python scripts/convert_to_sft.py \
  --input_dir data/prompts/general/ \
  --model qwen2.5vl

# Run model inference
python scripts/run_inference.py \
  --model-type qwen2.5vl \
  --input-file data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-dir results/

# Evaluate model performance
python scripts/run_evaluation.py \
  --predictions results/MindCube_tinybench_raw_qa_qwen2.5-vl-3b-instruct_responses.jsonl \
  --ground_truth data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-dir evaluation_results/
```

## 📋 **Available Features**

### Scaffold Generation Tasks
- `cogmap` - Generate cognitive maps only
- `reasoning` - Generate reasoning chains only
- `full_pipeline` - Generate both (recommended)

### Prompt Generation Tasks (8 types)
- `raw_qa` - Basic question-answering without scaffolds
- `ff_rsn` - Free-form reasoning generation
- `aug_cgmap_in` - Augmented cognitive map as input
- `aug_cgmap_out` - Augmented cognitive map as output
- `plain_cgmap_out` - Plain cognitive map as output
- `plain_cgmap_ffr_out` - Plain cognitive map with first-few reasoning
- `aug_cgmap_ffr_out` - Augmented cognitive map with first-few reasoning
- `cgmap_in_ffr_out` - Cognitive map input with first-few reasoning output

### Supported SFT Training Models
- `qwen2.5vl` - Qwen2.5-VL conversation format
- `llava` - LLaVA training format
- `instructblip` - InstructBLIP format

### Evaluation Metrics
- **Accuracy**: Basic answer correctness
- **F1 Score**: Precision and recall balance
- **Cognitive Map Similarity**: Semantic similarity evaluation
- **Reasoning Quality**: Multi-step reasoning assessment
- **Spatial Understanding**: Spatial relationship accuracy

## 📋 **Data Format Specifications**

### Raw Input Data Format
```json
{
  "id": "unique_identifier",
  "category": "spatial",
  "type": "rotation|translation|among|around",
  "meta_info": {"setting": "indoor|outdoor"},
  "question": "Based on the images provided: [question text] A. Option1 B. Option2 ...",
  "images": ["path/to/image1.jpg", "path/to/image2.jpg"],
  "gt_answer": "A|B|C|D"
}
```

### Scaffold Data Format (After Step 2)
```json
{
  "id": "unique_identifier",
  "category": "spatial",
  "type": "rotation",
  "question": "Based on the images provided: ...",
  "images": ["image1.jpg", "image2.jpg"],
  "gt_answer": "A",
  "cogmap": "The scene shows spatial relationships between...",
  "reasoning_chain": "Step-by-step reasoning process..."
}
```

### SFT Training Format Example (Qwen2.5-VL)
```json
{
  "images": ["image1.jpg", "image2.jpg"],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n<image>\n[Task]\nAnalyze spatial relationships..."
    },
    {
      "from": "gpt",
      "value": "<think>Step-by-step reasoning...</think><answer>A. Above</answer>"
    }
  ]
}
```

## 🛠️ **Command Help**

Get help for any script:
```bash
python scripts/data_processing.py --help
python scripts/generate_prompts.py --help
python scripts/run_inference.py --help
python scripts/run_evaluation.py --help
```

## 📝 **License**

This project is licensed under the MIT License.

---

**MindCube Framework** - Advancing spatial reasoning capabilities in vision-language models through comprehensive dataset generation and evaluation.


