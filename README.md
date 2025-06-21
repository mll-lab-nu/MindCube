# MindCube Dataset Framework

**Version 2.0** - Comprehensive spatial reasoning dataset generation and evaluation framework for vision-language models.

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

### Step 1: Raw Data Processing
- Original question-answer pairs with images
- Spatial relationship annotations (around, among, translation, rotation)
- Input format validation and preprocessing

### Step 2: Scaffold Data Generation
- **Cognitive Maps**: Scene understanding and object relationships
- **Reasoning Chains**: Step-by-step reasoning for spatial tasks
- **Full Pipeline**: Combined cognitive maps and reasoning (recommended)

### Step 3: Model Prompt Generation
- **8 Task Types**: Various input/output configurations for model training
- **Template-based Generation**: Consistent prompt formatting
- **Multi-modal Integration**: Text and visual input processing

### Step 4: SFT Training Data Generation
- **Model-Specific Formats**: Qwen2.5-VL, LLaVA, InstructBLIP support
- **Conversation Format**: Structured training data conversion
- **Extensible Architecture**: Easy to add new model formats

### Step 5: Model Operations & Evaluation
- **Multi-Model Inference**: Support for various VLM architectures
- **Comprehensive Evaluation**: Spatial reasoning metrics and cogmap evaluation
- **Performance Analysis**: Detailed metrics and error analysis

## 📁 Project Structure

```
07_MindCube_new/
├── src/                                # Core implementation modules
│   ├── scaffold_curation/              # Step 2: Scaffold data generation
│   │   ├── processors.py               # Data processing pipeline
│   │   ├── formatters.py               # Output formatting utilities
│   │   ├── cogmap/                     # Cognitive map generation
│   │   └── reasoning/                  # Reasoning chain generation
│   │
│   ├── prompt_generation/              # Step 3: Model prompt generation
│   │   ├── processors.py               # Prompt processing logic
│   │   ├── generators.py               # Task-specific generators
│   │   └── templates.py                # Prompt templates library
│   │
│   ├── training/                       # Step 4: Model training utilities
│   ├── inference/                      # Model inference interfaces  
│   │
│   ├── evaluation/                     # Step 5: Evaluation framework
│   │   ├── evaluator.py                # Main evaluation interface
│   │   ├── core/                       # Base evaluation metrics
│   │   ├── cogmap/                     # Cognitive map evaluation
│   │   └── metrics/                    # Specialized metrics
│   │
│   └── utils/                          # Shared utilities
│       ├── io_utils.py                 # File I/O operations
│       ├── text_utils.py               # Text processing utilities
│       └── spatial_utils.py            # Spatial reasoning utilities
│
├── scripts/                            # User-friendly interfaces
│   ├── data_processing.py              # Scaffold generation script
│   ├── generate_prompts.py             # Prompt generation script
│   ├── generate_reasoning.py           # Reasoning chain generation
│   ├── convert_to_sft.py               # SFT data conversion script
│   ├── run_inference.py                # Model inference script
│   ├── run_evaluation.py               # Evaluation script
│   ├── run_training.py                 # Training script
│   └── bash_scripts/                   # Batch processing scripts
│
├── data/                               # Data storage with organized structure
│   ├── raw/                            # Original input data
│   ├── scaffold/                       # Scaffold generation outputs
│   │   └── all/                        # Full pipeline outputs (recommended)
│   ├── prompts/                        # Generated prompts
│   │   ├── general/                    # All 8 prompt task types
│   │   └── training/                   # SFT training data
│   │       ├── qwen2.5vl/              # Qwen SFT training data
│   │       ├── llava/                  # LLaVA SFT training data
│   │       └── instructblip/           # InstructBLIP SFT training data
│   ├── results/                        # Model inference results
│   └── tmp/                            # Temporary processing files
│
├── configs/                            # Configuration files
│   └── qwen_inference.json             # Model inference configurations
│
├── experiments/                        # Experiment tracking
│   └── sft/                            # SFT training experiments
│
├── notebooks/                          # Jupyter notebooks for analysis
├── logs/                               # Processing and training logs
├── checkpoints/                        # Model checkpoints
└── Qwen2.5-VL/                        # Local model storage
```

## 🚀 Quick Start

### Environment Setup

```bash
# Create conda environment
conda create -n mindcube python=3.8
conda activate mindcube

# Install dependencies (create requirements.txt based on your needs)
# pip install -r requirements.txt

# Key dependencies typically include:
pip install torch torchvision transformers
pip install opencv-python pillow
pip install datasets accelerate
pip install vllm  # Optional: for faster inference
```

## 📋 Core Usage

### 🏗️ **Step 2: Scaffold Data Generation** (`data_processing.py`)

#### ⭐ **Full Pipeline Generation (Recommended)**
```bash
# Generate complete scaffold data (cognitive_map + reasoning_chain)
python scripts/data_processing.py \
  --input data/raw/MindCube_tinybench.jsonl \
  --task full_pipeline
python scripts/data_processing.py \
  --input data/raw/MindCube_train.jsonl \
  --task full_pipeline
# Output: ./data/scaffold/all/MindCube_tinybench.jsonl
# Contains: cognitive_map + reasoning_chain + all original fields
```

#### 🔧 **Component-wise Generation**
```bash
# Generate cognitive maps only
python scripts/data_processing.py \
  --input data/raw/MindCube_tinybench.jsonl \
  --task cogmap

# Generate reasoning chains only
python scripts/data_processing.py \
  --input data/raw/MindCube_tinybench.jsonl \
  --task reasoning
```

---

### 💬 **Step 3: Prompt Generation** (`generate_prompts.py`)

#### ⭐ **Generate All Task Types (Recommended)**
```bash
# Generate all 8 prompt task types from scaffold data
python scripts/generate_prompts.py \
  --input data/scaffold/all/MindCube_tinybench.jsonl \
  --all_tasks
python scripts/generate_prompts.py \
  --input data/scaffold/all/MindCube_train.jsonl \
  --all_tasks

# Output: ./data/prompts/general/
# Generates: MindCube_tinybench_{task_name}.jsonl (8 files)
```

#### 🎯 **Single Task Type Generation**
```bash
# Generate specific task type
python scripts/generate_prompts.py \
  --input data/scaffold/all/MindCube_tinybench.jsonl \
  --task raw_qa \
  --output data/prompts/general/custom_output.jsonl

# List all available task types
python scripts/generate_prompts.py --list_tasks
```

---

### 🏋️ **Step 4: SFT Data Conversion** (`convert_to_sft.py`)

#### ⭐ **Batch Convert All Prompt Files**
```bash
# Convert all prompt files to Qwen2.5-VL SFT format
python scripts/convert_to_sft.py \
  --input_dir data/prompts/general/ \
  --model qwen2.5vl
# Output: ./data/prompts/training/qwen2.5vl/

# Convert to LLaVA format
python scripts/convert_to_sft.py \
  --input_dir data/prompts/general/ \
  --model llava
# Output: ./data/prompts/training/llava/

# List supported models
python scripts/convert_to_sft.py --list_models
```

#### 🎯 **Single File Conversion**
```bash
# Convert specific task to Qwen format
python scripts/convert_to_sft.py \
  --input data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --model qwen2.5vl \
  --output data/prompts/training/qwen2.5vl/custom_sft.json
```

---

### 🤖 **Step 5: Model Inference** (`run_inference.py`)

#### ⭐ **Basic Qwen2.5-VL Inference**
```bash
# Use default HuggingFace model
python scripts/run_inference.py \
  --model-type qwen2.5vl \
  --input-file data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-dir results/

# Output will be auto-generated as:
# results/MindCube_tinybench_raw_qa_qwen2.5-vl-3b-instruct_responses.jsonl
```

#### 🚀 **Accelerated Inference with vLLM**
```bash
# Use vLLM for faster inference
python scripts/run_inference.py \
  --model-type qwen2.5vl \
  --backend vllm \
  --input-file data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-dir results/
```

#### ⚙️ **Fine-tuned Model Inference**
```bash
# Use your own fine-tuned checkpoint
python scripts/run_inference.py \
  --model-type qwen2.5vl \
  --model-path /path/to/your/checkpoint \
  --input-file data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-file results/qwen_responses.jsonl
```

---

### 📊 **Step 6: Model Evaluation** (`run_evaluation.py`)

#### ⭐ **Comprehensive Evaluation**
```bash
# Evaluate model responses with multiple metrics
python scripts/run_evaluation.py \
  --predictions results/model_responses.jsonl \
  --ground_truth data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-dir evaluation_results/

# Specify evaluation metrics
python scripts/run_evaluation.py \
  --predictions results/model_responses.jsonl \
  --ground_truth data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --metrics accuracy f1_score cogmap_similarity \
  --output-dir evaluation_results/
```

---

## 🔄 **Complete Workflow Example**

```bash
# Step 1: Generate scaffold data (one-time setup)
python scripts/data_processing.py \
  --input data/raw/MindCube_tinybench.jsonl \
  --task full_pipeline

# Step 2: Generate all prompt types (one-time setup)
python scripts/generate_prompts.py \
  --input data/scaffold/all/MindCube_tinybench.jsonl \
  --all_tasks

# Step 3: Convert to SFT training format (optional)
python scripts/convert_to_sft.py \
  --input_dir data/prompts/general/ \
  --model qwen2.5vl

# Step 4: Run model inference
python scripts/run_inference.py \
  --model-type qwen2.5vl \
  --input-file data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-dir results/

# Step 5: Evaluate model performance
python scripts/run_evaluation.py \
  --predictions results/MindCube_tinybench_raw_qa_qwen2.5-vl-3b-instruct_responses.jsonl \
  --ground_truth data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-dir evaluation_results/
```

## 📋 **Available Task Types & Features**

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
- `qwen2.5vl` - Qwen2.5-VL conversation format with image placeholders
- `llava` - LLaVA training format with single image focus
- `instructblip` - InstructBLIP format with text input/output pairs

### Evaluation Metrics
- **Accuracy**: Basic answer correctness
- **F1 Score**: Precision and recall balance
- **Cognitive Map Similarity**: Semantic similarity of generated cognitive maps
- **Reasoning Quality**: Multi-step reasoning evaluation
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
  "meta_info": {"setting": "indoor"},
  "question": "Based on the images provided: ...",
  "images": ["image1.jpg", "image2.jpg"],
  "gt_answer": "A",
  
  // Added by scaffold generation
  "cogmap": "The scene shows spatial relationships between...",
  "reasoning_chain": "Step-by-step reasoning process..."
}
```

### Model Prompt Format (After Step 3)
```json
{
  "id": "unique_identifier",
  "category": "spatial",
  "type": "rotation",
  "meta_info": {"setting": "indoor"},
  "question": "Based on the images provided: ...",
  "images": ["image1.jpg", "image2.jpg"],
  "gt_answer": "A",
  
  // Model-ready fields
  "input_prompt": "[Answer Format]\nBased on these images, answer the question...",
  "grounded_output": "Step-by-step reasoning leading to the final answer..."
}
```

### SFT Training Format Examples

**Qwen2.5-VL Format:**
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

## 🛠️ **Advanced Usage**

### Batch Processing
```bash
# Process all files in a directory
python scripts/data_processing.py \
  --batch_dir input_dir/ \
  --output_dir output_dir/ \
  --task full_pipeline
```

### Validation and Preview
```bash
# Validate scaffold data
python scripts/generate_prompts.py \
  --input scaffold.jsonl \
  --validate

# Preview generated prompts
python scripts/generate_prompts.py \
  --input scaffold.jsonl \
  --preview \
  --samples 3
```

### Custom Configuration
```bash
# Use custom inference configuration
python scripts/run_inference.py \
  --model-type qwen2.5vl \
  --config configs/custom_inference.json \
  --input-file prompts.jsonl \
  --output-dir results/
```

### Help and Documentation
```bash
# Get help for any script
python scripts/data_processing.py --help
python scripts/generate_prompts.py --help
python scripts/run_inference.py --help
python scripts/run_evaluation.py --help

# List available options
python scripts/generate_prompts.py --list_tasks
python scripts/convert_to_sft.py --list_models
python scripts/run_inference.py --list-models
```

## 🔬 **Research & Development**

### Experiment Tracking
The framework supports structured experiment tracking through the `experiments/` directory:
- SFT training experiments in `experiments/sft/`
- Model checkpoints in `checkpoints/`
- Processing logs in `logs/`

### Jupyter Notebooks
Use the `notebooks/` directory for:
- Data analysis and visualization
- Model performance analysis
- Prototype development
- Custom metric development

### Extending the Framework
The modular architecture makes it easy to:
- Add new prompt task types in `src/prompt_generation/templates.py`
- Implement new evaluation metrics in `src/evaluation/metrics/`
- Support new model formats in SFT conversion
- Add custom processing steps in the scaffold curation pipeline

## 🤝 **Contributing**

To contribute to MindCube:

1. Fork the repository
2. Create a feature branch
3. Follow the existing code structure and English comment convention
4. Add tests for new functionality
5. Submit a pull request

## 📝 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 **Contact**

For questions, issues, or collaboration opportunities, please:
- Open an issue on GitHub
- Contact the development team
- Join our research discussions

---

**MindCube Framework** - Advancing spatial reasoning capabilities in vision-language models through comprehensive dataset generation and evaluation.

# MindCube vLLM加速推理

专注于使用vLLM加速Qwen2.5-VL多图推理，支持最多4张图片输入，输出限制1536 tokens。

## 🚀 快速开始

### 1. 环境配置

```bash
# 1. 安装vLLM (需要CUDA 12.1+)
pip install vllm>=0.7.2

# 2. 安装Qwen2.5-VL依赖
pip install transformers>=4.51.3 accelerate qwen-vl-utils[decord]

# 3. 验证环境
python test_vllm_specific.py
```

### 2. vLLM推理配置

当前配置：`configs/vllm_optimized.json`

```json
{
  "model_type": "qwen2.5vl",
  "backend": "vllm",
  "generation_config": {
    "max_new_tokens": 1536,
    "temperature": 0.0
  },
  "vllm_config": {
    "gpu_memory_utilization": 0.95,
    "max_model_len": 32768,
    "limit_mm_per_prompt": {
      "image": 4,
      "video": 1
    },
    "trust_remote_code": true,
    "dtype": "bfloat16",
    "enable_prefix_caching": true,
    "enable_chunked_prefill": true,
    "max_num_seqs": 32,
    "max_num_batched_tokens": 4096
  }
}
```

### 3. 使用方法

#### 单次推理
```bash
python scripts/run_inference.py --backend vllm --config configs/vllm_optimized.json
```

#### 批量推理
```bash
python scripts/run_inference.py \
  --backend vllm \
  --batch-size 16 \
  --config configs/vllm_optimized.json \
  --input-file data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-dir data/outputs/
```

#### 多图推理示例

```python
from vllm import LLM, SamplingParams
from PIL import Image

# 初始化vLLM
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    gpu_memory_utilization=0.95,
    max_model_len=32768,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 4}
)

# 采样参数
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1536
)

# 多图推理
images = [Image.open(f"image_{i}.jpg") for i in range(4)]
prompt = "请分析这4张图片的共同特征。"

outputs = llm.generate(
    {
        "prompt": prompt,
        "multi_modal_data": {"image": images}
    },
    sampling_params=sampling_params
)

print(outputs[0].outputs[0].text)
```

## 🎯 关键特性

- ✅ **vLLM加速**：使用vLLM引擎获得最佳推理性能
- 🖼️ **多图支持**：支持最多4张图片同时输入
- 🚀 **高吞吐量**：优化配置支持大批量并发推理
- 💾 **智能缓存**：启用前缀缓存和分块预填充
- ⚡ **95% GPU利用率**：不关心内存使用，专注最大性能

## 📊 性能优化

### GPU配置
- **内存使用**：95% GPU内存利用率
- **序列长度**：支持32K上下文长度
- **批量大小**：推荐16-32，根据GPU内存调整
- **精度**：bfloat16用于最佳性能

### 多模态配置
- **图片限制**：最多4张图片/prompt
- **输出限制**：1536 tokens最大输出
- **视频支持**：支持但限制1个视频/prompt

## 🔧 故障排除

### 常见问题

1. **GPU内存不足**
   ```bash
   # 降低内存使用
   "gpu_memory_utilization": 0.8
   ```

2. **vLLM版本错误**
   ```bash
   pip install vllm>=0.7.2
   ```

3. **模型加载失败**
   ```bash
   # 确保trust_remote_code=true
   "trust_remote_code": true
   ```

### 性能调优

1. **增加批量大小**：
   - 修改`max_num_seqs`和`batch_size`

2. **优化内存使用**：
   - 调整`max_model_len`和`max_num_batched_tokens`

3. **多GPU支持**：
   - 设置`tensor_parallel_size`

## 📝 项目结构

```
07_MindCube_new/
├── configs/
│   └── vllm_optimized.json     # vLLM优化配置
├── test_vllm_specific.py       # vLLM测试脚本
├── scripts/
│   └── run_inference.py        # 主推理脚本
└── src/
    └── inference/               # 推理引擎代码
```

## 🎨 使用案例

### 多图比较分析
```python
# 比较多张图片的异同
prompt = "请比较这几张图片的异同点，并给出详细分析。"
images = [img1, img2, img3, img4]
```

### 批量文档理解
```python
# 批量处理文档图片
prompts = ["总结这份文档的要点。" for _ in range(batch_size)]
image_batches = [doc_images for doc_images in document_batches]
```

### 视觉问答
```python
# 基于图片内容回答问题
prompt = "图片中有什么物体？它们的位置关系如何？"
```

## ⚠️ 注意事项

1. **仅支持vLLM后端**：本配置专门为vLLM优化
2. **高内存使用**：95% GPU内存利用率需要足够的显存
3. **输出限制**：严格限制1536 tokens输出
4. **图片数量**：最多4张图片/prompt

---

🚀 **立即开始vLLM加速推理！** 