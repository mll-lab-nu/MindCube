<h1 align="center">MindCube: Spatial Mental Modeling from Limited Views</h1>
<!-- Badges -->
<div align="center">

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2412.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2412.XXXXX) -->
[![Homepage](https://img.shields.io/badge/🏠-Homepage-blue.svg)](https://mind-cube.github.io/)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/Inevitablevalor/MindCube)
[![Checkpoints](https://img.shields.io/badge/🤗-Checkpoints-green.svg)](https://huggingface.co/Inevitablevalor/models)

</div>
<p align="center">
    <a href="https://limanling.github.io/">Baiqiao Yin<sup>1, 4*</sup></a>, 
    <a href="https://www.linkedin.com/in/shiyu-zhao-1124a0266/">Qineng Wang<sup>1*</sup></a>, 
    <a href="https://qinengwang-aiden.github.io/">Pingyue Zhang<sup>1</sup></a>, 
    <a href="https://jameskrw.github.io/">Jianshu Zhang<sup>1</sup></a>, 
    <a href="https://bryanzhou008.github.io/">Kangrui Wang<sup>1</sup></a>, 
    <a href="https://example.com/sanjana-srivastava">Zihan Wang<sup>1</sup></a>, 
    <a href="https://example.com/cem-gokmen">Jieyu Zhang<sup>3</sup></a>, 
    <a href="https://example.com/tony-lee">Keshigeyan Chandrasegaran<sup>2</sup></a>, 
    <a href="https://sites.google.com/site/lieranli/">Han Liu<sup>1</sup></a>, 
    <a href="https://example.com/ruohan-zhang">Ranjay Krishna<sup>3</sup></a>, 
    <a href="https://example.com/weiyu-liu">Saining Xie<sup>4</sup></a>, 
    <a href="https://profiles.stanford.edu/fei-fei-li">Manling Li†<sup>1</sup></a>, 
    <a href="https://jiayuanm.com/">Jiajun Wu<sup>2†</sup></a>, 
    <a href="https://jiajunwu.com/">Li Fei-Fei<sup>2†</sup></a>
</p>
<p align="center">*Equal contribution, †Equal advising</p>
<p align="center"><sup>1</sup>Northwestern University, <sup>2</sup>Stanford University, <sup>3</sup>University of Washington, <sup>4</sup>New York University</p>


## 📢 Updates

- **[2025-06-23]** We open-source the MindCube framework and dataset.

## 🌟 Overview

MindCube is a modular framework for generating and evaluating spatial reasoning datasets for multimodal AI models. The project follows a complete pipeline from raw data to model evaluation, with specialized modules for scaffold data curation, prompt generation, model inference, training, and comprehensive evaluation.

## ⚙️ Environment Setup

Follow these steps to set up your development environment. This process will create an isolated Python environment with all necessary dependencies for running MindCube.

```bash
git clone git@github.com:QinengWang-Aiden/MindCube.git
cd MindCube
```

First, we'll create a dedicated conda environment to avoid conflicts with other projects:

```bash
conda create -n mindcube python=3.10 -y
conda activate mindcube
```

Next, install PyTorch with CUDA support. Make sure to adjust the CUDA version according to your system:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 # change to your cuda version
```

Finally, install the attention mechanism and other required dependencies:

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -r requirements.txt
```

## 📥 Download `MindCube` Dataset

Once your environment is ready, download the MindCube dataset which contains the spatial reasoning questions and images:

```bash
bash scripts/bash_scripts/download_data.bash
```

---

## 🚀 Quick Start

### 📋 Eval Data Generation

The data generation process transforms raw spatial reasoning data into structured formats suitable for model training and evaluation.

#### Approach 1: One Command Line Generation for All Data

For convenience, use this single command to generate all required data formats:

```bash 
bash scripts/bash_scripts/generate_eval_data.bash
```

#### Approach 2: Detailed Steps

If you prefer to understand each step or need fine-grained control, follow these detailed steps:

**Step 1: Scaffold Data Generation**

This step processes raw JSONL files and generates cognitive maps and reasoning chains that serve as scaffolds for spatial understanding:

```bash
python scripts/data_processing.py \
  --input data/raw/MindCube_train.jsonl \
  --task full_pipeline
python scripts/data_processing.py \
  --input data/raw/MindCube_tinybench.jsonl \
  --task full_pipeline
```

**Step 2: General Prompts Generation**

Now we create various prompt formats (8 different task types) that will be used for model training and evaluation:

```bash
python scripts/generate_prompts.py \
  --input data/scaffold/all/MindCube_train.jsonl \
  --all_tasks
python scripts/generate_prompts.py \
  --input data/scaffold/all/MindCube_tinybench.jsonl \
  --all_tasks
```

**Step 3: Model Format Data Transformation**

Finally, convert the general prompts into model-specific formats. Currently, we support Qwen2.5VL format:

```bash
python scripts/convert_to_sft.py \
  --input_dir data/prompts/general/ \
  --model qwen2.5vl # Currently, we only support Qwen2.5VL Format
```

#### Expected Output Directory

After completing these steps, you should see the following directory structure:

`data/scaffold/all`: 2 files

`data/prompts/general`: 16 files

`data/prompts/training/qwen2.5vl`: 16 files

### 🧊 Frozen VLM Inference

With your data prepared, you can now run inference using pre-trained vision-language models without any fine-tuning.

#### Approach 1: Batch Inference (All 6 Task Configurations)

Run inference on all task configurations simultaneously for comprehensive evaluation:

```bash
bash scripts/bash_scripts/run_frozen_vlm_all_tasks_qwen.sh --max-tasks-per-gpu 2 # You can adjust this number based on your GPU's capacity (Default is 1)
```

#### Approach 2: Run Inference Individually

For more control or when working with limited GPU memory, run inference on specific tasks:

```bash
python scripts/run_inference.py \
  --model-type qwen2.5vl \
  --input-file data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-dir data/results/frozen_vlm
  # you can adjust input-file and output-dir here
```

#### Expected Output Directory

The inference results will be saved in structured directories for easy analysis:

`data/results/frozen_vlm/`: n jsonl files (based on your command)

`logs/inference`: All frozen inference logs

### 📊 Evaluation

After obtaining model predictions, evaluate the performance using our comprehensive evaluation metrics.

#### Approach 1: Batch Evaluation

Evaluate all inference results at once for a complete performance overview:

```bash
bash scripts/bash_scripts/run_batch_evaluation.sh data/results/frozen_vlm/ # You can adjust the path to the jsonl files you would like to evaluate
```

#### Approach 2: Run Evaluation Individually

For detailed analysis of specific models or tasks, run evaluation individually:

```bash
python scripts/run_evaluation.py \
  -i data/results/frozen_vlm/MindCube_tinybench_raw_qa_qwen2.5-vl-3b-instruct_responses.jsonl
  -o data/evaluate/frozen_vlm/MindCube_tinybench_raw_qa_qwen2.5-vl-3b-instruct_responses_eval_results.json
  # you can adjust the input and output here
```

#### Expected Output Directory

Evaluation results will be organized for easy interpretation and comparison:

`data/evaluate/frozen_vlm`: n json files (based on your command)

---

## 🏋️ SFT Training (from `Qwen2.5VL-3B-Instruct`)

This section guides you through supervised fine-tuning (SFT) to adapt pre-trained models specifically for spatial reasoning tasks.

### (Optional) Step 0: Environment Setup

If you plan to process video data or need multimedia support, install ffmpeg:

```bash
conda install -c conda-forge ffmpeg -y # skip this if you already installed ffmpeg in your device
```

### Step 1: Clone Qwen Repo

We need the specialized Qwen2.5-VL repository that contains our custom modifications for MindCube training:

```bash
git clone git@github.com:QinengWang-Aiden/Qwen2.5-VL-MindCube.git
```

### Step 2: Add Training Patches into `Qwen2.5-VL-MindCube/qwen-vl-finetune/qwenvl/data/__init__.py`

This step integrates MindCube datasets into the Qwen training pipeline.

#### 2.1 Verify the Patching status

First, let's check if the MindCube datasets are properly registered in the training system:

```bash
python experiments/sft/patch_qwen_data.py verify
```

**Expected Output**

```bash
Project root: /path/to/your/MindCube
Target file: /path/to/your/MindCube/Qwen2.5-VL-MindCube/qwen-vl-finetune/qwenvl/data/__init__.py
Command: verify

Found 0/6 MindCube datasets in data_dict:
Missing datasets:
  ❌ raw_qa
  ❌ plain_cgmap_out
  ❌ ff_rsn
  ❌ aug_cgmap_out
  ❌ aug_cgmap_ffr_out
  ❌ plain_cgmap_ffr_out
```

#### 2.2 Patch the `__init__.py`

Now apply the patches to enable MindCube dataset support in the training pipeline:

```bash
python experiments/sft/patch_qwen_data.py patch
```

**Expected Output**

```bash
✅ Successfully patched Qwen __init__.py with MindCube datasets
Found 6/6 MindCube datasets in data_dict:
  ✅ raw_qa
  ✅ aug_cgmap_out
  ✅ plain_cgmap_out
  ✅ ff_rsn
  ✅ aug_cgmap_ffr_out
  ✅ plain_cgmap_ffr_out
```

### (Optional) Step 3: Customize Your Configuration File

Before starting training, you may want to adjust the configuration based on your hardware and training preferences.

#### GPU Env Setup: `experiments/sft/config_hardware.sh`

Configure your GPU settings according to your available hardware:

```bash
# Hardware configuration
GPU_DEVICES="0"          # Modify based on available GPUs (e.g., "0,1,2,3" for 4 GPUs)
NUM_PROCESSES=1          # Should match number of GPUs
BATCH_SIZE=1             # Per-device batch size (adjust based on GPU memory)
```

#### Customize Your Task Hyperparameters (`config_raw_qa.sh` as Example)

Adjust training hyperparameters for optimal performance on your specific task:

```bash
# Training hyperparameters
LEARNING_RATE=1e-5
NUM_EPOCHS=3

# Output configuration
OUTPUT_BASE_DIR="experiments/sft/results"
RUN_NAME="qwen2vl-${TASK_NAME}_sft"

# Additional training arguments
MAX_PIXELS=90000
MIN_PIXELS=784
MODEL_MAX_LENGTH=8192
SAVE_STEPS=5
SAVE_TOTAL_LIMIT=12
```

### Step 4: SFT Training

Now we're ready to start the actual training process. The model will learn to better understand spatial relationships through supervised fine-tuning.

#### Approach 1: Batch Training All Task Configurations (Run in a Sequence)

Train on all task types sequentially for comprehensive spatial reasoning capabilities:

```bash
bash scripts/bash_scripts/run_sft_all_tasks_qwen.sh
```

#### Approach 2: Run SFT Individually

For focused training or resource constraints, train on specific tasks:

```bash
bash experiments/sft/train_qwen_sft.sh config_raw_qa.sh # or you can replace with any legal task config here
```

#### Expected Output Directories

Training artifacts will be organized for easy access and model deployment:

`checkpoints/sft/`: List of all tasks saved checkpoints

`logs/sft_training/`: All training logs

### Step 5: SFT Checkpoints Inference

After training, test your fine-tuned models on the evaluation datasets to measure improvement.

#### Approach 1: Batch Inference

Run inference using all trained checkpoints for comprehensive evaluation:

```bash
bash scripts/bash_scripts/run_sft_ckpt_inference_qwen.sh --max-tasks-per-gpu 2 # You can adjust this number based on your GPU's capacity (Default is 1)
```

#### Approach 2: Run Inference Individually

Test specific checkpoints for detailed analysis:

```bash
python scripts/run_inference.py \
  --model-type qwen2.5vl \
  --model-path checkpoints/sft/raw_qa/checkpoint-5
  --input-file data/prompts/general/MindCube_tinybench_raw_qa.jsonl \
  --output-dir data/results/sft/raw_qa
  # you can adjust input-file and output-dir here
```

#### Expected Output

Fine-tuned model results will be organized by task for easy comparison with baseline models:

`data/results/sft/`: a list of task name directories

`data/results/sft/<task_name>`: a list of jsonl files of inference results

### Step 6: Evaluation

Finally, evaluate your fine-tuned models to quantify the improvement in spatial reasoning capabilities.

#### Approach 1: Batch Evaluation

Evaluate all fine-tuned model results comprehensively:

```bash
bash scripts/bash_scripts/run_batch_evaluation.sh data/results/sft/ # You can adjust the path to the jsonl files you would like to evaluate
```

#### Approach 2: Run Evaluation Individually

For detailed analysis of specific fine-tuned models:

```bash
python scripts/run_evaluation.py \
  -i data/results/sft/raw_qa/MindCube_tinybench_raw_qa_checkpoint-5_responses.jsonl
  -o data/evaluate/sft/raw_qa/MindCube_tinybench_raw_qa_checkpoint-5_responses_eval_results.json
  # you can adjust the input and output here
```

#### Expected Output Directory

Evaluation results will show the effectiveness of your fine-tuning approach:

`data/evaluate/sft`: a list of task name directories

`data/results/sft/<task_name>`: a list of json files as evaluation results

---

## 🔄 Processing Pipeline Overview

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

## 🛠️ Command Help

Get help for any script:
```bash
python scripts/data_processing.py --help
python scripts/generate_prompts.py --help
python scripts/run_inference.py --help
python scripts/run_evaluation.py --help
```

---

## 🗒️ Checklist

- [ ] Add RL Training Description
- [ ] Release RL Training Checkpoints

## 🔗 Other MLL-Lab Projects

Explore other exciting projects from our **MLL-Lab**:

- **[EAI](https://embodied-agent-interface.github.io/)** 
- **[RAGEN](https://ragen-ai.github.io/)**
- **[VAGEN](https://github.com/RAGEN-AI/VAGEN)**

## 📝 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
