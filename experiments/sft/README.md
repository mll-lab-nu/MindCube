# Qwen2.5-VL SFT Training Scripts

This directory contains independent training scripts for Qwen2.5-VL fine-tuning across different MindCube tasks.

## Files Overview

- `setup_environment.sh`: One-time setup script to patch Qwen's data configuration
- `train_qwen_sft.sh`: Main training script that can use configuration files
- `patch_qwen_data.py`: Script to patch/restore Qwen's data configuration with MindCube datasets
- `config_template.sh`: Template for creating task-specific configurations
- Configuration files for specific tasks:
  - `config_raw_qa.sh`: Raw question-answering task
  - `config_aug_cgmap_out.sh`: Augmented cognitive map output task
  - `config_ff_rsn.sh`: Free-form reasoning task
  - `config_cog_reasoning.sh`: Cognitive reasoning task (legacy)

## Setup Instructions

**⚠️ IMPORTANT: You must run the setup script once before training!**

### Step 1: Clone Qwen Repository (if not already done)
```bash
cd /path/to/your/07_MindCube_new/
git clone https://github.com/QwenLM/Qwen2.5-VL.git
```

### Step 2: Setup Environment
```bash
cd /path/to/your/07_MindCube_new/experiments/sft
./setup_environment.sh
```

This will:
- Verify Qwen2.5-VL installation
- Patch Qwen's data configuration with MindCube datasets
- Create a backup of the original configuration

### Step 3: Run Training

#### Method 1: Using Default Configuration
```bash
cd /path/to/your/07_MindCube_new/experiments/sft
./train_qwen_sft.sh
```

#### Method 2: Using Configuration File
```bash
cd /path/to/your/07_MindCube_new/experiments/sft
./train_qwen_sft.sh config_raw_qa.sh
# or
./train_qwen_sft.sh config_aug_cgmap_out.sh
# or
./train_qwen_sft.sh config_ff_rsn.sh
```

## Creating New Task Configurations

1. Copy the template:
```bash
cp config_template.sh config_your_task.sh
```

2. Edit the configuration file:
```bash
nano config_your_task.sh
```

3. Modify the following key parameters:
   - `TASK_NAME`: Name of your task
   - `DATASET_NAME`: Dataset identifier used in the training code
   - `LEARNING_RATE`: Learning rate for your task
   - `BATCH_SIZE`: Batch size per GPU
   - `NUM_EPOCHS`: Number of training epochs

4. Run training with your configuration:
```bash
./train_qwen_sft.sh config_your_task.sh
```

## MindCube Dataset Integration

This system integrates MindCube datasets by patching Qwen's data configuration:

1. **`patch_qwen_data.py`** directly modifies Qwen's `__init__.py` to add MindCube datasets
2. **`setup_environment.sh`** provides a user-friendly interface for the one-time setup
3. **Available MindCube datasets**:
   - `raw_qa`: Raw question-answering
   - `aug_cgmap_out`: Augmented cognitive map output
   - `plain_cgmap_out`: Plain cognitive map output
   - `ff_rsn`: Free-form reasoning
   - `aug_cgmap_ffr_out`: Augmented cognitive map + free-form reasoning output

### Why Patch Instead of Runtime Injection?
- **Reliability**: Direct file modification is more reliable than runtime injection
- **Transparency**: Users can see exactly what datasets are available
- **Simplicity**: No need for complex Python path management

## Key Configuration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `TASK_NAME` | Identifier for your task | `"cog_reasoning"` |
| `DATASET_NAME` | Dataset name used in training | `"cog_reasoning_sft"` |
| `MODEL_NAME` | HuggingFace model identifier | `"Qwen/Qwen2.5-VL-3B-Instruct"` |
| `LEARNING_RATE` | Training learning rate | `1e-5` |
| `BATCH_SIZE` | Batch size per GPU | `4` |
| `GRAD_ACCUM_STEPS` | Gradient accumulation steps | `32` |
| `NUM_EPOCHS` | Number of training epochs | `3` |
| `GPU_DEVICES` | GPU devices to use | `"0,1,2,3"` |
| `NUM_PROCESSES` | Number of processes (should match GPU count) | `4` |

## Output Directory Structure

Training outputs will be saved to:
```
experiments/sft/results/
├── cog_reasoning/          # Task-specific results
│   ├── checkpoint-5/
│   ├── checkpoint-10/
│   └── ...
├── your_task_name/         # Your custom task results
│   ├── checkpoint-5/
│   └── ...
```

## Advanced Usage

### Environment Variables

You can override distributed training settings:
```bash
export MASTER_ADDR="your_master_node"
export MASTER_PORT="29500"
export WORLD_SIZE="1"
./train_qwen_sft.sh config_your_task.sh
```

### Custom GPU Configuration

Modify GPU settings in your config file:
```bash
GPU_DEVICES="0,1"  # Use only 2 GPUs
NUM_PROCESSES=2    # Match the number of GPUs
```

## Troubleshooting

1. **Permission denied**: Make sure the script is executable:
   ```bash
   chmod +x train_qwen_sft.sh
   ```

2. **Path not found**: Ensure you're running from the correct directory:
   ```bash
   cd /projects/b1222/userdata/qineng/01_projects/07_MindCube_new/experiments/sft
   ```

3. **GPU memory issues**: Reduce batch size or enable gradient checkpointing in your config

4. **Dataset not found**: Verify the `DATASET_NAME` matches what's available in your training code

## Manual Operations

### Restoring Original Qwen Configuration (if needed)
If you want to restore the original Qwen data configuration:
```bash
cd /path/to/your/07_MindCube_new/experiments/sft
python patch_qwen_data.py restore
```

### Checking Current Status
To verify if MindCube datasets are currently available:
```bash
cd /path/to/your/07_MindCube_new/experiments/sft
python patch_qwen_data.py verify
```

### Re-applying Patch
If you need to re-apply the patch (e.g., after updating Qwen):
```bash
cd /path/to/your/07_MindCube_new/experiments/sft
python patch_qwen_data.py patch
```

## Adding New Datasets

To add support for new datasets:
1. Add the dataset definition to `patch_qwen_data.py`
2. Update the `MINDCUBE_DATA_DICT_ENTRIES` template
3. Re-run the patch: `python patch_qwen_data.py patch`
4. Create a corresponding configuration file with the appropriate `DATASET_NAME` 