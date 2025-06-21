# MindCube Evaluation Framework

A minimal-effort reorganized evaluation system for cognitive mapping and spatial reasoning tasks.

## 🎯 Overview

This framework provides **two evaluation modes** with a **unified interface**:

- **Basic Mode**: Answer accuracy evaluation only (fast, lightweight)
- **Cognitive Map Mode**: Full spatial reasoning analysis with similarity metrics (comprehensive)

## 🚀 Quick Start

### Installation
```bash
# Navigate to the evaluation directory
cd src/evaluation

# The framework uses standard Python libraries - no additional installation needed
```

### Basic Usage

```python
from src.evaluation import evaluate, auto_evaluate

# Method 1: Basic evaluation (answer accuracy only)
results = evaluate("responses.jsonl", "basic")

# Method 2: Full cognitive map evaluation
results = evaluate("responses.jsonl", "cogmap")

# Method 3: Auto-detect content type (recommended)
results = auto_evaluate("responses.jsonl")
```

### Command Line Usage

```bash
# Basic evaluation
python cli.py --input responses.jsonl --task basic

# Cognitive map evaluation  
python cli.py --input responses.jsonl --task cogmap

# Auto-detect task type
python cli.py --input responses.jsonl --auto

# Batch evaluation
python cli.py --batch_dir results/ --output_dir analysis/

# Quick guide
python cli.py --guide
```

## 📊 Evaluation Modes

### 1. Basic Mode (`task="basic"`)
- **Purpose**: Answer accuracy evaluation only
- **Speed**: Fast (~1-2 seconds for 100 examples)
- **Use case**: Quick performance checks, basic task evaluation
- **Metrics**: Answer accuracy by setting, error analysis

```python
results = evaluate("responses.jsonl", "basic")
print(f"Accuracy: {results['results']['gen_cogmap_accuracy']*100:.1f}%")
```

### 2. Cognitive Map Mode (`task="cogmap"`)
- **Purpose**: Comprehensive spatial reasoning analysis
- **Speed**: Slower (~10-30 seconds for 100 examples)
- **Use case**: Research analysis, detailed cognitive mapping evaluation
- **Metrics**: Spatial similarity, rotation-invariant isomorphism, graph analysis

```python
results = evaluate("responses.jsonl", "cogmap")
cogmap_metrics = results['results']['cogmap_similarity']
print(f"Valid graphs: {cogmap_metrics['valid_percent']:.1f}%")
print(f"Spatial similarity: {cogmap_metrics['avg_overall_similarity']:.3f}")
```

### 3. Auto-Detection (`auto_evaluate`)
- **Purpose**: Automatically choose evaluation mode based on content
- **Logic**: Checks for cognitive map data in responses
- **Recommended**: For batch processing and general use

```python
results = auto_evaluate("responses.jsonl")
# Automatically uses cogmap mode if JSON/cognitive maps detected
# Falls back to basic mode otherwise
```

## 📁 Output Structure

### Basic Results
```python
{
  'results': {
    'total': 100,
    'gen_cogmap_correct': 75,
    'gen_cogmap_accuracy': 0.75,
    'settings': {
      'around': {'total': 25, 'gen_cogmap_correct': 20, 'accuracy': 0.80},
      'rotation': {'total': 25, 'gen_cogmap_correct': 18, 'accuracy': 0.72},
      # ... more settings
    }
  },
  'error_cases': [...] # Failed answer extractions for debugging
}
```

### Cognitive Map Results (Additional)
```python
{
  'results': {
    # ... basic results above, plus:
    'cogmap_similarity': {
      'total_valid': 80,
      'valid_percent': 80.0,
      'parsable_json_count': 85,
      'valid_format_count': 82,
      'rotation_invariant_isomorphic_count': 45,
      'avg_directional_similarity': 0.734,
      'avg_facing_similarity': 0.698,
      'avg_overall_similarity': 0.721
    }
  }
}
```

## 🔧 Advanced Usage

### Custom Evaluation Settings

```python
from src.evaluation import CogMapEvaluator

# Quick cognitive map check (no detailed metrics)
evaluator = CogMapEvaluator(include_detailed_metrics=False)
results = evaluator.evaluate("responses.jsonl")

# Custom basic evaluator
from src.evaluation import BasicEvaluator
evaluator = BasicEvaluator()
results = evaluator.evaluate("responses.jsonl")
```

### Batch Processing

```python
from src.evaluation import batch_evaluate
import glob

# Process all files in a directory
batch_evaluate("results/", "analysis/")

# Custom batch processing
for file in glob.glob("results/*.jsonl"):
    results = auto_evaluate(file, f"{file}.results.json")
    print(f"{file}: {results['results']['gen_cogmap_accuracy']*100:.1f}%")
```

### Integration with Existing Scripts

Replace your current evaluation calls:

```python
# OLD: Multiple different evaluation scripts
# python base_sft/eval/eval_main.py --input file.jsonl
# python cog_map/vlm_gen_map/eval/eval_cogmap_main.py --input file.jsonl

# NEW: Single interface
from src.evaluation import auto_evaluate
results = auto_evaluate("file.jsonl", "results.json")
```

## 📂 Architecture

```
src/evaluation/
├── core/                    # Shared utilities
│   ├── extractors.py        # Answer/JSON extraction
│   ├── base_metrics.py      # Basic accuracy metrics
│   └── io_utils.py          # I/O operations
├── cogmap/                  # Cognitive map specific
│   ├── cogmap_metrics.py    # Spatial similarity metrics
│   ├── graph_operations.py  # Graph analysis (copied from original)
│   └── cogmap_evaluator.py  # Cogmap evaluation orchestrator
├── evaluator.py             # Main evaluation interface
├── cli.py                   # Command-line interface
└── __init__.py              # Public API
```

## 🔄 Migration from Original Code

### For Basic Tasks
```python
# OLD
python base_sft/eval/eval_main.py --input responses.jsonl --output results.json

# NEW
python src/evaluation/cli.py --input responses.jsonl --task basic --output results.json
# OR
from src.evaluation import evaluate
results = evaluate("responses.jsonl", "basic", "results.json")
```

### For Cognitive Map Tasks
```python
# OLD
python cog_map/vlm_gen_map/eval/eval_cogmap_main.py --input responses.jsonl --output results.json

# NEW  
python src/evaluation/cli.py --input responses.jsonl --task cogmap --output results.json
# OR
from src.evaluation import evaluate
results = evaluate("responses.jsonl", "cogmap", "results.json")
```

### For Batch Evaluation
```python
# OLD: Manual script management

# NEW
python src/evaluation/cli.py --batch_dir results/ --output_dir analysis/
# OR
from src.evaluation import batch_evaluate
batch_evaluate("results/", "analysis/")
```

## ⚡ Performance Notes

- **Basic mode**: ~0.01-0.02 seconds per example
- **Cognitive map mode**: ~0.1-0.3 seconds per example  
- **Memory usage**: Minimal for basic mode, moderate for cognitive map mode
- **Parallelization**: Can be easily parallelized for batch processing

## 🐛 Debugging

### Common Issues

1. **Import errors**: Make sure you're in the correct directory
   ```bash
   cd src/evaluation
   python -c "from evaluation import evaluate; print('✅ Import successful')"
   ```

2. **File not found**: Check file paths are correct
   ```python
   import os
   print(os.path.exists("your_file.jsonl"))
   ```

3. **No answers extracted**: Check the `error_cases` in results
   ```python
   results = evaluate("file.jsonl", "basic")
   print(f"Extraction errors: {len(results['error_cases']['gen_cogmap_error'])}")
   ```

### Verbose Output
```python
# Enable detailed error reporting
import logging
logging.basicConfig(level=logging.DEBUG)

results = evaluate("responses.jsonl", "cogmap")
```

## 📋 Requirements

- Python 3.7+
- Standard libraries: `json`, `re`, `typing`, `collections`, `glob`, `os`
- For cognitive map mode: `numpy` (already in your requirements.txt)

## 🤝 Contributing

The framework is designed for easy extension:

1. **Adding new metrics**: Extend `cogmap_metrics.py`
2. **New evaluation modes**: Add to `evaluator.py`  
3. **Custom extractors**: Extend `core/extractors.py`

## 📄 License

Same as the main MindCube project. 