# MindCube Multi-Seed & Self-Consistency Evaluation System

## 🎯 Overview

This system provides comprehensive evaluation for MindCube with:
1. **Multi-seed inference** for statistical robustness  
2. **Self-consistency analysis** via majority voting
3. **Meta evaluation** with p-values and confidence intervals
4. **Flexible directory support** for different experiments (frozen vs SFT)

## 📁 File Structure

```
tmp/
├── run_frozen_all_seeds.bash       # Multi-seed inference runner
├── self_consistency.py             # Self-consistency analysis
├── run_self_consistency.bash       # Self-consistency batch runner  
├── meta_evaluation.py              # Meta analysis with statistics
├── run_meta_evaluation.bash        # Complete evaluation pipeline
└── usage_summary.md                # This file
```

## 🚀 Quick Start

### Step 1: Run Multi-Seed Inference
```bash
# For frozen model results
bash tmp/run_frozen_all_seeds.bash

# Results saved to: tmp_results/frozen_multi_seeds/
```

### Step 2: Run Self-Consistency Analysis
```bash
# Analyze frozen results with majority voting
bash tmp/run_self_consistency.bash

# Results saved to: tmp_results/self_consistency/
```

### Step 3: Run Meta Evaluation
```bash
# Complete pipeline for frozen results
bash tmp/run_meta_evaluation.bash

# For SFT results comparison:
bash tmp/run_meta_evaluation.bash --multi-seed-suffix sft_multi_seeds

# Results saved to: tmp_results/meta_analysis/
```

## 📊 New Features Added

### 1. **Statistical Analysis**
- **P-values**: One-sample t-test for multi-seed variance
- **Confidence intervals**: 95% CI for all metrics
- **Paired t-tests**: Comparing consensus vs individual seeds
- **Effect sizes**: Cohen's d for practical significance

### 2. **Flexible Directory Support**
```bash
# Use frozen results (default)
--multi-seed-suffix frozen_multi_seeds

# Use SFT results  
--multi-seed-suffix sft_multi_seeds

# Use custom directory
--multi-seed-dir /path/to/custom/results
```

### 3. **Comprehensive Metrics**
- **Overall accuracy**: Main performance metric
- **Setting-specific**: rotation, among, around accuracy
- **Graph metrics**: valid_rate, isomorphic_rate, overall_similarity
- **Statistical significance**: p < 0.01, 0.05, 0.10 flags

## 🔬 Output Format

### Meta Analysis JSON Structure
```json
{
  "metadata": {
    "tasks": ["raw_qa", "aug_cgmap_in", ...],
    "seeds": [42, 123, 456],
    "consensus_strategy": "majority",
    "multi_seed_suffix": "frozen_multi_seeds",
    "scipy_available": false
  },
  "multi_seed_results": {
    "raw_qa": {
      "individual_seeds": {
        "seed_42": {"overall_accuracy": 0.75, ...},
        "seed_123": {"overall_accuracy": 0.73, ...},
        "seed_456": {"overall_accuracy": 0.77, ...}
      },
      "statistics": {
        "overall_accuracy_mean": 0.75,
        "overall_accuracy_std": 0.02,
        "overall_accuracy_p_value": 0.001,
        "overall_accuracy_ci_95_lower": 0.70,
        "overall_accuracy_ci_95_upper": 0.80,
        "overall_accuracy_significant_p05": true
      }
    }
  },
  "consensus_results": {
    "raw_qa": {
      "metrics": {
        "overall_accuracy": 0.78,
        "consensus_confidence_mean": 0.89,
        "perfect_agreement_rate": 0.65
      }
    }
  },
  "comparison": {
    "raw_qa": {
      "overall_accuracy": {
        "multi_seed_mean": 0.75,
        "consensus_value": 0.78,
        "improvement": 0.03,
        "improvement_percentage": 4.0,
        "paired_p_value": 0.02,
        "effect_size": 1.5
      }
    }
  }
}
```

## 📈 Key Metrics Explained

### Multi-Seed Statistics
- `{metric}_mean`: Average across seeds
- `{metric}_std`: Standard deviation  
- `{metric}_p_value`: Significance test vs 0
- `{metric}_ci_95_lower/upper`: 95% confidence interval
- `{metric}_significant_p05`: p < 0.05 flag

### Consensus vs Seeds Comparison  
- `improvement`: Consensus - Multi-seed mean
- `improvement_percentage`: Relative improvement %
- `paired_p_value`: Paired t-test significance
- `effect_size`: Cohen's d (improvement/std)

### Self-Consistency Quality
- `consensus_confidence_mean`: Average voting confidence
- `perfect_agreement_rate`: % with 100% seed agreement  
- `high_confidence_rate`: % with ≥2/3 seed agreement

## 🎯 For Rebuttal

### Statistical Robustness
```python
# Extract significance results
meta_results = json.load(open('meta_analysis_majority.json'))
for task, stats in meta_results['multi_seed_results'].items():
    p_val = stats['statistics']['overall_accuracy_p_value']
    ci_lower = stats['statistics']['overall_accuracy_ci_95_lower'] 
    ci_upper = stats['statistics']['overall_accuracy_ci_95_upper']
    print(f"{task}: p={p_val:.3f}, CI=[{ci_lower:.3f}, {ci_upper:.3f}]")
```

### Self-Consistency Improvements
```python
# Extract consensus improvements
for task, comp in meta_results['comparison'].items():
    improvement = comp['overall_accuracy']['improvement_percentage']
    p_value = comp['overall_accuracy']['paired_p_value']
    print(f"{task}: +{improvement:.1f}% improvement (p={p_value:.3f})")
```

## 🛠 Advanced Usage

### Compare Frozen vs SFT Results
```bash
# Generate both analyses
bash tmp/run_meta_evaluation.bash --multi-seed-suffix frozen_multi_seeds
bash tmp/run_meta_evaluation.bash --multi-seed-suffix sft_multi_seeds

# Compare JSON outputs for performance differences
```

### Custom Consensus Strategies
```bash
# Try different voting strategies
bash tmp/run_self_consistency.bash --strategy majority
bash tmp/run_self_consistency.bash --strategy exact
bash tmp/run_self_consistency.bash --strategy similarity
```

### Error Bar Visualization
```python
import json
import matplotlib.pyplot as plt

# Load results
data = json.load(open('meta_analysis_majority.json'))

# Plot with error bars
tasks = list(data['multi_seed_results'].keys())
means = [data['multi_seed_results'][task]['statistics']['overall_accuracy_mean'] for task in tasks]
errors = [data['multi_seed_results'][task]['statistics']['overall_accuracy_margin_of_error'] for task in tasks]

plt.errorbar(tasks, means, yerr=errors, fmt='o-')
plt.xticks(rotation=45)
plt.ylabel('Overall Accuracy')
plt.title('Multi-Seed Results with 95% Confidence Intervals')
plt.tight_layout()
plt.show()
```

## 🔧 Dependencies

- **Required**: Python 3.7+, existing MindCube evaluation framework
- **Optional**: scipy (for precise statistical calculations)
- **Fallback**: Manual statistical implementations when scipy unavailable

## 📞 Troubleshooting

### Import Errors
```bash
# If scipy missing, script uses manual calculations
# Warning shown: "scipy not available, using manual statistical calculations"
```

### Missing Multi-Seed Results
```bash
# Ensure you've run inference first
bash tmp/run_frozen_all_seeds.bash

# Check for result files
ls tmp_results/frozen_multi_seeds/*seed*responses.jsonl
```

### Directory Issues
```bash
# Check paths match your setup
ls /projects/b1222/userdata/qineng/01_projects/07_MindCube_new/tmp_results/
```

---

**For questions or issues, refer to the script help:**
```bash
bash tmp/run_meta_evaluation.bash --help
python tmp/meta_evaluation.py --help
``` 