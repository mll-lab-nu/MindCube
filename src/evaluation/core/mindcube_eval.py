"""
MindCube Evaluation API

Simple Python API for evaluation that can be imported from project root.

Usage:
    import mindcube_eval as mc
    
    # Basic evaluation
    results = mc.evaluate("responses.jsonl", "basic")
    
    # Cognitive map evaluation
    results = mc.evaluate("responses.jsonl", "cogmap")
    
    # Auto-detect 
    results = mc.auto_evaluate("responses.jsonl")
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import evaluation functions
from src.evaluation.evaluator import evaluate, auto_evaluate, BasicEvaluator
from src.evaluation.cogmap import CogMapEvaluator
from src.evaluation import quick_start_guide, batch_evaluate

# Re-export main functions
__all__ = [
    'evaluate',
    'auto_evaluate', 
    'BasicEvaluator',
    'CogMapEvaluator',
    'quick_start_guide',
    'batch_evaluate'
]


def quick_eval(jsonl_path: str, task_type: str = "auto"):
    """
    Quick evaluation with minimal code.
    
    Args:
        jsonl_path: Path to JSONL file 
        task_type: "basic", "cogmap", or "auto"
        
    Returns:
        Accuracy percentage
    """
    if task_type == "auto":
        results = auto_evaluate(jsonl_path)
    else:
        results = evaluate(jsonl_path, task_type)
    
    return results['results']['gen_cogmap_accuracy'] * 100


# Convenience function
def help():
    """Show usage guide."""
    print("""
🚀 MindCube Evaluation API

=== BASIC USAGE ===

import mindcube_eval as mc

# Method 1: Auto-detect task type (recommended)
results = mc.auto_evaluate("responses.jsonl")

# Method 2: Specify task type  
results = mc.evaluate("responses.jsonl", "basic")     # Answer accuracy only
results = mc.evaluate("responses.jsonl", "cogmap")    # Full cognitive map analysis

# Method 3: Quick evaluation (just get accuracy)
accuracy = mc.quick_eval("responses.jsonl", "auto")
print(f"Accuracy: {accuracy:.1f}%")

=== ADVANCED USAGE ===

# Custom evaluators
evaluator = mc.BasicEvaluator()
results = evaluator.evaluate("responses.jsonl")

evaluator = mc.CogMapEvaluator(include_detailed_metrics=False)
results = evaluator.evaluate("responses.jsonl")

# Batch evaluation
mc.batch_evaluate("results_dir/", "output_dir/")

=== RESULT STRUCTURE ===

results = {
    'results': {
        'total': 100,
        'gen_cogmap_correct': 75,
        'gen_cogmap_accuracy': 0.75,
        'settings': {...},          # Per-setting breakdown
        'cogmap_similarity': {...}  # Only for cogmap evaluation
    },
    'error_cases': [...]           # Failed extractions for debugging
}
""")


if __name__ == "__main__":
    help() 