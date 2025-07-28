#!/usr/bin/env python3
"""
Meta Evaluation Script for MindCube Multi-Seed and Self-Consistency Results

This script runs comprehensive evaluation on:
1. Individual seed results
2. Self-consistency consensus results
3. Outputs comparative meta analysis

Usage:
    python tmp/meta_evaluation.py \
        --multi-seed-dir /path/to/multi_seed_results \
        --consensus-dir /path/to/consensus_results \
        --output-file meta_analysis.json \
        --tasks raw_qa,aug_cgmap_in \
        --seeds 42,123,456
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import statistics
import math

# Try to import scipy for statistical tests, fallback to manual implementation
try:
    from scipy import stats as scipy_stats
    from scipy.stats import t
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using manual statistical calculations")

# Add src to path for imports
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.evaluation.evaluator import evaluate


def make_json_serializable(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


def calculate_t_statistic(values: List[float]) -> tuple:
    """
    Calculate t-statistic and p-value for a one-sample t-test against 0.
    
    Args:
        values: List of values to test
        
    Returns:
        Tuple of (t_statistic, p_value, degrees_of_freedom)
    """
    if len(values) < 2:
        return 0.0, 1.0, 0
    
    n = len(values)
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values)
    
    if std_val == 0:
        return float('inf') if mean_val != 0 else 0.0, 0.0 if mean_val != 0 else 1.0, n-1
    
    # One-sample t-test against 0
    t_stat = mean_val / (std_val / math.sqrt(n))
    df = n - 1
    
    if SCIPY_AVAILABLE:
        # Use scipy for accurate p-value
        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df))
    else:
        # Rough approximation for p-value using normal distribution
        # This is less accurate but better than nothing
        if abs(t_stat) > 3:
            p_value = 0.01
        elif abs(t_stat) > 2:
            p_value = 0.05
        elif abs(t_stat) > 1.96:
            p_value = 0.05
        else:
            p_value = 0.1
    
    return t_stat, p_value, df


def calculate_confidence_interval(values: List[float], confidence_level: float = 0.95) -> tuple:
    """
    Calculate confidence interval for the mean.
    
    Args:
        values: List of values
        confidence_level: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound, margin_of_error)
    """
    if len(values) < 2:
        return 0.0, 0.0, 0.0
    
    n = len(values)
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values)
    
    if std_val == 0:
        return mean_val, mean_val, 0.0
    
    # Standard error
    se = std_val / math.sqrt(n)
    
    # Degrees of freedom
    df = n - 1
    
    # Critical t-value
    if SCIPY_AVAILABLE:
        t_critical = scipy_stats.t.ppf((1 + confidence_level) / 2, df)
    else:
        # Approximation: use 1.96 for 95% CI, 2.58 for 99% CI
        if confidence_level >= 0.99:
            t_critical = 2.58
        elif confidence_level >= 0.95:
            t_critical = 1.96
        else:
            t_critical = 1.645  # 90% CI
    
    # Margin of error
    margin_of_error = t_critical * se
    
    # Confidence interval
    lower_bound = mean_val - margin_of_error
    upper_bound = mean_val + margin_of_error
    
    return lower_bound, upper_bound, margin_of_error


def calculate_paired_t_test(values1: List[float], values2: List[float]) -> tuple:
    """
    Calculate paired t-test between two sets of values.
    
    Args:
        values1: First set of values
        values2: Second set of values
        
    Returns:
        Tuple of (t_statistic, p_value, degrees_of_freedom, mean_difference)
    """
    if len(values1) != len(values2) or len(values1) < 2:
        return 0.0, 1.0, 0, 0.0
    
    # Calculate differences
    differences = [v2 - v1 for v1, v2 in zip(values1, values2)]
    
    if not differences:
        return 0.0, 1.0, 0, 0.0
    
    # One-sample t-test on differences
    t_stat, p_value, df = calculate_t_statistic(differences)
    mean_diff = statistics.mean(differences)
    
    return t_stat, p_value, df, mean_diff


def load_evaluation_result(result_file: str, task_type: str = "cogmap") -> Optional[Dict]:
    """
    Load and evaluate a single result file.
    
    Args:
        result_file: Path to the result JSONL file
        task_type: Type of evaluation task
        
    Returns:
        Evaluation result dictionary or None if failed
    """
    try:
        if not os.path.exists(result_file):
            print(f"Warning: Result file not found: {result_file}")
            return None
        
        # Run evaluation without saving to file
        result = evaluate(result_file, task_type, output_path=None)
        return result['results']
    
    except Exception as e:
        print(f"Error evaluating {result_file}: {e}")
        return None


def extract_key_metrics(eval_result: Dict) -> Dict:
    """
    Extract key metrics from evaluation result.
    
    Args:
        eval_result: Full evaluation result
        
    Returns:
        Dictionary with key metrics
    """
    if not eval_result:
        return {
            'overall_accuracy': 0.0,
            'rotation_accuracy': 0.0,
            'among_accuracy': 0.0,
            'around_accuracy': 0.0,
            'total_samples': 0,
            'valid_rate': 0.0,
            'isomorphic_rate': 0.0,
            'overall_similarity': 0.0
        }
    
    # Basic accuracy metrics
    overall_accuracy = eval_result.get('gen_cogmap_accuracy', 0.0)
    total_samples = eval_result.get('total', 0)
    
    # Setting-specific accuracies
    settings = eval_result.get('settings', {})
    rotation_accuracy = settings.get('rotation', {}).get('gen_cogmap_accuracy', 0.0)
    among_accuracy = settings.get('among', {}).get('gen_cogmap_accuracy', 0.0)
    around_accuracy = settings.get('around', {}).get('gen_cogmap_accuracy', 0.0)
    
    # Graph metrics (if available)
    cogmap_sim = eval_result.get('cogmap_similarity', {})
    valid_rate = cogmap_sim.get('valid_accuracy', 0.0)
    isomorphic_rate = cogmap_sim.get('isomorphic_accuracy', 0.0)
    overall_similarity = cogmap_sim.get('avg_overall_similarity', 0.0)
    
    return {
        'overall_accuracy': round(overall_accuracy, 4),
        'rotation_accuracy': round(rotation_accuracy, 4),
        'among_accuracy': round(among_accuracy, 4),
        'around_accuracy': round(around_accuracy, 4),
        'total_samples': total_samples,
        'valid_rate': round(valid_rate, 4),
        'isomorphic_rate': round(isomorphic_rate, 4),
        'overall_similarity': round(overall_similarity, 4)
    }


def evaluate_multi_seed_results(multi_seed_dir: str, tasks: List[str], seeds: List[int]) -> Dict:
    """
    Evaluate all multi-seed results.
    
    Args:
        multi_seed_dir: Directory containing multi-seed results
        tasks: List of task names
        seeds: List of seed values
        
    Returns:
        Dictionary with multi-seed evaluation results
    """
    results = {}
    
    for task in tasks:
        print(f"\n🔍 Evaluating multi-seed results for task: {task}")
        task_results = {
            'individual_seeds': {},
            'statistics': {}
        }
        
        # Evaluate each seed individually
        seed_metrics = []
        for seed in seeds:
            # Find result file for this seed and task
            pattern = f"*{task}*seed{seed}*responses.jsonl"
            matching_files = list(Path(multi_seed_dir).glob(pattern))
            
            if not matching_files:
                print(f"  Warning: No result file found for task {task} seed {seed}")
                continue
            
            result_file = matching_files[0]
            print(f"  Evaluating seed {seed}: {result_file.name}")
            
            # Run evaluation
            eval_result = load_evaluation_result(str(result_file))
            metrics = extract_key_metrics(eval_result)
            
            task_results['individual_seeds'][f'seed_{seed}'] = metrics
            seed_metrics.append(metrics)
        
        # Calculate statistics across seeds
        if seed_metrics:
            statistics = calculate_seed_statistics(seed_metrics)
            task_results['statistics'] = statistics
        
        results[task] = task_results
    
    return results


def evaluate_consensus_results(consensus_dir: str, tasks: List[str], strategy: str = "majority") -> Dict:
    """
    Evaluate self-consistency consensus results.
    
    Args:
        consensus_dir: Directory containing consensus results
        tasks: List of task names
        strategy: Voting strategy used
        
    Returns:
        Dictionary with consensus evaluation results
    """
    results = {}
    
    for task in tasks:
        print(f"\n🔍 Evaluating consensus results for task: {task}")
        
        # Find consensus result file
        consensus_file = Path(consensus_dir) / f"{task}_consensus_{strategy}.jsonl"
        
        if not consensus_file.exists():
            print(f"  Warning: Consensus file not found: {consensus_file}")
            results[task] = {'metrics': extract_key_metrics(None)}
            continue
        
        print(f"  Evaluating consensus: {consensus_file.name}")
        
        # Run evaluation
        eval_result = load_evaluation_result(str(consensus_file))
        metrics = extract_key_metrics(eval_result)
        
        # Add consensus-specific metrics
        consensus_metrics = calculate_consensus_metrics(str(consensus_file))
        metrics.update(consensus_metrics)
        
        results[task] = {'metrics': metrics}
    
    return results


def calculate_seed_statistics(seed_metrics: List[Dict]) -> Dict:
    """
    Calculate statistics across multiple seeds including p-values and confidence intervals.
    
    Args:
        seed_metrics: List of metric dictionaries from different seeds
        
    Returns:
        Dictionary with statistics (mean, std, min, max, p_value, confidence_interval)
    """
    if not seed_metrics:
        return {}
    
    statistics_result = {}
    
    # Get all metric keys
    metric_keys = set()
    for metrics in seed_metrics:
        metric_keys.update(metrics.keys())
    
    for key in metric_keys:
        if key == 'total_samples':
            # For total_samples, just take the first value (should be same across seeds)
            statistics_result[key] = seed_metrics[0].get(key, 0)
            continue
        
        values = [metrics.get(key, 0.0) for metrics in seed_metrics]
        
        # Filter out None/null values
        values = [v for v in values if v is not None]
        
        if values:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            
            statistics_result[f'{key}_mean'] = round(mean_val, 4)
            statistics_result[f'{key}_std'] = round(std_val, 4)
            statistics_result[f'{key}_min'] = round(min(values), 4)
            statistics_result[f'{key}_max'] = round(max(values), 4)
            
            # Add statistical tests if we have multiple values
            if len(values) >= 2:
                # Calculate p-value (one-sample t-test against 0)
                t_stat, p_value, df = calculate_t_statistic(values)
                statistics_result[f'{key}_t_statistic'] = round(t_stat, 4)
                statistics_result[f'{key}_p_value'] = round(p_value, 4)
                statistics_result[f'{key}_degrees_of_freedom'] = df
                
                # Calculate 95% confidence interval
                ci_lower, ci_upper, margin_error = calculate_confidence_interval(values, 0.95)
                statistics_result[f'{key}_ci_95_lower'] = round(ci_lower, 4)
                statistics_result[f'{key}_ci_95_upper'] = round(ci_upper, 4)
                statistics_result[f'{key}_margin_of_error'] = round(margin_error, 4)
                
                # Standard error
                se = std_val / math.sqrt(len(values)) if std_val > 0 else 0.0
                statistics_result[f'{key}_standard_error'] = round(se, 4)
                
                # Add significance flags (ensure Python bool, not numpy bool)
                statistics_result[f'{key}_significant_p01'] = bool(p_value < 0.01)
                statistics_result[f'{key}_significant_p05'] = bool(p_value < 0.05)
                statistics_result[f'{key}_significant_p10'] = bool(p_value < 0.10)
            else:
                # Single value case
                statistics_result[f'{key}_t_statistic'] = 0.0
                statistics_result[f'{key}_p_value'] = 1.0
                statistics_result[f'{key}_degrees_of_freedom'] = 0
                statistics_result[f'{key}_ci_95_lower'] = mean_val
                statistics_result[f'{key}_ci_95_upper'] = mean_val
                statistics_result[f'{key}_margin_of_error'] = 0.0
                statistics_result[f'{key}_standard_error'] = 0.0
                statistics_result[f'{key}_significant_p01'] = False
                statistics_result[f'{key}_significant_p05'] = False
                statistics_result[f'{key}_significant_p10'] = False
    
    return statistics_result


def calculate_consensus_metrics(consensus_file: str) -> Dict:
    """
    Calculate consensus-specific metrics.
    
    Args:
        consensus_file: Path to consensus result file
        
    Returns:
        Dictionary with consensus metrics
    """
    try:
        consensus_data = []
        with open(consensus_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    consensus_data.append(json.loads(line.strip()))
        
        if not consensus_data:
            return {}
        
        # Calculate consensus confidence statistics
        confidence_scores = []
        perfect_agreement = 0
        high_confidence = 0  # ≥ 2/3 agreement
        
        for item in consensus_data:
            confidence = item.get('consensus_confidence', 0.0)
            confidence_scores.append(confidence)
            
            if confidence == 1.0:
                perfect_agreement += 1
            if confidence >= 0.67:  # 2/3 agreement
                high_confidence += 1
        
        total_samples = len(consensus_data)
        
        return {
            'consensus_confidence_mean': round(statistics.mean(confidence_scores), 4) if confidence_scores else 0.0,
            'consensus_confidence_std': round(statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0, 4),
            'perfect_agreement_rate': round(perfect_agreement / total_samples, 4) if total_samples > 0 else 0.0,
            'high_confidence_rate': round(high_confidence / total_samples, 4) if total_samples > 0 else 0.0,
            'perfect_agreement_count': perfect_agreement,
            'high_confidence_count': high_confidence
        }
    
    except Exception as e:
        print(f"Error calculating consensus metrics: {e}")
        return {}


def compare_results(multi_seed_results: Dict, consensus_results: Dict) -> Dict:
    """
    Compare multi-seed and consensus results with statistical tests.
    
    Args:
        multi_seed_results: Multi-seed evaluation results
        consensus_results: Consensus evaluation results
        
    Returns:
        Dictionary with comparison results including p-values
    """
    comparison = {}
    
    for task in multi_seed_results.keys():
        if task not in consensus_results:
            continue
        
        task_comparison = {}
        
        # Get multi-seed statistics and individual seed values
        multi_seed_stats = multi_seed_results[task].get('statistics', {})
        multi_seed_individual = multi_seed_results[task].get('individual_seeds', {})
        consensus_metrics = consensus_results[task].get('metrics', {})
        
        # Compare key metrics
        metrics_to_compare = [
            'overall_accuracy', 'rotation_accuracy', 'among_accuracy', 
            'around_accuracy', 'valid_rate', 'isomorphic_rate', 'overall_similarity'
        ]
        
        for metric in metrics_to_compare:
            multi_seed_mean = multi_seed_stats.get(f'{metric}_mean', 0.0)
            multi_seed_std = multi_seed_stats.get(f'{metric}_std', 0.0)
            multi_seed_ci_lower = multi_seed_stats.get(f'{metric}_ci_95_lower', multi_seed_mean)
            multi_seed_ci_upper = multi_seed_stats.get(f'{metric}_ci_95_upper', multi_seed_mean)
            multi_seed_p_value = multi_seed_stats.get(f'{metric}_p_value', 1.0)
            
            consensus_value = consensus_metrics.get(metric, 0.0)
            
            improvement = consensus_value - multi_seed_mean if multi_seed_mean > 0 else 0.0
            
            # Extract individual seed values for paired t-test
            seed_values = []
            for seed_key in sorted(multi_seed_individual.keys()):
                seed_data = multi_seed_individual[seed_key]
                if metric in seed_data:
                    seed_values.append(seed_data[metric])
            
            # Perform paired t-test (comparing each seed value to consensus value)
            if len(seed_values) >= 2:
                consensus_values = [consensus_value] * len(seed_values)
                t_stat, p_value_paired, df, mean_diff = calculate_paired_t_test(seed_values, consensus_values)
            else:
                t_stat, p_value_paired, df, mean_diff = 0.0, 1.0, 0, 0.0
            
            task_comparison[metric] = {
                'multi_seed_mean': multi_seed_mean,
                'multi_seed_std': multi_seed_std,
                'multi_seed_ci_95_lower': multi_seed_ci_lower,
                'multi_seed_ci_95_upper': multi_seed_ci_upper,
                'multi_seed_p_value': multi_seed_p_value,
                'consensus_value': consensus_value,
                'improvement': round(improvement, 4),
                'improvement_percentage': round((improvement / multi_seed_mean * 100) if multi_seed_mean > 0 else 0.0, 2),
                'paired_t_statistic': round(t_stat, 4),
                'paired_p_value': round(p_value_paired, 4),
                'paired_degrees_of_freedom': df,
                'consensus_vs_seeds_significant_p01': bool(p_value_paired < 0.01),
                'consensus_vs_seeds_significant_p05': bool(p_value_paired < 0.05),
                'consensus_vs_seeds_significant_p10': bool(p_value_paired < 0.10),
                'effect_size': round(improvement / multi_seed_std if multi_seed_std > 0 else 0.0, 4)  # Cohen's d approximation
            }
        
        comparison[task] = task_comparison
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Meta evaluation for multi-seed and self-consistency results')
    parser.add_argument('--multi-seed-dir', type=str,
                      help='Directory containing multi-seed results (if not provided, will use --multi-seed-suffix)')
    parser.add_argument('--multi-seed-suffix', type=str, default='frozen_multi_seeds',
                      help='Suffix for multi-seed directory under tmp_results/ (default: frozen_multi_seeds)')
    parser.add_argument('--consensus-dir', type=str, required=True,
                      help='Directory containing consensus results')
    parser.add_argument('--output-file', type=str, required=True,
                      help='Output file for meta analysis JSON')
    parser.add_argument('--tasks', type=str, 
                      default='raw_qa,aug_cgmap_in,ff_rsn,aug_cgmap_ffr_out,plain_cgmap_ffr_out,cgmap_in_ffr_out',
                      help='Comma-separated list of tasks')
    parser.add_argument('--seeds', type=str, default='42,123,456',
                      help='Comma-separated list of seeds')
    parser.add_argument('--strategy', type=str, default='majority',
                      help='Consensus strategy used')
    
    args = parser.parse_args()
    
    # Determine multi-seed directory
    if args.multi_seed_dir:
        multi_seed_dir = args.multi_seed_dir
    else:
        # Construct path using suffix
        base_dir = "/workspace/MindCube/tmp_results"
        multi_seed_dir = os.path.join(base_dir, args.multi_seed_suffix)
    
    # Parse inputs
    tasks = [task.strip() for task in args.tasks.split(',')]
    seeds = [int(seed.strip()) for seed in args.seeds.split(',')]
    
    print("🔬 MindCube Meta Evaluation")
    print(f"📋 Tasks: {tasks}")
    print(f"🎲 Seeds: {seeds}")
    print(f"🎯 Strategy: {args.strategy}")
    print(f"📁 Multi-seed dir: {multi_seed_dir}")
    print(f"📁 Consensus dir: {args.consensus_dir}")
    print(f"📄 Output file: {args.output_file}")
    
    # Evaluate multi-seed results
    print("\n" + "="*50)
    print("🎲 Evaluating Multi-Seed Results")
    print("="*50)
    multi_seed_results = evaluate_multi_seed_results(multi_seed_dir, tasks, seeds)
    
    # Evaluate consensus results
    print("\n" + "="*50)
    print("🤖 Evaluating Self-Consistency Results")
    print("="*50)
    consensus_results = evaluate_consensus_results(args.consensus_dir, tasks, args.strategy)
    
    # Compare results
    print("\n" + "="*50)
    print("📊 Comparing Results")
    print("="*50)
    comparison = compare_results(multi_seed_results, consensus_results)
    
    # Create final meta analysis
    meta_analysis = {
        'metadata': {
            'tasks': tasks,
            'seeds': seeds,
            'consensus_strategy': args.strategy,
            'multi_seed_dir': multi_seed_dir,
            'multi_seed_suffix': args.multi_seed_suffix,
            'consensus_dir': args.consensus_dir,
            'evaluation_timestamp': str(os.path.getctime(args.output_file)) if os.path.exists(args.output_file) else "unknown",
            'scipy_available': bool(SCIPY_AVAILABLE)
        },
        'multi_seed_results': multi_seed_results,
        'consensus_results': consensus_results,
        'comparison': comparison
    }
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Make sure all data is JSON serializable
    meta_analysis_clean = make_json_serializable(meta_analysis)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(meta_analysis_clean, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Meta analysis saved to: {args.output_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("📈 Summary")
    print("="*50)
    
    for task in tasks:
        if task in comparison:
            print(f"\n📋 Task: {task}")
            task_comp = comparison[task]
            
            overall_improvement = task_comp.get('overall_accuracy', {}).get('improvement_percentage', 0.0)
            print(f"  Overall accuracy improvement: {overall_improvement:+.2f}%")
            
            rotation_improvement = task_comp.get('rotation_accuracy', {}).get('improvement_percentage', 0.0)
            print(f"  Rotation accuracy improvement: {rotation_improvement:+.2f}%")
            
            among_improvement = task_comp.get('among_accuracy', {}).get('improvement_percentage', 0.0)
            print(f"  Among accuracy improvement: {among_improvement:+.2f}%")
            
            if task in consensus_results:
                perfect_agreement = consensus_results[task]['metrics'].get('perfect_agreement_rate', 0.0)
                print(f"  Perfect agreement rate: {perfect_agreement:.1%}")


if __name__ == '__main__':
    main() 