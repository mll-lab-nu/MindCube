#!/usr/bin/env python3
"""
Self-Consistency Script for MindCube Multi-Seed Results

This script reads results from multiple seeds and applies self-consistency to get
the final consensus answers. Supports multiple voting strategies.

Usage:
    python tmp/self_consistency.py \
        --input-dir /path/to/multi_seed_results \
        --output-dir /path/to/consensus_results \
        --task raw_qa \
        --seeds 42,123,456 \
        --strategy majority
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import difflib


def parse_answer_choice(answer: str) -> Optional[str]:
    """
    Extract the choice (A, B, C, D) from various answer formats.
    
    Args:
        answer: Raw answer text
        
    Returns:
        Extracted choice or None if not found
    """
    if not answer:
        return None
    
    # Clean the answer
    answer = answer.strip()
    
    # Pattern 1: <answer>A. Something</answer>
    match = re.search(r'<answer>\s*([A-D])\.?[^<]*</answer>', answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 2: Answer: A or A. Something
    match = re.search(r'(?:answer\s*:?\s*)?([A-D])\.?(?:\s|$)', answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 3: Just the letter at the beginning
    match = re.search(r'^([A-D])\.?(?:\s|$)', answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 4: The answer is A/B/C/D
    match = re.search(r'(?:the\s+answer\s+is\s+)?([A-D])(?:\s|\.|\)|$)', answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return None


def majority_vote(answers: List[str]) -> Tuple[str, float]:
    """
    Perform majority voting on a list of answers.
    
    Args:
        answers: List of answer strings
        
    Returns:
        Tuple of (consensus_answer, confidence_score)
    """
    if not answers:
        return "", 0.0
    
    # Extract choices from all answers
    choices = []
    for answer in answers:
        choice = parse_answer_choice(answer)
        if choice:
            choices.append(choice)
    
    if not choices:
        # If no clear choices, return the first answer as fallback
        return answers[0], 0.0
    
    # Count occurrences
    choice_counts = Counter(choices)
    most_common = choice_counts.most_common(1)[0]
    consensus_choice = most_common[0]
    vote_count = most_common[1]
    
    # Calculate confidence
    confidence = vote_count / len(choices)
    
    # Find the full answer that corresponds to the consensus choice
    for answer in answers:
        if parse_answer_choice(answer) == consensus_choice:
            return answer, confidence
    
    # Fallback
    return answers[0], confidence


def exact_match_vote(answers: List[str]) -> Tuple[str, float]:
    """
    Vote based on exact string matches.
    
    Args:
        answers: List of answer strings
        
    Returns:
        Tuple of (consensus_answer, confidence_score)
    """
    if not answers:
        return "", 0.0
    
    # Clean answers for comparison
    cleaned_answers = [answer.strip().lower() for answer in answers]
    answer_counts = Counter(cleaned_answers)
    
    if len(answer_counts) == 1:
        # All answers are the same
        return answers[0], 1.0
    
    # Find most common
    most_common = answer_counts.most_common(1)[0]
    most_common_cleaned = most_common[0]
    vote_count = most_common[1]
    
    # Find original answer that matches
    for i, cleaned in enumerate(cleaned_answers):
        if cleaned == most_common_cleaned:
            consensus_answer = answers[i]
            break
    else:
        consensus_answer = answers[0]
    
    confidence = vote_count / len(answers)
    return consensus_answer, confidence


def similarity_vote(answers: List[str], threshold: float = 0.7) -> Tuple[str, float]:
    """
    Vote based on string similarity using difflib.
    
    Args:
        answers: List of answer strings
        threshold: Similarity threshold for grouping
        
    Returns:
        Tuple of (consensus_answer, confidence_score)
    """
    if not answers:
        return "", 0.0
    
    if len(answers) == 1:
        return answers[0], 1.0
    
    # Group similar answers
    groups = []
    for answer in answers:
        added_to_group = False
        for group in groups:
            # Check similarity with first answer in group
            similarity = difflib.SequenceMatcher(None, answer.lower(), group[0].lower()).ratio()
            if similarity >= threshold:
                group.append(answer)
                added_to_group = True
                break
        
        if not added_to_group:
            groups.append([answer])
    
    # Find largest group
    largest_group = max(groups, key=len)
    consensus_answer = largest_group[0]  # Use first answer from largest group
    confidence = len(largest_group) / len(answers)
    
    return consensus_answer, confidence


def load_seed_results(input_dir: str, task: str, seeds: List[int]) -> Dict[int, List[Dict]]:
    """
    Load results from multiple seed files.
    
    Args:
        input_dir: Directory containing seed result files
        task: Task name
        seeds: List of seed values
        
    Returns:
        Dictionary mapping seed to list of results
    """
    seed_results = {}
    
    for seed in seeds:
        # Find the result file for this seed and task
        pattern = f"*{task}*seed{seed}*responses.jsonl"
        matching_files = list(Path(input_dir).glob(pattern))
        
        if not matching_files:
            print(f"Warning: No result file found for task {task} seed {seed}")
            continue
        
        if len(matching_files) > 1:
            print(f"Warning: Multiple files found for task {task} seed {seed}, using first one")
        
        result_file = matching_files[0]
        print(f"Loading results from: {result_file}")
        
        # Load results
        results = []
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line.strip()))
        
        seed_results[seed] = results
        print(f"Loaded {len(results)} results for seed {seed}")
    
    return seed_results


def align_results_by_id(seed_results: Dict[int, List[Dict]]) -> Dict[str, Dict[int, Dict]]:
    """
    Align results from different seeds by their ID.
    
    Args:
        seed_results: Dictionary mapping seed to list of results
        
    Returns:
        Dictionary mapping sample ID to seed-result mapping
    """
    aligned_results = defaultdict(dict)
    
    for seed, results in seed_results.items():
        for result in results:
            sample_id = result.get('id')
            if sample_id:
                aligned_results[sample_id][seed] = result
    
    return dict(aligned_results)


def apply_self_consistency(aligned_results: Dict[str, Dict[int, Dict]], 
                          strategy: str = 'majority') -> List[Dict]:
    """
    Apply self-consistency to aligned results.
    
    Args:
        aligned_results: Aligned results by sample ID
        strategy: Voting strategy ('majority', 'exact', 'similarity')
        
    Returns:
        List of consensus results
    """
    consensus_results = []
    
    # Strategy mapping
    strategy_functions = {
        'majority': majority_vote,
        'exact': exact_match_vote,
        'similarity': similarity_vote
    }
    
    if strategy not in strategy_functions:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    vote_function = strategy_functions[strategy]
    
    total_samples = len(aligned_results)
    valid_consensus = 0
    
    for sample_id, seed_results_dict in aligned_results.items():
        if not seed_results_dict:
            continue
        
        # Extract answers from all seeds
        answers = []
        seeds_used = []
        base_result = None
        
        for seed in sorted(seed_results_dict.keys()):
            result = seed_results_dict[seed]
            answer = result.get('answer', '')
            if answer:
                answers.append(answer)
                seeds_used.append(seed)
                if base_result is None:
                    base_result = result.copy()
        
        if not answers:
            print(f"Warning: No answers found for sample {sample_id}")
            continue
        
        # Apply voting strategy
        consensus_answer, confidence = vote_function(answers)
        
        if confidence > 0:
            valid_consensus += 1
        
        # Create consensus result
        consensus_result = base_result.copy()
        consensus_result['consensus_answer'] = consensus_answer
        consensus_result['individual_answers'] = {f'seed_{seed}': answers[i] for i, seed in enumerate(seeds_used)}
        consensus_result['consensus_confidence'] = confidence
        consensus_result['consensus_strategy'] = strategy
        consensus_result['seeds_used'] = seeds_used
        consensus_result['num_seeds'] = len(seeds_used)
        
        # Extract individual choices for analysis
        individual_choices = {}
        for i, seed in enumerate(seeds_used):
            choice = parse_answer_choice(answers[i])
            if choice:
                individual_choices[f'seed_{seed}_choice'] = choice
        consensus_result['individual_choices'] = individual_choices
        
        # Extract consensus choice
        consensus_choice = parse_answer_choice(consensus_answer)
        if consensus_choice:
            consensus_result['consensus_choice'] = consensus_choice
        
        consensus_results.append(consensus_result)
    
    print(f"Processed {len(consensus_results)}/{total_samples} samples")
    print(f"Valid consensus for {valid_consensus}/{len(consensus_results)} samples")
    
    return consensus_results


def main():
    parser = argparse.ArgumentParser(description='Apply self-consistency to multi-seed results')
    parser.add_argument('--input-dir', type=str, required=True,
                      help='Directory containing multi-seed result files')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save consensus results')
    parser.add_argument('--task', type=str, required=True,
                      help='Task name (e.g., raw_qa, aug_cgmap_in)')
    parser.add_argument('--seeds', type=str, default='42,123,456',
                      help='Comma-separated list of seeds (default: 42,123,456)')
    parser.add_argument('--strategy', type=str, default='majority',
                      choices=['majority', 'exact', 'similarity'],
                      help='Voting strategy (default: majority)')
    parser.add_argument('--similarity-threshold', type=float, default=0.7,
                      help='Similarity threshold for similarity strategy (default: 0.7)')
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    print(f"Using seeds: {seeds}")
    print(f"Task: {args.task}")
    print(f"Strategy: {args.strategy}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load seed results
    print("\n=== Loading Seed Results ===")
    seed_results = load_seed_results(args.input_dir, args.task, seeds)
    
    if not seed_results:
        print("Error: No seed results loaded!")
        return
    
    # Align results by ID
    print("\n=== Aligning Results by ID ===")
    aligned_results = align_results_by_id(seed_results)
    print(f"Found {len(aligned_results)} unique samples")
    
    # Apply self-consistency
    print(f"\n=== Applying Self-Consistency ({args.strategy}) ===")
    if args.strategy == 'similarity':
        # Pass threshold for similarity strategy
        consensus_results = apply_self_consistency(aligned_results, args.strategy)
    else:
        consensus_results = apply_self_consistency(aligned_results, args.strategy)
    
    # Save results
    output_file = Path(args.output_dir) / f"{args.task}_consensus_{args.strategy}.jsonl"
    print(f"\n=== Saving Results ===")
    print(f"Output file: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in consensus_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(consensus_results)} consensus results")
    
    # Print statistics
    print(f"\n=== Statistics ===")
    if consensus_results:
        confidence_scores = [r.get('consensus_confidence', 0) for r in consensus_results]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        print(f"Average confidence: {avg_confidence:.3f}")
        
        high_confidence = sum(1 for c in confidence_scores if c >= 0.67)  # 2/3 agreement
        print(f"High confidence samples (≥2/3 agreement): {high_confidence}/{len(consensus_results)} ({high_confidence/len(consensus_results)*100:.1f}%)")
        
        perfect_agreement = sum(1 for c in confidence_scores if c == 1.0)
        print(f"Perfect agreement samples: {perfect_agreement}/{len(consensus_results)} ({perfect_agreement/len(consensus_results)*100:.1f}%)")


if __name__ == '__main__':
    main()
