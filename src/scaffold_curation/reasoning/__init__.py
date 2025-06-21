"""
Reasoning Chain Generation Module

This module contains reasoning chain generation functionality including:
- Generators for around, among, translation, and rotation scenarios  
- Reasoning chain generation logic for different spatial reasoning settings

Main Components:
- generators: Unified reasoning chain generation logic
- around_reasoning: Around scenario reasoning with spatial transformations
- among_reasoning: Among scenario reasoning 
- translation_reasoning: Translation scenario reasoning
- rotation_reasoning: Rotation scenario reasoning
"""

from .generators import (
    ReasoningChainGenerator,
    generate_around_reasoning,
    generate_among_reasoning,
    generate_translation_reasoning,
    generate_rotation_reasoning,
    add_reasoning_chain_to_item,
    convert_reasoning_to_string
)

__all__ = [
    # Generator classes
    'ReasoningChainGenerator',
    
    # Generation functions for each setting
    'generate_around_reasoning',
    'generate_among_reasoning', 
    'generate_translation_reasoning',
    'generate_rotation_reasoning',
    
    # Utility functions
    'add_reasoning_chain_to_item',
    'convert_reasoning_to_string'
] 