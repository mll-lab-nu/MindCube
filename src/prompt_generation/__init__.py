"""
Prompt Generation Module

This module handles the generation of final model-ready prompts from scaffold data.
Supports multiple task settings with different prompt construction strategies.

Main Components:
- generators: Core prompt generation logic for different tasks
- processors: Batch processing and file operations
- templates: Prompt templates and formatting utilities
"""

from .processors import (
    generate_task_prompts,
    batch_generate_prompts,
    PromptProcessor
)

from .generators import (
    PromptGenerator,
    RawQAGenerator,
    FFRSNGenerator
)

__all__ = [
    # Main processing functions
    'generate_task_prompts',
    'batch_generate_prompts',
    
    # Processor classes
    'PromptProcessor',
    
    # Generator classes
    'PromptGenerator',
    'RawQAGenerator',
    'FFRSNGenerator'
] 