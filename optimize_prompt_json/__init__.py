"""Iterative LLM prompt optimization for JSON extraction from text."""

__version__ = "0.1.1"

# Prompt type constants (used across modules)
PROMPT_TYPE_JSON_GENERATION = "json_generation"
PROMPT_TYPE_TEXT_GENERATION = "text_generation"
PROMPT_TYPE_JSON_EXTRACTION = "json_extraction"
PROMPT_TYPE_REFINEMENT = "prompt_refinement"
PROMPT_TYPE_BASELINE_EXTRACTION = "baseline_extraction"

# Expose main API for library users
from .pipeline import OptimizationConfig, run_optimization
