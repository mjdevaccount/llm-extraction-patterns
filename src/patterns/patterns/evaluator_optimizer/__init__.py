"""
Evaluator-Optimizer Pattern: Draft-Critique-Refine

A workflow that iteratively improves output quality:
  1. Draft: Generate initial output
  2. Evaluate: Critique the output with scoring
  3. Optimize: Refine based on feedback (repeat until satisfied)

Use for:
  - Code review automation
  - Content refinement
  - Quality gates (before final delivery)
  - Iterative design feedback

Implementation Status: Basic (Dec 2025)
  - Core pattern defined
  - Grader interface for custom scoring
  - Example: code_review.py
"""

from .graph import create_evaluator_optimizer_graph
from .grader import BaseGrader, CodeQualityGrader

__all__ = [
    "create_evaluator_optimizer_graph",
    "BaseGrader",
    "CodeQualityGrader",
]
