"""Two-Model Pattern: Smart Reasoning + Lightweight Extraction.

This pattern implements the two-model approach where:
1. Main LLM (large, smart) generates unstructured analysis
2. Extractor LLM (small, specialized) converts to structured JSON

Key Benefits:
- Cost efficiency: Use large model only for reasoning
- VRAM efficiency: Can fit both models on 16GB GPU
- Separation of concerns: Reasoning vs. formatting
"""

from .abstractions import IExtractorFormatter
from .formatters import NuExtractFormatter, StandardExtractorFormatter
from .nodes import TwoModelExtractionNode

__all__ = [
    "IExtractorFormatter",
    "NuExtractFormatter",
    "StandardExtractorFormatter",
    "TwoModelExtractionNode",
]

