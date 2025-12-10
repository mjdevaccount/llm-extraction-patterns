"""Two-Model Pattern: Smart Reasoning + Lightweight Extraction.

This pattern implements the two-model approach where:
1. Main LLM (large, smart) generates unstructured analysis
2. Extractor LLM (small, specialized) converts to structured JSON

Key Benefits:
- Cost efficiency: Use large model only for reasoning
- VRAM efficiency: Can fit both models on 16GB GPU
- Separation of concerns: Reasoning vs. formatting
"""

from .nodes import TwoModelExtractionNode

# Extractor formatters are in core (reusable across patterns)
from core.extractor_formatters import (
    IExtractorFormatter,
    NuExtractFormatter,
    StandardExtractorFormatter,
)

# JSON repair utilities are now in core
from core.json_repair import repair_json, extract_json_object

__all__ = [
    "IExtractorFormatter",
    "NuExtractFormatter",
    "StandardExtractorFormatter",
    "TwoModelExtractionNode",
    "repair_json",
    "extract_json_object",
]

