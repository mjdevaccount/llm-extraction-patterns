"""
Abstractions for Two-Model Pattern

Defines the interface for extractor formatters that map unstructured
text from the main LLM to structured input for the extractor LLM.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Union
from pydantic import BaseModel


class IExtractorFormatter(ABC):
    """
    Abstract interface for extractor formatters.
    
    Single Responsibility: Format unstructured text + schema into
    extractor LLM prompt format.
    
    Open/Closed Principle: New extractor formats can be added
    without modifying the extraction node.
    
    This abstraction handles the mapping between:
    - Main LLM output (unstructured text)
    - Extractor LLM input (formatted prompt)
    """
    
    @abstractmethod
    def format_prompt(
        self,
        unstructured_text: str,
        schema: Union[Type[BaseModel], Dict[str, Any]],
        max_length: int = 4000
    ) -> str:
        """
        Format unstructured text and schema into extractor prompt.
        
        Args:
            unstructured_text: Raw output from main LLM
            schema: Pydantic model or dict schema defining structure
            max_length: Maximum prompt length (for truncation)
        
        Returns:
            Formatted prompt string ready for extractor LLM
        """
        pass
    
    @abstractmethod
    def extract_json_from_response(
        self,
        extractor_response: str
    ) -> str:
        """
        Extract JSON string from extractor LLM response.
        
        Different extractors may wrap JSON in different formats
        (markdown, special tokens, etc.).
        
        Args:
            extractor_response: Raw response from extractor LLM
        
        Returns:
            JSON string (may need parsing)
        """
        pass
    
    @abstractmethod
    def get_extractor_config(self) -> Dict[str, Any]:
        """
        Get recommended configuration for extractor LLM.
        
        Returns:
            Dict with recommended settings (temperature, etc.)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Formatter identifier."""
        pass

