"""
Concrete Extractor Formatter Implementations

Implements specific formatting strategies for different extractor models.
"""

import json
from typing import Any, Dict, Type, Union
from pydantic import BaseModel

from .abstractions import IExtractorFormatter


class NuExtractFormatter(IExtractorFormatter):
    """
    Formatter for NuExtract model format.
    
    NuExtract expects:
    <|input|>
    ### Template:
    {schema_json}
    ### Text:
    {unstructured_text}
    <|output|>
    
    And returns JSON after <|output|> token.
    """
    
    def format_prompt(
        self,
        unstructured_text: str,
        schema: Union[Type[BaseModel], Dict[str, Any]],
        max_length: int = 4000
    ) -> str:
        """
        Format prompt in NuExtract's expected format.
        
        NuExtract expects exact pattern:
        <|input|>
        ### Template:
        {schema_json}
        ### Text:
        {unstructured_text}
        <|output|>
        
        Args:
            unstructured_text: Raw output from main LLM
            schema: Pydantic model or dict schema (NuExtract prefers dict with string templates)
            max_length: Maximum prompt length
        
        Returns:
            Formatted prompt string
        """
        # Convert schema to JSON string
        # NuExtract wants dict schemas with string templates, not Pydantic models
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # For Pydantic, convert to dict template format
            schema_dict = schema.model_json_schema()
        elif isinstance(schema, dict):
            schema_dict = schema
        else:
            raise ValueError(f"Schema must be Pydantic model or dict, got {type(schema)}")
        
        schema_str = json.dumps(schema_dict, indent=4)
        
        # Build NuExtract prompt format
        # Ollama's nuextract wraps with <|input|> / <|output|> internally
        # We just provide the Template and Text sections
        prompt = f"""### Template:

{schema_str}

### Text:

{unstructured_text}
"""
        
        return prompt
    
    def extract_json_from_response(
        self,
        extractor_response: str
    ) -> str:
        """
        Extract JSON from NuExtract response.
        
        NuExtract with Ollama tends to return the JSON directly or prefixed with the template.
        Extract the outermost { ... } block.
        
        Args:
            extractor_response: Raw response from NuExtract
        
        Returns:
            JSON string (just the { ... } block)
        """
        # Keep only the JSON object - find outermost { ... }
        start = extractor_response.find("{")
        end = extractor_response.rfind("}")
        
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in nuextract output")
        
        return extractor_response[start : end + 1]
    
    def get_extractor_config(self) -> Dict[str, Any]:
        """
        Get recommended configuration for NuExtract.
        
        NuExtract needs:
        - temperature=0.0 (deterministic extraction)
        """
        return {
            "temperature": 0.0,
        }
    
    @property
    def name(self) -> str:
        return "NuExtract"


class StandardExtractorFormatter(IExtractorFormatter):
    """
    Generic formatter for standard LLM models (Ollama, OpenAI, etc.).
    
    Uses a simple prompt format that works with most models:
    - Clear instructions for JSON extraction
    - Schema provided as JSON
    - Unstructured text to extract from
    
    Works with models like qwen2.5:7b, mistral:7b, llama3.2, etc.
    """
    
    def format_prompt(
        self,
        unstructured_text: str,
        schema: Union[Type[BaseModel], Dict[str, Any]],
        max_length: int = 4000
    ) -> str:
        """
        Format prompt for standard LLM extraction.
        
        Args:
            unstructured_text: Raw output from main LLM
            schema: Pydantic model or dict schema
            max_length: Maximum prompt length
        
        Returns:
            Formatted prompt string
        """
        # Convert schema to JSON string
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_dict = schema.model_json_schema()
        elif isinstance(schema, dict):
            schema_dict = schema
        else:
            raise ValueError(f"Schema must be Pydantic model or dict, got {type(schema)}")
        
        schema_str = json.dumps(schema_dict, indent=2)
        
        # Build standard extraction prompt
        prompt = f"""Extract structured data from the following text and return it as valid JSON matching the provided schema.

Schema:
{schema_str}

Text to extract from:
{unstructured_text}

Return only valid JSON matching the schema. Do not include any explanation or markdown formatting."""

        # Truncate if needed
        if len(prompt) > max_length:
            available = max_length - len(schema_str) - 300  # Buffer for prompt text
            truncated_text = unstructured_text[:available] + "..."
            prompt = f"""Extract structured data from the following text and return it as valid JSON matching the provided schema.

Schema:
{schema_str}

Text to extract from:
{truncated_text}

Return only valid JSON matching the schema. Do not include any explanation or markdown formatting."""
        
        return prompt
    
    def extract_json_from_response(
        self,
        extractor_response: str
    ) -> str:
        """
        Extract JSON from standard LLM response.
        
        Standard models may return JSON in code blocks or plain text.
        
        Args:
            extractor_response: Raw response from extractor LLM
        
        Returns:
            JSON string
        """
        import re
        
        # Try to find JSON in code blocks first
        json_block = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", extractor_response, re.DOTALL)
        if json_block:
            return json_block.group(1).strip()
        
        # Try to find JSON object/array directly
        json_match = re.search(r"(\{.*?\}|\[.*?\])", extractor_response, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Fallback: return entire response (might be JSON)
        return extractor_response.strip()
    
    def get_extractor_config(self) -> Dict[str, Any]:
        """
        Get recommended configuration for standard extractor models.
        
        Standard models need:
        - temperature=0.0 or very low (deterministic extraction)
        - Clear instructions in prompt
        - Higher max_tokens for complex schemas
        """
        return {
            "temperature": 0.0,
            "num_predict": 4000,  # Higher for complex schemas
        }
    
    @property
    def name(self) -> str:
        return "Standard"

