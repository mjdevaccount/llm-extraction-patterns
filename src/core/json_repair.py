"""
JSON Repair Utilities - Core Module

Simple, reusable JSON repair functions for handling malformed JSON from LLMs.
These are synchronous utility functions for common JSON issues.

For more advanced repair strategies (async, schema-aware), see extraction_strategies.py
"""

import json
import re
from typing import Dict, Any, Optional


def repair_unquoted_keys(json_str: str) -> str:
    """
    Fix unquoted keys: bin: "" -> "bin": ""
    Also handles: key: value -> "key": value (for any value type)
    
    Args:
        json_str: Potentially malformed JSON string
    
    Returns:
        Repaired JSON string
    """
    # Pattern 1: Unquoted key followed by quoted string (most common)
    repaired = re.sub(r'(\w+):\s*"', r'"\1": "', json_str)
    
    # Pattern 2: Unquoted key at start of object or after comma
    # Match: { key: or , key: (where key is not already quoted)
    # This handles keys before any value type
    repaired = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:\s*)', r'\1"\2"\3', repaired)
    
    return repaired


def repair_single_quotes(json_str: str) -> str:
    """
    Convert single quotes to double quotes.
    
    Args:
        json_str: JSON string with single quotes
    
    Returns:
        JSON string with double quotes
    """
    return json_str.replace("'", '"')


def repair_trailing_commas(json_str: str) -> str:
    """
    Remove trailing commas in objects and arrays.
    
    Args:
        json_str: JSON string with trailing commas
    
    Returns:
        JSON string without trailing commas
    """
    repaired = re.sub(r',\s*}', '}', json_str)
    repaired = re.sub(r',\s*]', ']', repaired)
    return repaired


def repair_json(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to repair common JSON issues.
    
    Applies repairs in order:
    1. Unquoted keys
    2. Single quotes to double quotes
    3. Trailing commas
    
    Args:
        json_str: Potentially malformed JSON string
    
    Returns:
        Repaired JSON dict, or None if repair failed
    """
    # Try direct parse first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Apply repairs
    repaired = json_str
    repaired = repair_unquoted_keys(repaired)
    repaired = repair_single_quotes(repaired)
    repaired = repair_trailing_commas(repaired)
    
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def extract_json_object(text: str) -> Optional[str]:
    """
    Extract the outermost JSON object from text.
    
    Useful when extractor returns JSON mixed with other text.
    
    Args:
        text: Text that may contain JSON
    
    Returns:
        JSON string (just the { ... } block), or None if not found
    """
    start = text.find("{")
    end = text.rfind("}")
    
    if start == -1 or end == -1 or start >= end:
        return None
    
    return text[start : end + 1]

