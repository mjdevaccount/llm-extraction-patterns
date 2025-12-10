"""
LLM Factory - Core Module

Centralizes LLM creation to avoid repetition and enable easy configuration.
Moved from pattern-specific folders to core for reusability.
"""

import os
from typing import Optional, Any

try:
    from langchain_ollama import ChatOllama
    HAS_LANGCHAIN_OLLAMA = True
except ImportError:
    HAS_LANGCHAIN_OLLAMA = False
    ChatOllama = None

# Fallback to OllamaClient if langchain-ollama not available
try:
    from .llm_client import LLMClient
    HAS_LLM_CLIENT = True
except ImportError:
    HAS_LLM_CLIENT = False
    LLMClient = None


BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_reasoning_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    num_ctx: int = 4096,
    num_predict: int = 2048,
) -> Any:
    """
    Get main reasoning LLM (high temperature for creative analysis).
    
    Args:
        model: Model name (defaults to REASONING_MODEL env var or "qwen2.5:14b")
        base_url: Ollama base URL (defaults to OLLAMA_BASE_URL env var)
        temperature: Temperature for reasoning (default 0.7)
        num_ctx: Context window size (default 4096)
        num_predict: Max tokens to generate (default 2048)
    
    Returns:
        ChatOllama or LLMClient instance
    """
    model = model or os.getenv("REASONING_MODEL", "qwen2.5:14b")
    base_url = base_url or BASE_URL
    
    if HAS_LANGCHAIN_OLLAMA:
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )
    elif HAS_LLM_CLIENT:
        return LLMClient(
            provider="ollama",
            model=model,
            base_url=base_url
        )
    else:
        raise ImportError(
            "Neither langchain-ollama nor LLMClient available. "
            "Install with: pip install langchain-ollama"
        )


def get_extractor_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    num_ctx: int = 4096,
    num_predict: int = 4000,
) -> Any:
    """
    Get extractor LLM (low temperature for deterministic extraction).
    
    Args:
        model: Model name (defaults to EXTRACTOR_MODEL env var or "nuextract")
        base_url: Ollama base URL (defaults to OLLAMA_BASE_URL env var)
        temperature: Temperature for extraction (default 0.0)
        num_ctx: Context window size (default 4096)
        num_predict: Max tokens to generate (default 4000)
    
    Returns:
        ChatOllama or LLMClient instance
    """
    model = model or os.getenv("EXTRACTOR_MODEL", "nuextract")
    base_url = base_url or BASE_URL
    
    if HAS_LANGCHAIN_OLLAMA:
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )
    elif HAS_LLM_CLIENT:
        return LLMClient(
            provider="ollama",
            model=model,
            base_url=base_url
        )
    else:
        raise ImportError(
            "Neither langchain-ollama nor LLMClient available. "
            "Install with: pip install langchain-ollama"
        )


def get_extractor_llm_from_formatter(
    formatter: Any,  # IExtractorFormatter from core.extractor_formatters
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Any:
    """
    Get extractor LLM with config from formatter's recommendations.
    
    Args:
        formatter: IExtractorFormatter instance (from core.extractor_formatters)
        model: Model name (defaults to EXTRACTOR_MODEL env var or "nuextract")
        base_url: Ollama base URL (defaults to OLLAMA_BASE_URL env var)
    
    Returns:
        ChatOllama or LLMClient instance configured per formatter
    """
    config = formatter.get_extractor_config()
    model = model or os.getenv("EXTRACTOR_MODEL", "nuextract")
    base_url = base_url or BASE_URL
    
    if HAS_LANGCHAIN_OLLAMA:
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=config.get("temperature", 0.0),
            num_ctx=4096,
            num_predict=config.get("num_predict", 4000),
        )
    elif HAS_LLM_CLIENT:
        return LLMClient(
            provider="ollama",
            model=model,
            base_url=base_url
        )
    else:
        raise ImportError(
            "Neither langchain-ollama nor LLMClient available. "
            "Install with: pip install langchain-ollama"
        )

