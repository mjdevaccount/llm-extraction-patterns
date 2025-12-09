"""LLM provider abstraction - unified interface for all LLM providers."""

from abc import ABC, abstractmethod
from typing import Optional, Literal
from pydantic import BaseModel
import os


class LLMResponse(BaseModel):
    """Normalized response from any LLM provider."""

    content: str
    stop_reason: str  # "end_turn", "max_tokens", "stop_sequence"
    usage: dict  # {"input_tokens": X, "output_tokens": Y}


class LLMClient(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate a response. Returns just the text for simplicity."""
        pass

    @abstractmethod
    def stream(self, system: str, user: str, temperature: float = 0.7):
        """Stream tokens. Useful for long outputs."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI GPT-4 / GPT-4o implementation."""

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install with: pip install openai")

        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens or 2048,
            **kwargs,
        )
        return response.choices[0].message.content

    def stream(self, system: str, user: str, temperature: float = 0.7):
        """Stream from OpenAI."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicClient(LLMClient):
    """Claude API implementation."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Install with: pip install anthropic")

        self.model = model
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Call Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens or 2048,
            **kwargs,
        )
        return response.content[0].text

    def stream(self, system: str, user: str, temperature: float = 0.7):
        """Stream from Anthropic."""
        with self.client.messages.stream(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        ) as stream:
            for text in stream.text_stream:
                yield text


class OllamaClient(LLMClient):
    """Local Ollama LLM implementation (for running locally)."""

    def __init__(self, model: str = "mistral:7b", base_url: str = "http://localhost:11434"):
        try:
            from ollama import Client
        except ImportError:
            raise ImportError("Install with: pip install ollama")

        self.model = model
        self.client = Client(host=base_url)

    def generate(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Call local Ollama."""
        # Ollama uses 'options' dict for parameters
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        
        # Merge any additional options from kwargs
        if "options" in kwargs:
            options.update(kwargs.pop("options"))
        
        response = self.client.generate(
            model=self.model,
            prompt=f"{system}\n\nUser: {user}",
            options=options if options else None,
            stream=False,
            **kwargs,
        )
        return response["response"]

    def stream(self, system: str, user: str, temperature: float = 0.7):
        """Stream from local Ollama."""
        response = self.client.generate(
            model=self.model,
            prompt=f"{system}\n\nUser: {user}",
            temperature=temperature,
            stream=True,
        )
        for chunk in response:
            yield chunk["response"]


# ============================================================
# Factory Function
# ============================================================


def create_llm_client(
    provider: Literal["openai", "anthropic", "ollama"] = "openai",
    model: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """Factory function to create the right LLM client.

    Usage:
        llm = create_llm_client(provider="openai", model="gpt-4")
        response = llm.generate(system="You are helpful", user="Hello")
    """
    provider = provider.lower()

    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4", **kwargs)
    elif provider == "anthropic":
        return AnthropicClient(model=model or "claude-3-5-sonnet-20241022", **kwargs)
    elif provider == "ollama":
        return OllamaClient(model=model or "mistral:7b", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
