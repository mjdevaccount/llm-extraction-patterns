"""
LLM Adapter for Dependency Inversion

Wraps concrete LLM implementations (LangChain, etc.) to provide
a consistent interface (Dependency Inversion Principle).
"""

from typing import Any, List, Type
from pydantic import BaseModel

from .abstractions import ILLMProvider


class LangChainLLMAdapter(ILLMProvider):
    """
    Adapter for LangChain ChatModel implementations.
    
    Wraps LangChain LLMs to provide ILLMProvider interface.
    """
    
    def __init__(self, llm: Any):
        """
        Initialize adapter with LangChain LLM.
        
        Args:
            llm: LangChain ChatModel instance
        """
        self.llm = llm
    
    async def invoke(self, messages: List[Any]) -> str:
        """Invoke LLM and return text response."""
        response = await self.llm.ainvoke(messages)
        return response.content if hasattr(response, 'content') else str(response)
    
    async def invoke_structured(
        self, 
        messages: List[Any], 
        schema: Type[BaseModel]
    ) -> BaseModel:
        """Invoke LLM with structured output."""
        if hasattr(self.llm, 'with_structured_output'):
            structured_llm = self.llm.with_structured_output(schema)
            return await structured_llm.ainvoke(messages)
        else:
            # Fallback: invoke and parse manually
            text = await self.invoke(messages)
            # Try to extract JSON and parse
            import json
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return schema(**data)
            raise ValueError(f"Could not extract structured output from: {text}")
    
    @property
    def supports_structured_output(self) -> bool:
        """Check if LLM supports structured output."""
        return hasattr(self.llm, 'with_structured_output')

