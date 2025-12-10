"""Adapter to make LLMClient work with LangChain-style interfaces."""

from typing import Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel

from .llm_client import LLMClient


class LLMClientAdapter(BaseChatModel):
    """
    Adapter that wraps LLMClient to work as a LangChain ChatModel.
    
    This allows patterns/iev/nodes.py to use local_models/nodes/ 
    with the simpler LLMClient interface.
    """
    
    def __init__(self, llm_client: LLMClient, temperature: float = 0.7):
        """
        Initialize adapter.
        
        Args:
            llm_client: LLMClient instance (from core.llm_client)
            temperature: Default temperature for generation
        """
        super().__init__()
        self.llm_client = llm_client
        self.temperature = temperature
    
    @property
    def _llm_type(self) -> str:
        """Return LLM type identifier."""
        return "llm_client_adapter"
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate response from messages.
        
        Converts LangChain messages to LLMClient format.
        """
        # Extract system and user messages
        system_parts = []
        user_parts = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_parts.append(msg.content)
            elif isinstance(msg, AIMessage):
                # Previous assistant response - include in context
                user_parts.append(f"Assistant: {msg.content}")
            elif hasattr(msg, 'content'):
                # System message or other
                if hasattr(msg, 'type') and msg.type == 'system':
                    system_parts.append(msg.content)
                else:
                    user_parts.append(msg.content)
        
        system = "\n".join(system_parts) if system_parts else "You are a helpful assistant."
        user = "\n".join(user_parts)
        
        # Get temperature from kwargs or use default
        temperature = kwargs.get('temperature', self.temperature)
        
        # Call LLMClient
        response_text = await self.llm_client.generate(
            system=system,
            user=user,
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k != 'temperature'}
        )
        
        # Return LangChain-style response
        from langchain_core.messages import AIMessage
        return AIMessage(content=response_text)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronous generate (not implemented - use async)."""
        raise NotImplementedError("Use ainvoke() for async generation")

