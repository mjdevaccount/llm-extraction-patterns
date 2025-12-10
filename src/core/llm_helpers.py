"""
LLM Helper Functions

Reusable helper functions for working with different LLM interfaces.
"""

from typing import Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import LLMClient if available
try:
    from .llm_client import LLMClient
    HAS_LLM_CLIENT = True
except ImportError:
    HAS_LLM_CLIENT = False
    LLMClient = None


def is_llm_client(llm: Any) -> bool:
    """Check if llm is an LLMClient instance."""
    if not HAS_LLM_CLIENT:
        return False
    return isinstance(llm, LLMClient)


async def invoke_llm(llm: Any, messages: List[BaseMessage], system: str = None) -> str:
    """
    Unified LLM invocation - works with both LLMClient and LangChain ChatModel.
    
    Args:
        llm: Either LLMClient or LangChain ChatModel
        messages: List of LangChain messages
        system: Optional system prompt (for LLMClient)
    
    Returns:
        Response text content
    """
    if is_llm_client(llm):
        # LLMClient interface: simple generate() method
        user_parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_parts.append(msg.content)
            elif isinstance(msg, AIMessage):
                user_parts.append(f"Assistant: {msg.content}")
        
        user = "\n".join(user_parts)
        system_prompt = system or "You are a helpful assistant."
        
        # LLMClient.generate() is sync, wrap in async
        if hasattr(llm, 'generate'):
            import asyncio
            # Run sync generate() in thread pool to make it async
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: llm.generate(system=system_prompt, user=user)
            )
            return result
        else:
            raise ValueError(f"LLMClient {type(llm)} doesn't have generate() method")
    else:
        # LangChain ChatModel interface: ainvoke()
        response = await llm.ainvoke(messages)
        return response.content if hasattr(response, 'content') else str(response)

