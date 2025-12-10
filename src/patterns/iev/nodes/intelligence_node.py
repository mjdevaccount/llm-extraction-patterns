"""
Intelligence Node for IEV Pattern

Single Responsibility: Analyze input and generate unstructured analysis.
"""

import logging
import time
from typing import Any, Dict

from core.llm_client import LLMClient

logger = logging.getLogger(__name__)


async def intelligence_node(
    state: Dict[str, Any],
    llm_client: LLMClient,
    intelligence_prompt: str,
) -> Dict[str, Any]:
    """
    Intelligence phase: Analyze input.
    
    Args:
        state: IEV state dict with "input" key
        llm_client: LLM client instance
        intelligence_prompt: Prompt template for intelligence phase
    
    Returns:
        Updated state with "analysis" key
    """
    logger.info("[IEV] Intelligence phase")
    start_time = time.time()
    
    try:
        prompt = intelligence_prompt.format(input=state["input"])
        
        # LLMClient.generate() is sync, wrap in async; LangChain has ainvoke()
        import asyncio
        if hasattr(llm_client, 'generate'):
            analysis = await asyncio.to_thread(
                llm_client.generate,
                system="You are a helpful assistant that analyzes information.",
                user=prompt,
            )
        else:
            # LangChain ChatModel
            from langchain_core.messages import HumanMessage
            response = await llm_client.ainvoke([HumanMessage(content=prompt)])
            analysis = response.content if hasattr(response, 'content') else str(response)
        
        duration = (time.time() - start_time) * 1000
        logger.info(f"[IEV] Intelligence complete ({duration:.1f}ms)")
        
        return {"analysis": analysis}
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.exception(f"[IEV] Intelligence error: {e}")
        return {"error": str(e)}

