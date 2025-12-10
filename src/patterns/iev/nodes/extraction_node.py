"""
Extraction Node for IEV Pattern

Single Responsibility: Extract structured data from analysis with retry logic.
"""

import json
import logging
import re
import time
from typing import Any, Dict, Type

from pydantic import BaseModel, ValidationError

from core.llm_client import LLMClient
from core.retry_strategies import IRetryStrategy
from core.json_repair import extract_json_object, repair_json

logger = logging.getLogger(__name__)


async def extraction_node(
    state: Dict[str, Any],
    llm_client: LLMClient,
    output_schema: Type[BaseModel],
    extraction_prompt: str,
    retry_strategy: IRetryStrategy,
    max_extraction_retries: int = 4,
) -> Dict[str, Any]:
    """
    Extraction phase: Extract structured data with retry logic.
    
    Args:
        state: IEV state dict with "analysis" key
        llm_client: LLM client instance
        output_schema: Pydantic schema for extraction
        extraction_prompt: Prompt template for extraction phase
        retry_strategy: Retry strategy for failed extractions
        max_extraction_retries: Maximum extraction retry attempts
    
    Returns:
        Updated state with "extracted" key and "extraction_attempts"
    """
    logger.info("[IEV] Extraction phase")
    attempt = 0
    
    try:
        if "error" in state and state["error"]:
            return {"error": state["error"]}

        analysis = state.get("analysis", "")
        base_prompt = extraction_prompt.format(analysis=analysis)

        # Add schema to prompt
        schema_json = output_schema.model_json_schema()
        base_prompt += f"\n\nExtract data matching this schema (return valid JSON only):\n{json.dumps(schema_json, indent=2)}"

        # Try extraction with retries
        for attempt in range(max_extraction_retries):
            logger.info(f"[IEV] Extraction attempt {attempt + 1}/{max_extraction_retries}")
            
            try:
                # LLMClient.generate() is sync, wrap in async; LangChain has ainvoke()
                import asyncio
                if hasattr(llm_client, 'generate'):
                    response_text = await asyncio.to_thread(
                        llm_client.generate,
                        system="You are a helpful assistant that extracts structured data.",
                        user=base_prompt,
                    )
                else:
                    # LangChain ChatModel
                    from langchain_core.messages import HumanMessage
                    response = await llm_client.ainvoke([HumanMessage(content=base_prompt)])
                    response_text = response.content if hasattr(response, 'content') else str(response)

                # Extract and repair JSON using core utilities
                json_text = extract_json_object(response_text)
                if json_text is None:
                    # Fallback: try code block extraction
                    json_block = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", response_text, re.DOTALL)
                    if json_block:
                        json_text = json_block.group(1)
                    else:
                        json_text = response_text.strip()
                
                json_data = repair_json(json_text)
                if json_data is None:
                    raise ValueError(f"Could not extract or repair JSON from: {response_text[:200]}")

                # Parse with Pydantic
                parsed = output_schema.model_validate(json_data)
                
                logger.info(f"[IEV] Extraction successful (attempt {attempt + 1})")
                
                return {"extracted": parsed, "extraction_attempts": attempt + 1}
            
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                logger.warning(f"[IEV] Extraction attempt {attempt + 1} failed: {e}")
                
                # If we have more retries, try retry strategy
                if attempt < max_extraction_retries - 1:
                    logger.info(f"[IEV] Using retry strategy: {retry_strategy.name}")
                    retry_result = await retry_strategy.retry(
                        llm=llm_client,
                        original_prompt=base_prompt,
                        failed_output=response_text if 'response_text' in locals() else "",
                        error_message=str(e),
                        schema=output_schema,
                        attempt_number=attempt + 1,
                    )
                    
                    if retry_result is not None:
                        logger.info(f"[IEV] Retry successful on attempt {attempt + 1}")
                        parsed = output_schema.model_validate(retry_result)
                        logger.info(f"[IEV] Extraction complete (attempt {attempt + 2})")
                        return {"extracted": parsed, "extraction_attempts": attempt + 2}
                
                # Continue to next attempt or fail
                if attempt == max_extraction_retries - 1:
                    # Last attempt failed
                    logger.error(f"[IEV] Extraction failed after {max_extraction_retries} attempts")
                    return {
                        "error": f"Extraction failed after {max_extraction_retries} attempts: {e}",
                        "extraction_attempts": attempt + 1,
                    }
        
        # Should not reach here
        raise RuntimeError("Extraction loop exited unexpectedly")
    
    except Exception as e:
        logger.exception(f"[IEV] Extraction error: {e}")
        return {"error": str(e), "extraction_attempts": attempt + 1}

