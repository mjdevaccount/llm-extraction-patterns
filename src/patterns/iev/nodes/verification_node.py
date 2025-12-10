"""
Verification Node for IEV Pattern

Single Responsibility: Verify extracted data using optional validation strategy.
"""

import json
import logging
import time
from typing import Any, Dict, Type, Optional

from pydantic import BaseModel

from core.llm_client import LLMClient
from core.abstractions import IValidationStrategy

logger = logging.getLogger(__name__)


async def verification_node(
    state: Dict[str, Any],
    llm_client: LLMClient,
    output_schema: Type[BaseModel],
    verification_prompt: str,
    validation_strategy: Optional[IValidationStrategy] = None,
) -> Dict[str, Any]:
    """
    Verification phase: Verify extracted data using optional validation strategy.
    
    Args:
        state: IEV state dict with "extracted" key
        llm_client: LLM client instance
        output_schema: Pydantic schema for validation
        verification_prompt: Prompt template for verification phase
        validation_strategy: Optional validation strategy (SOLID pattern)
    
    Returns:
        Updated state with "verification_status" and "validated" keys
    """
    logger.info("[IEV] Verification phase")
    
    try:
        if "error" in state and state["error"]:
            logger.warning(f"[IEV] Verification skipped due to error")
            return {
                "verification_status": "REJECTED",
                "error": state["error"],
            }

        extracted = state.get("extracted")
        if extracted is None:
            logger.warning(f"[IEV] Verification skipped - no extracted data")
            return {
                "verification_status": "REJECTED",
                "error": "No extracted data to verify",
            }

        # Use validation strategy if provided (SOLID: Open/Closed Principle)
        if validation_strategy:
            logger.info(f"[IEV] Using validation strategy: {validation_strategy.mode_name}")
            # Convert to dict if Pydantic model
            if isinstance(extracted, BaseModel):
                extracted_dict = extracted.model_dump()
            else:
                extracted_dict = extracted
            
            validation_result = await validation_strategy.validate(
                data=extracted_dict,
                schema=output_schema,
                validation_rules={},  # Can be customized
                llm=llm_client,
                max_retries=2,
            )
            
            status = "APPROVED" if validation_result.get("outcome") == "APPROVED" else "REJECTED"
            validated = extracted if status == "APPROVED" else None
            
            logger.info(f"[IEV] Verification complete: {status}")
            
            return {
                "verification_status": status,
                "validated": validated,
            }
        
        # Fallback: Simple LLM-based verification
        if isinstance(extracted, BaseModel):
            extracted_dict = extracted.model_dump()
        else:
            extracted_dict = extracted

        prompt = verification_prompt.format(
            extracted=json.dumps(extracted_dict, indent=2, default=str)
        )
        
        # LLMClient.generate() is sync, wrap in async; LangChain has ainvoke()
        import asyncio
        if hasattr(llm_client, 'generate'):
            verification_result = await asyncio.to_thread(
                llm_client.generate,
                system="You are a helpful assistant that verifies data quality and safety.",
                user=prompt,
            )
        else:
            # LangChain ChatModel
            from langchain_core.messages import HumanMessage
            response = await llm_client.ainvoke([HumanMessage(content=prompt)])
            verification_result = response.content if hasattr(response, 'content') else str(response)
        
        verification_result = verification_result.upper()

        if "APPROVED" in verification_result:
            status = "APPROVED"
            validated = extracted
        else:
            status = "REJECTED"
            validated = None

        logger.info(f"[IEV] Verification complete: {status}")
        
        return {
            "verification_status": status,
            "validated": validated,
        }
    except Exception as e:
        logger.exception(f"[IEV] Verification error: {e}")
        return {
            "verification_status": "REJECTED",
            "error": str(e),
        }

