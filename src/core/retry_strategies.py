"""
Retry Strategies - Core Module

SOLID-compliant retry strategies for extraction when JSON repair fails.
Implements intelligent retry logic that refines prompts based on errors.

SOLID Principles:
  - Single Responsibility: Each strategy has one retry approach
  - Open/Closed: New strategies can be added without modifying extraction node
  - Dependency Inversion: ExtractionNode depends on IRetryStrategy interface
"""

import logging
import json
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel
from .abstractions import IRetryStrategy
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class PromptRefinementRetry(IRetryStrategy):
    """
    Retry by refining the original prompt with stricter instructions.
    
    Attempt 1: "Extract data matching schema"
    Attempt 2: "Extract data ONLY as valid JSON, no explanation"
    Attempt 3: "Return only valid JSON object, no markdown, no extra text"
    """
    
    def __init__(self, max_retries: int = 3):
        self._max_retries = max_retries
    
    async def retry(
        self,
        llm: Any,  # LLMClient or ILLMProvider
        original_prompt: str,
        failed_output: str,
        error_message: str,
        schema: Type[BaseModel],
        attempt_number: int,
    ) -> Optional[Dict[str, Any]]:
        """Refine prompt based on retry attempt."""
        
        if attempt_number == 1:
            # First retry: Add stricter JSON instruction
            refined_prompt = original_prompt + (
                "\n\n⚠️ IMPORTANT: Return ONLY valid JSON, no explanation, no markdown."
            )
        elif attempt_number == 2:
            # Second retry: Even more explicit
            refined_prompt = original_prompt + (
                "\n\n⚠️ CRITICAL: Output must be ONLY a valid JSON object. "
                "No code blocks (```json), no extra text before/after, just raw JSON."
            )
        else:
            # Further retries: Include the error
            refined_prompt = original_prompt + (
                f"\n\nPrevious attempt failed with error: {error_message}"
                "\n\nTry again. Output ONLY raw JSON with no extra text."
            )
        
        logger.info(f"[RetryStrategy] Extraction retry {attempt_number}: Refined prompt")
        
        try:
            response = await llm.generate(
                system="You are a JSON extraction assistant. Output ONLY valid JSON.",
                user=refined_prompt,
            )
            
            # Try to parse the response
            # First, extract JSON if in code blocks
            import re
            json_block = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", response, re.DOTALL)
            if json_block:
                response = json_block.group(1)
            
            # Try to parse
            parsed = json.loads(response)
            
            # Validate against schema
            schema.model_validate(parsed)
            
            logger.info(f"[RetryStrategy] Extraction retry {attempt_number}: Success!")
            return parsed
        
        except Exception as e:
            logger.warning(f"[RetryStrategy] Extraction retry {attempt_number} failed: {e}")
            return None
    
    @property
    def name(self) -> str:
        return "PromptRefinement"
    
    @property
    def max_retries(self) -> int:
        return self._max_retries


class LLMAssistedRepairRetry(IRetryStrategy):
    """
    Retry by asking LLM to repair its own broken JSON.
    
    Sends:
      "This JSON is broken. Fix it and return only valid JSON: {broken_json}"
    """
    
    def __init__(self, max_retries: int = 2):
        self._max_retries = max_retries
    
    async def retry(
        self,
        llm: Any,  # LLMClient or ILLMProvider
        original_prompt: str,
        failed_output: str,
        error_message: str,
        schema: Type[BaseModel],
        attempt_number: int,
    ) -> Optional[Dict[str, Any]]:
        """Ask LLM to repair the broken JSON."""
        
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        
        repair_prompt = (
            f"The following JSON is broken or invalid:\n\n{failed_output}\n\n"
            f"Error: {error_message}\n\n"
            f"Repair it to match this schema (return ONLY valid JSON):\n{schema_json}"
        )
        
        logger.info(f"[RetryStrategy] Extraction repair {attempt_number}: LLM-assisted")
        
        try:
            response = await llm.generate(
                system="You are a JSON repair expert. Return ONLY valid JSON.",
                user=repair_prompt,
            )
            
            # Extract JSON if in code blocks
            import re
            json_block = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", response, re.DOTALL)
            if json_block:
                response = json_block.group(1)
            
            # Parse and validate
            parsed = json.loads(response)
            schema.model_validate(parsed)
            
            logger.info(f"[RetryStrategy] Extraction repair {attempt_number}: Success!")
            return parsed
        
        except Exception as e:
            logger.warning(f"[RetryStrategy] Extraction repair {attempt_number} failed: {e}")
            return None
    
    @property
    def name(self) -> str:
        return "LLMAssistedRepair"
    
    @property
    def max_retries(self) -> int:
        return self._max_retries


class CompositeRetryStrategy(IRetryStrategy):
    """
    Combines multiple strategies in sequence.
    
    Tries:
      1. PromptRefinement (up to 2 retries)
      2. LLMAssistedRepair (up to 2 retries)
      3. Fallback: Return None
    
    This is the recommended default for production use.
    """
    
    def __init__(
        self,
        refinement_strategy: PromptRefinementRetry = None,
        repair_strategy: LLMAssistedRepairRetry = None,
    ):
        self.refinement_strategy = refinement_strategy or PromptRefinementRetry(max_retries=2)
        self.repair_strategy = repair_strategy or LLMAssistedRepairRetry(max_retries=2)
        self._retry_count = 0
    
    async def retry(
        self,
        llm: Any,  # LLMClient or ILLMProvider
        original_prompt: str,
        failed_output: str,
        error_message: str,
        schema: Type[BaseModel],
        attempt_number: int,
    ) -> Optional[Dict[str, Any]]:
        """Try refinement first, then repair."""
        
        # Try prompt refinement for first 2 attempts
        if attempt_number <= self.refinement_strategy.max_retries:
            result = await self.refinement_strategy.retry(
                llm=llm,
                original_prompt=original_prompt,
                failed_output=failed_output,
                error_message=error_message,
                schema=schema,
                attempt_number=attempt_number,
            )
            if result is not None:
                return result
        
        # Try LLM-assisted repair for next attempts
        if attempt_number <= (self.refinement_strategy.max_retries + self.repair_strategy.max_retries):
            repair_attempt = attempt_number - self.refinement_strategy.max_retries
            result = await self.repair_strategy.retry(
                llm=llm,
                original_prompt=original_prompt,
                failed_output=failed_output,
                error_message=error_message,
                schema=schema,
                attempt_number=repair_attempt,
            )
            if result is not None:
                return result
        
        # All retries exhausted
        logger.warning(f"[RetryStrategy] All retry strategies exhausted after {attempt_number} attempts")
        return None
    
    @property
    def name(self) -> str:
        return "CompositeRetry (PromptRefinement → LLMRepair)"
    
    @property
    def max_retries(self) -> int:
        return self.refinement_strategy.max_retries + self.repair_strategy.max_retries

