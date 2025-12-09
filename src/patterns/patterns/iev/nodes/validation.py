"""Validation Node for SOLID Design.

Single Responsibility: Validate extracted data with configurable repair strategies.
"""

import json
import re
import logging
from typing import Any, Dict, Optional, Type, Callable, Tuple
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ValidationError

from langchain_core.messages import HumanMessage

from .base_node import BaseNode, NodeExecutionError, NodeStatus
from ..abstractions import ILLMProvider
from ..strategies import (
    StrictValidationStrategy,
    RetryValidationStrategy,
    BestEffortValidationStrategy,
)
from ..llm_adapter import LangChainLLMAdapter
from .llm_adapter import invoke_llm, is_llm_client

logger = logging.getLogger(__name__)


class ValidationMode(str, Enum):
    """
    Validation mode for ValidationNode.
    
    STRICT: Fail fast on Pydantic error (no automatic repair)
    RETRY: Use LLM repair loop with validation error feedback
    BEST_EFFORT: Allow local defaults/fixes (for non-critical pipelines)
    """
    STRICT = "strict"
    RETRY = "retry"
    BEST_EFFORT = "best_effort"


class ValidationNode(BaseNode):
    """
    Semantic validation and repair node (December 2025 Pattern).
    
    Single Responsibility: Validate extracted data with configurable repair strategies.
    
    Temperature: 0.0 (strict enforcement) or LLM temperature for repair loops
    
    Input Requirements:
        - state["extracted"]: Dict from extraction node
    
    Output:
        - Adds `validated` key with validated/repaired data
        - Adds `validation_warnings` list
        - Adds `validation_metadata` dict with mode, outcome, repairs
    
    Validation Modes (December 2025 Best Practice):
        1. STRICT: Fail fast on Pydantic error (no automatic repair)
           - Use for: Critical pipelines (finance, healthcare, legal)
           - Behavior: Raises NodeExecutionError on validation failure
        
        2. RETRY: LLM repair loop with validation error feedback
           - Use for: Production pipelines needing semantic repair
           - Behavior: Re-prompts LLM with Pydantic errors, retries up to max_retries
           - Pattern: Matches BentoML, Haystack, Instructor-style validation
        
        3. BEST_EFFORT: Local defaults/fixes (mechanical repairs only)
           - Use for: Analytics, internal dashboards, non-critical contexts
           - Behavior: Applies safe type coercion, uses schema defaults, minimal guessing
    
    Validation Layers:
        1. Pydantic schema validation (types, constraints, enums)
        2. Custom semantic rules (business logic via validation_rules)
        3. Schema-driven repair (uses Field(ge=, le=, default=) from Pydantic)
    
    Design Pattern: Strategy
        Different validation modes can be swapped without changing node logic.
        Domain semantics live in Pydantic schemas and validation_rules, not in the node.
    
    Example (STRICT mode - recommended for production):
        validation = ValidationNode(
            output_schema=AdoptionPrediction,
            mode=ValidationMode.STRICT,
            validation_rules={
                "timeline_sanity": lambda x: 1 <= x["adoption_timeline_months"] <= 60,
            }
        )
    
    Example (RETRY mode - with LLM repair):
        validation = ValidationNode(
            output_schema=AdoptionPrediction,
            mode=ValidationMode.RETRY,
            llm=repair_llm,  # Low-temperature LLM for repair
            max_retries=2,
        )
    
    Example (BEST_EFFORT mode - for analytics):
        validation = ValidationNode(
            output_schema=AdoptionPrediction,
            mode=ValidationMode.BEST_EFFORT,
        )
    """
    
    def __init__(
        self,
        output_schema: Type[BaseModel],
        validation_rules: Dict[str, Callable] = None,
        mode: ValidationMode = ValidationMode.STRICT,
        llm: Any = None,  # Optional LLM for RETRY mode repair loop
        max_retries: int = 2,  # Max retries for RETRY mode
        name: str = "validation",
        description: str = "Validate and repair extracted data",
    ):
        """
        Initialize validation node.
        
        Args:
            output_schema: Pydantic model to validate against
                Should use Field(ge=, le=, default=) for constraints
            validation_rules: Dict of {rule_name: rule_func}
                where rule_func(data) returns True if valid, raises/returns False otherwise
            mode: ValidationMode enum (STRICT, RETRY, BEST_EFFORT)
            llm: Optional LLMClient (cloud) or LangChain ChatModel (local/brittle) for RETRY mode repair loop
                Required if mode=ValidationMode.RETRY
                Note: RETRY mode works best with LangChain ChatModel for local/brittle LLMs
            max_retries: Maximum retry attempts for RETRY mode (default: 2)
            name: Node identifier
            description: Human-readable description
        
        Example of validation_rules:
            {
                "timeline_bounds": lambda x: 1 <= x["adoption_timeline_months"] <= 60,
                "disruption_positive": lambda x: x["disruption_score"] >= 0,
            }
        
        Note:
            For STRICT mode, validation failures raise NodeExecutionError immediately.
            For RETRY mode, validation errors are fed back to LLM for repair.
            For BEST_EFFORT mode, only safe mechanical repairs are applied.
        """
        super().__init__(name=name, description=description)
        self.output_schema = output_schema
        self.validation_rules = validation_rules or {}
        self.mode = mode
        
        self.llm = llm
        self.max_retries = max_retries
        
        # Validate mode requirements
        if self.mode == ValidationMode.RETRY and llm is None:
            logger.warning(
                f"[{self.name}] RETRY mode requires llm parameter. "
                "Falling back to STRICT mode behavior."
            )
            self.mode = ValidationMode.STRICT
        
        # SOLID Refactoring: Initialize validation strategy
        # Only create adapter if not LLMClient (for local/brittle LLMs)
        if llm and not is_llm_client(llm):
            llm_adapter = LangChainLLMAdapter(llm)
        else:
            llm_adapter = None
        
        strategy_map = {
            ValidationMode.STRICT: StrictValidationStrategy(),
            ValidationMode.RETRY: RetryValidationStrategy(llm_adapter) if llm_adapter else StrictValidationStrategy(),
            ValidationMode.BEST_EFFORT: BestEffortValidationStrategy(),
        }
        self._validation_strategy = strategy_map.get(self.mode, StrictValidationStrategy())
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute validation based on configured mode.
        
        Process (varies by mode):
            STRICT:
                1. Validate with Pydantic schema
                2. Apply semantic rules
                3. Raise on any failure
        
            RETRY:
                1. Validate with Pydantic schema
                2. If fails, re-prompt LLM with error details
                3. Retry validation up to max_retries
                4. Apply semantic rules
        
            BEST_EFFORT:
                1. Validate with Pydantic schema
                2. If fails, apply safe mechanical repairs
                3. Use schema defaults for missing fields
                4. Apply semantic rules
        
        Args:
            state: Must have "extracted" key
        
        Returns:
            State with added keys:
                - validated: Validated/repaired data dict
                - validation_warnings: List of warnings/repairs applied
                - validation_metadata: Dict with mode, outcome, repairs
        
        Raises:
            NodeExecutionError: If validation fails (STRICT mode) or all retries exhausted (RETRY mode)
        """
        start_time = datetime.now()
        self.metrics.status = NodeStatus.RUNNING
        self.metrics.input_keys = ["extracted"]
        
        try:
            if "extracted" not in state:
                raise NodeExecutionError(
                    node_name=self.name,
                    reason="Missing 'extracted' from extraction node",
                    state=state
                )
            
            # Store original data for audit trail
            original_data = state["extracted"].copy()
            data = original_data.copy()
            warnings = []
            repairs = {}  # Track per-field repairs
            validation_outcome = "strict_pass"
            
            # Clean extraction artifacts
            if "extraction_status" in data and len(data) == 1:
                warnings.append("Extraction returned only status marker, using schema defaults")
                data = {}
            
            # Remove undefined/null values and status markers
            data = {k: v for k, v in data.items() 
                   if k != "extraction_status" and v is not None 
                   and not (hasattr(v, '__class__') and 'Undefined' in str(type(v)))}
            
            # Layer 1: Pydantic validation with mode-specific repair
            validated = None
            validation_error = None
            
            # SOLID Refactoring: Use strategy pattern
            # Adapter already created in __init__, just pass it
            llm_adapter = LangChainLLMAdapter(self.llm) if (self.llm and not is_llm_client(self.llm)) else None
            result = await self._validation_strategy.validate(
                data=data,
                schema=self.output_schema,
                validation_rules=self.validation_rules,
                llm=llm_adapter,
                max_retries=self.max_retries,
            )
            validated = self.output_schema(**result["validated"])
            warnings.extend(result.get("warnings", []))
            repairs = result.get("repairs", {})
            validation_outcome = result.get("outcome", "strict_pass")
            
            validated_dict = validated.model_dump()
            
            # Layer 2: Custom semantic rules
            for rule_name, rule_func in self.validation_rules.items():
                try:
                    result = rule_func(validated_dict)
                    if not result:
                        warnings.append(f"Semantic rule '{rule_name}' returned False")
                except Exception as e:
                    warnings.append(f"Semantic rule '{rule_name}' raised: {e}")
            
            # Store results with metadata
            state["validated"] = validated_dict
            if warnings:
                state["validation_warnings"] = warnings
                self.metrics.warnings = warnings
            
            # Add validation metadata (December 2025 best practice)
            state["validation_metadata"] = {
                "mode": self.mode.value,
                "outcome": validation_outcome,
                "repairs": repairs,
                "original_keys": list(original_data.keys()),
                "validated_keys": list(validated_dict.keys()),
            }
            
            # Update metrics
            self.metrics.output_keys = ["validated", "validation_metadata"]
            self.metrics.status = NodeStatus.SUCCESS
            
            logger.info(
                f"[{self.name}] Validation complete (mode: {self.mode.value}, outcome: {validation_outcome})" + (
                    f" with {len(warnings)} warnings" if warnings else ""
                )
            )
            
            return state
        
        except Exception as e:
            self.metrics.status = NodeStatus.FAILED
            self.metrics.error_message = str(e)
            raise await self.on_error(e, state)
        
        finally:
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.duration_ms = elapsed
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        return "extracted" in state
    
    def _format_validation_errors(self, error: ValidationError) -> str:
        """Format Pydantic validation errors for human-readable messages."""
        errors = []
        for err in error.errors():
            field = ".".join(str(loc) for loc in err.get("loc", ["unknown"]))
            error_type = err.get("type", "unknown")
            msg = err.get("msg", "")
            errors.append(f"{field}: {error_type} - {msg}")
        return "; ".join(errors)

