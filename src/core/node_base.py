"""
Node Base Classes and Utilities

Reusable base classes that handle common node patterns:
- LLM invocation (both sync LLMClient and async LangChain)
- State error propagation
- Timing/metrics
- Logging
- JSON extraction and repair

New nodes should extend these bases to avoid rewriting boilerplate.
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ValidationError

from .json_repair import extract_json_object, repair_json
from .llm_helpers import invoke_llm
from .types import NodeStatus, NodeMetrics, NodeExecutionError

logger = logging.getLogger(__name__)


# =============================================================================
# STATUS CONSTANTS - Use these instead of string comparisons
# =============================================================================

class NodeResult(str, Enum):
    """Standard node result statuses."""
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class VerificationStatus(str, Enum):
    """Standard verification statuses."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ValidationMode(str, Enum):
    """
    Validation mode for validation nodes.
    
    STRICT: Fail fast on Pydantic error (no automatic repair)
    RETRY: Use LLM repair loop with validation error feedback
    BEST_EFFORT: Allow local defaults/fixes (for non-critical pipelines)
    """
    STRICT = "strict"
    RETRY = "retry"
    BEST_EFFORT = "best_effort"


# =============================================================================
# RESPONSE EXTRACTION - Unified JSON extraction from LLM responses
# =============================================================================

def extract_json_from_response(
    response_text: str,
    fallback_to_raw: bool = True,
) -> Optional[str]:
    """
    Extract JSON from LLM response text.
    
    Tries in order:
    1. Direct JSON object extraction
    2. Code block extraction (```json ... ```)
    3. Raw text (if fallback_to_raw)
    
    Args:
        response_text: Raw LLM response
        fallback_to_raw: If True, return raw text if no JSON found
    
    Returns:
        JSON string or None
    """
    # Try direct extraction first
    json_obj = extract_json_object(response_text)
    if json_obj:
        return json_obj
    
    # Try code block extraction
    json_block = re.search(
        r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
        response_text,
        re.DOTALL
    )
    if json_block:
        return json_block.group(1)
    
    # Fallback to raw
    if fallback_to_raw:
        return response_text.strip()
    
    return None


def extract_and_repair_json(
    response_text: str,
) -> Optional[Dict[str, Any]]:
    """
    Extract and repair JSON from LLM response in one step.
    
    Args:
        response_text: Raw LLM response
    
    Returns:
        Parsed dict or None
    """
    json_str = extract_json_from_response(response_text)
    if json_str is None:
        return None
    return repair_json(json_str)


def parse_with_schema(
    data: Union[str, Dict[str, Any]],
    schema: Type[BaseModel],
) -> BaseModel:
    """
    Parse data with Pydantic schema.
    
    Args:
        data: JSON string or dict
        schema: Pydantic model class
    
    Returns:
        Validated Pydantic model
    
    Raises:
        ValidationError: If validation fails
    """
    if isinstance(data, str):
        parsed = extract_and_repair_json(data)
        if parsed is None:
            raise ValueError(f"Could not parse JSON from: {data[:200]}")
        data = parsed
    
    return schema.model_validate(data)


# =============================================================================
# STATE UTILITIES - Common state operations
# =============================================================================

def check_state_error(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Check if state has an error from previous node.
    
    Use at start of execute() to propagate errors.
    
    Args:
        state: Current state dict
    
    Returns:
        Error dict if error present, None otherwise
    """
    if state.get("error"):
        return {"error": state["error"]}
    return None


def require_state_keys(
    state: Dict[str, Any],
    required_keys: List[str],
    node_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Check that required keys exist in state.
    
    Args:
        state: Current state dict
        required_keys: Keys that must exist
        node_name: Node name for error message
    
    Returns:
        Error dict if missing keys, None otherwise
    """
    missing = [k for k in required_keys if k not in state]
    if missing:
        return {
            "error": f"[{node_name}] Missing required keys: {missing}"
        }
    return None


# =============================================================================
# TIMING UTILITIES
# =============================================================================

@dataclass
class NodeTimer:
    """Context manager for timing node execution."""
    node_name: str
    _start: float = field(default=0.0, init=False)
    duration_ms: float = field(default=0.0, init=False)
    
    def __enter__(self):
        self._start = time.time()
        logger.info(f"[{self.node_name}] Starting")
        return self
    
    def __exit__(self, *args):
        self.duration_ms = (time.time() - self._start) * 1000
        logger.info(f"[{self.node_name}] Complete ({self.duration_ms:.1f}ms)")


# =============================================================================
# PYDANTIC UTILITIES - Error formatting, validation helpers
# =============================================================================

def format_validation_errors(error: ValidationError) -> str:
    """
    Format Pydantic validation errors for human-readable messages.
    
    Args:
        error: Pydantic ValidationError
    
    Returns:
        Formatted error string
    """
    errors = []
    for err in error.errors():
        field = ".".join(str(loc) for loc in err.get("loc", ["unknown"]))
        error_type = err.get("type", "unknown")
        msg = err.get("msg", "")
        errors.append(f"{field}: {error_type} - {msg}")
    return "; ".join(errors)


def get_schema_defaults(schema: Type[BaseModel]) -> Dict[str, Any]:
    """
    Extract default values from a Pydantic schema.
    
    Args:
        schema: Pydantic model class
    
    Returns:
        Dict of field_name -> default_value
    """
    defaults = {}
    for name, field_info in schema.model_fields.items():
        if field_info.default is not None:
            defaults[name] = field_info.default
        elif field_info.default_factory is not None:
            defaults[name] = field_info.default_factory()
    return defaults


# =============================================================================
# ABSTRACT BASE NODE - Extend this for new nodes
# =============================================================================

class LLMNode(ABC):
    """
    Abstract base for nodes that call LLMs.
    
    Handles:
    - LLM invocation (both LLMClient and LangChain)
    - Error propagation
    - Timing/logging
    - State validation
    - Metrics collection
    
    Subclasses implement:
    - execute_impl(): Core logic
    - required_keys: List of required state keys
    
    Example:
        class MyNode(LLMNode):
            required_keys = ["input"]
            
            async def execute_impl(self, state: Dict) -> Dict:
                response = await self.invoke("Analyze: {input}", state)
                return {"analysis": response}
    """
    
    required_keys: List[str] = []
    output_keys: List[str] = []
    
    def __init__(
        self,
        llm: Any,
        name: str,
        system_prompt: str = "You are a helpful assistant.",
        description: str = "",
    ):
        """
        Initialize LLM node.
        
        Args:
            llm: LLMClient or LangChain ChatModel
            name: Node identifier for logging
            system_prompt: System prompt for LLM
            description: Human-readable description
        """
        self.llm = llm
        self.name = name
        self.system_prompt = system_prompt
        self.description = description or f"{name} node"
        self.metrics = NodeMetrics(name=name)
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute node with standard error handling, timing, and metrics.
        
        Subclasses should NOT override this. Override execute_impl instead.
        """
        start_time = datetime.now()
        self.metrics.status = NodeStatus.RUNNING
        self.metrics.input_keys = self.required_keys
        
        try:
            # Check for error from previous node
            if state.get("error"):
                logger.warning(f"[{self.name}] Skipped due to previous error")
                self.metrics.status = NodeStatus.FAILED
                return {"error": state["error"]}
            
            # Check required keys
            missing = [k for k in self.required_keys if k not in state]
            if missing:
                raise NodeExecutionError(
                    node_name=self.name,
                    reason=f"Missing required keys: {missing}",
                    state=state,
                )
            
            # Execute core logic
            result = await self.execute_impl(state)
            
            # Update state
            state.update(result)
            
            # Update metrics
            self.metrics.output_keys = self.output_keys or list(result.keys())
            self.metrics.status = NodeStatus.SUCCESS
            
            return state
            
        except Exception as e:
            self.metrics.status = NodeStatus.FAILED
            self.metrics.error_message = str(e)
            logger.exception(f"[{self.name}] Error: {e}")
            raise await self.on_error(e, state)
        
        finally:
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.duration_ms = elapsed
            logger.info(f"[{self.name}] Complete ({elapsed:.1f}ms)")
    
    @abstractmethod
    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core node logic. Override in subclasses.
        
        Args:
            state: Validated state with required keys
        
        Returns:
            Dict with results to merge into state
        """
        pass
    
    async def on_error(self, error: Exception, state: Dict[str, Any]) -> Exception:
        """
        Handle errors during execution. Override for custom handling.
        
        Args:
            error: The exception that occurred
            state: Current state
        
        Returns:
            Exception to raise
        """
        logger.error(
            f"[{self.name}] Error: {error}\n"
            f"State keys: {list(state.keys())}"
        )
        return error
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for this node."""
        return self.metrics.to_dict()
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate that state has all required keys."""
        return all(k in state for k in self.required_keys)
    
    async def invoke(
        self,
        prompt_template: str,
        state: Dict[str, Any],
        system: Optional[str] = None,
    ) -> str:
        """
        Invoke LLM with prompt template.
        
        Args:
            prompt_template: Template with {key} placeholders
            state: State dict for template values
            system: Override system prompt
        
        Returns:
            LLM response text
        """
        from langchain_core.messages import HumanMessage
        
        prompt = prompt_template.format(**state)
        messages = [HumanMessage(content=prompt)]
        
        return await invoke_llm(
            self.llm,
            messages,
            system=system or self.system_prompt
        )
    
    async def invoke_and_extract_json(
        self,
        prompt_template: str,
        state: Dict[str, Any],
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Invoke LLM and extract/repair JSON from response.
        
        Args:
            prompt_template: Template with {key} placeholders
            state: State dict for template values
            system: Override system prompt
        
        Returns:
            Parsed JSON dict
        
        Raises:
            ValueError: If JSON extraction fails
        """
        response = await self.invoke(prompt_template, state, system)
        result = extract_and_repair_json(response)
        if result is None:
            raise ValueError(f"Could not extract JSON from: {response[:200]}")
        return result
    
    async def invoke_and_parse(
        self,
        prompt_template: str,
        state: Dict[str, Any],
        schema: Type[BaseModel],
        system: Optional[str] = None,
    ) -> BaseModel:
        """
        Invoke LLM and parse response with Pydantic schema.
        
        Args:
            prompt_template: Template with {key} placeholders
            state: State dict for template values
            schema: Pydantic model class
            system: Override system prompt
        
        Returns:
            Validated Pydantic model
        
        Raises:
            ValidationError: If validation fails
        """
        response = await self.invoke(prompt_template, state, system)
        return parse_with_schema(response, schema)


# =============================================================================
# INTELLIGENCE NODE BASE - For free-form reasoning nodes
# =============================================================================

class IntelligenceNodeBase(LLMNode):
    """
    Base for nodes that generate free-form analysis.
    
    Handles:
    - Prompt formatting from state
    - Message history management
    
    Subclasses can override:
    - build_prompt(): Custom prompt construction
    """
    
    output_keys = ["analysis"]
    
    def __init__(
        self,
        llm: Any,
        prompt_template: str,
        required_state_keys: List[str] = None,
        name: str = "intelligence",
        description: str = "Free-form reasoning phase",
    ):
        """
        Initialize intelligence node.
        
        Args:
            llm: LLMClient or LangChain ChatModel (high temperature)
            prompt_template: Template with {key} placeholders
            required_state_keys: Keys that must exist in state
            name: Node identifier
            description: Human-readable description
        """
        super().__init__(
            llm=llm,
            name=name,
            system_prompt="You are a helpful assistant that analyzes information.",
            description=description,
        )
        self.prompt_template = prompt_template
        self.required_keys = required_state_keys or []
    
    def build_prompt(self, state: Dict[str, Any]) -> str:
        """
        Build prompt from template and state. Override for custom logic.
        
        Args:
            state: Current state with required keys
        
        Returns:
            Formatted prompt string
        """
        format_dict = {k: state.get(k) for k in self.required_keys}
        return self.prompt_template.format(**format_dict)
    
    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute intelligence analysis."""
        prompt = self.build_prompt(state)
        
        from langchain_core.messages import HumanMessage, AIMessage
        
        messages = state.get("messages", [])
        messages.append(HumanMessage(content=prompt))
        
        response_text = await invoke_llm(
            self.llm,
            messages,
            system=self.system_prompt
        )
        
        # Update message history
        if hasattr(self.llm, 'ainvoke'):
            messages.append(AIMessage(content=response_text))
        
        logger.info(f"[{self.name}] Analysis: {len(response_text)} chars")
        
        return {
            "analysis": response_text,
            "messages": messages,
        }


# =============================================================================
# EXTRACTION NODE BASE - For nodes that extract structured data
# =============================================================================

class ExtractionNodeBase(LLMNode):
    """
    Base for nodes that extract structured data from text.
    
    Handles:
    - JSON extraction and repair
    - Pydantic validation
    - Retry logic
    - with_structured_output() support (LangChain)
    
    Subclasses can override:
    - build_prompt(): Custom prompt construction
    """
    
    required_keys = ["analysis"]
    output_keys = ["extracted"]
    
    def __init__(
        self,
        llm: Any,
        output_schema: Union[Type[BaseModel], Dict[str, Any]],
        prompt_template: Optional[str] = None,
        name: str = "extraction",
        max_retries: int = 3,
        description: str = "Extract structured data from analysis",
    ):
        """
        Initialize extraction node.
        
        Args:
            llm: LLMClient or LangChain ChatModel (low temperature)
            output_schema: Pydantic model or dict template
            prompt_template: Optional custom prompt template
            name: Node identifier
            max_retries: Max extraction attempts
            description: Human-readable description
        """
        super().__init__(
            llm=llm,
            name=name,
            system_prompt="You are a helpful assistant that extracts structured data.",
            description=description,
        )
        self.output_schema = output_schema
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self._is_pydantic = isinstance(output_schema, type) and issubclass(output_schema, BaseModel)
    
    def build_prompt(self, state: Dict[str, Any]) -> str:
        """
        Build extraction prompt. Override for custom prompts.
        
        Args:
            state: Current state with "analysis" key
        
        Returns:
            Prompt string
        """
        if self.prompt_template:
            return self.prompt_template.format(**state)
        
        analysis = state.get("analysis", "")[:1500]  # Limit context
        
        if self._is_pydantic:
            schema_json = self.output_schema.model_json_schema()
        else:
            schema_json = self.output_schema
        
        return (
            f"Extract structured data from the following text.\n\n"
            f"Text:\n{analysis}\n\n"
            f"Schema (return valid JSON only):\n{json.dumps(schema_json, indent=2)}"
        )
    
    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data with retry logic."""
        prompt = self.build_prompt(state)
        warnings = []
        
        # Try with_structured_output() first (LangChain only)
        if self._is_pydantic and hasattr(self.llm, 'with_structured_output'):
            try:
                from langchain_core.messages import HumanMessage
                structured_llm = self.llm.with_structured_output(self.output_schema)
                validated_model = await structured_llm.ainvoke([HumanMessage(content=prompt)])
                data = validated_model.model_dump() if hasattr(validated_model, 'model_dump') else dict(validated_model)
                logger.info(f"[{self.name}] Structured output extraction successful")
                return {"extracted": data, "extraction_attempts": 1}
            except Exception as e:
                logger.info(f"[{self.name}] with_structured_output failed: {e}, falling back to manual")
                warnings.append(f"with_structured_output failed: {e}")
        
        # Manual extraction with retries
        for attempt in range(self.max_retries):
            logger.info(f"[{self.name}] Attempt {attempt + 1}/{self.max_retries}")
            
            try:
                response = await self.invoke(prompt, {}, None)
                data = extract_and_repair_json(response)
                
                if data is None:
                    raise ValueError(f"Could not extract JSON from response")
                
                # Validate with Pydantic if schema is a model
                if self._is_pydantic:
                    parsed = self.output_schema.model_validate(data)
                    result = {"extracted": parsed.model_dump(), "extraction_attempts": attempt + 1}
                else:
                    result = {"extracted": data, "extraction_attempts": attempt + 1}
                
                if warnings:
                    result["extraction_warnings"] = warnings
                
                return result
            
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                logger.warning(f"[{self.name}] Attempt {attempt + 1} failed: {e}")
                warnings.append(f"Attempt {attempt + 1}: {e}")
                
                if attempt == self.max_retries - 1:
                    raise NodeExecutionError(
                        node_name=self.name,
                        reason=f"Extraction failed after {self.max_retries} attempts: {e}",
                        state=state,
                    )
        
        # Should not reach here
        raise NodeExecutionError(
            node_name=self.name,
            reason="Extraction loop exited unexpectedly",
            state=state,
        )


# =============================================================================
# VALIDATION NODE BASE - For nodes that validate/verify extracted data
# =============================================================================

class ValidationNodeBase(LLMNode):
    """
    Base for nodes that validate extracted data with configurable strategies.
    
    Handles:
    - Pydantic validation
    - Custom semantic rules
    - Approval/rejection logic
    - Validation metadata
    
    Subclasses can override:
    - build_prompt(): Custom verification prompt
    """
    
    required_keys = ["extracted"]
    output_keys = ["validated", "validation_metadata"]
    
    def __init__(
        self,
        output_schema: Type[BaseModel],
        validation_rules: Optional[Dict[str, Callable]] = None,
        mode: ValidationMode = ValidationMode.STRICT,
        llm: Any = None,  # Optional for RETRY mode
        max_retries: int = 2,
        name: str = "validation",
        description: str = "Validate and repair extracted data",
    ):
        """
        Initialize validation node.
        
        Args:
            output_schema: Pydantic model to validate against
            validation_rules: Dict of {rule_name: rule_func(data) -> bool}
            mode: ValidationMode (STRICT, RETRY, BEST_EFFORT)
            llm: Optional LLM for RETRY mode
            max_retries: Max retries for RETRY mode
            name: Node identifier
            description: Human-readable description
        """
        super().__init__(
            llm=llm,
            name=name,
            system_prompt="You are a helpful assistant that verifies data quality.",
            description=description,
        )
        self.output_schema = output_schema
        self.validation_rules = validation_rules or {}
        self.mode = mode
        self.max_retries = max_retries
        
        # Warn if RETRY mode without LLM
        if self.mode == ValidationMode.RETRY and llm is None:
            logger.warning(f"[{self.name}] RETRY mode requires llm. Falling back to STRICT.")
            self.mode = ValidationMode.STRICT
    
    async def execute_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted data."""
        extracted = state.get("extracted")
        
        if extracted is None:
            return {
                "verification_status": VerificationStatus.REJECTED,
                "error": "No extracted data to verify",
            }
        
        # Convert Pydantic to dict if needed
        if isinstance(extracted, BaseModel):
            data = extracted.model_dump()
        else:
            data = dict(extracted)
        
        warnings = []
        repairs = {}
        validation_outcome = "strict_pass"
        
        # Layer 1: Pydantic validation
        try:
            validated = self.output_schema.model_validate(data)
            validated_dict = validated.model_dump()
        except ValidationError as e:
            if self.mode == ValidationMode.STRICT:
                raise NodeExecutionError(
                    node_name=self.name,
                    reason=f"Validation failed: {format_validation_errors(e)}",
                    state=state,
                )
            elif self.mode == ValidationMode.BEST_EFFORT:
                # Apply defaults for missing fields
                defaults = get_schema_defaults(self.output_schema)
                for field, default in defaults.items():
                    if field not in data:
                        data[field] = default
                        repairs[field] = f"Applied default: {default}"
                        warnings.append(f"Applied default for {field}")
                
                # Try again
                validated = self.output_schema.model_validate(data)
                validated_dict = validated.model_dump()
                validation_outcome = "best_effort_repaired"
            else:
                # RETRY mode - would need LLM repair loop
                raise NodeExecutionError(
                    node_name=self.name,
                    reason=f"Validation failed: {format_validation_errors(e)}",
                    state=state,
                )
        
        # Layer 2: Custom semantic rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                result = rule_func(validated_dict)
                if not result:
                    warnings.append(f"Semantic rule '{rule_name}' returned False")
            except Exception as e:
                warnings.append(f"Semantic rule '{rule_name}' raised: {e}")
        
        result = {
            "validated": validated_dict,
            "verification_status": VerificationStatus.APPROVED,
            "validation_metadata": {
                "mode": self.mode.value,
                "outcome": validation_outcome,
                "repairs": repairs,
            },
        }
        
        if warnings:
            result["validation_warnings"] = warnings
            self.metrics.warnings = warnings
        
        return result


# Alias for backwards compatibility
VerificationNodeBase = ValidationNodeBase

