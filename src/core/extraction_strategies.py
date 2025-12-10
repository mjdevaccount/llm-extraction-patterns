"""
Extraction Strategies - Core Module

SOLID-compliant strategy pattern implementations for JSON repair and validation.
These are async, schema-aware strategies for advanced extraction workflows.

For simple synchronous JSON repair utilities, see json_repair.py
"""

import json
import re
import logging
from typing import Any, Dict, Optional, Type, Tuple

from pydantic import BaseModel, ValidationError

# Import abstractions from core
from .abstractions import IJSONRepairStrategy, IValidationStrategy, ILLMProvider

logger = logging.getLogger(__name__)


# ============================================================================
# JSON Repair Strategies
# ============================================================================

class IncrementalRepairStrategy(IJSONRepairStrategy):
    """
    Mechanical JSON repair: close braces, remove trailing commas.
    
    Single Responsibility: Fix common JSON syntax errors.
    """
    
    @property
    def name(self) -> str:
        return "incremental_repair"
    
    async def repair(
        self, 
        json_text: str, 
        schema: Type[BaseModel]
    ) -> Optional[Dict[str, Any]]:
        """Attempt incremental repair of JSON."""
        try:
            # Remove trailing commas before } or ]
            json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
            
            # Close unclosed braces/brackets
            open_braces = json_text.count('{') - json_text.count('}')
            open_brackets = json_text.count('[') - json_text.count(']')
            
            json_text += '}' * open_braces
            json_text += ']' * open_brackets
            
            # Try to parse
            data = json.loads(json_text)
            return data
        except Exception as e:
            logger.debug(f"Incremental repair failed: {e}")
            return None


class LLMRepairStrategy(IJSONRepairStrategy):
    """
    LLM-based JSON repair: ask LLM to fix JSON.
    
    Single Responsibility: Use LLM to semantically repair JSON.
    """
    
    def __init__(self, llm: ILLMProvider):
        self.llm = llm
    
    @property
    def name(self) -> str:
        return "llm_repair"
    
    async def repair(
        self, 
        json_text: str, 
        schema: Type[BaseModel]
    ) -> Optional[Dict[str, Any]]:
        """Attempt LLM-based repair of JSON."""
        try:
            prompt = (
                f"The following JSON is malformed. Fix it and return ONLY valid JSON:\n\n"
                f"{json_text}\n\n"
                f"Expected schema: {schema.__name__}\n"
                f"Return only the corrected JSON, no explanation."
            )
            
            from langchain_core.messages import HumanMessage
            response = await self.llm.invoke([HumanMessage(content=prompt)])
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data
            
            return None
        except Exception as e:
            logger.debug(f"LLM repair failed: {e}")
            return None


class RegexRepairStrategy(IJSONRepairStrategy):
    """
    Regex-based extraction: extract fields using regex patterns.
    
    Single Responsibility: Extract data using regex fallback.
    """
    
    @property
    def name(self) -> str:
        return "regex_repair"
    
    async def repair(
        self, 
        json_text: str, 
        schema: Type[BaseModel]
    ) -> Optional[Dict[str, Any]]:
        """Attempt regex-based extraction."""
        try:
            data = {}
            
            # Get schema fields
            if hasattr(schema, 'model_fields'):
                schema_fields = schema.model_fields
            elif hasattr(schema, '__fields__'):
                schema_fields = schema.__fields__
            else:
                return None
            
            combined_text = json_text.lower()
            
            # Extract each field
            for field_name, field_info in schema_fields.items():
                field_type = None
                if hasattr(field_info, 'annotation'):
                    field_type = field_info.annotation
                elif hasattr(field_info, 'type_'):
                    field_type = field_info.type_
                
                # Try to extract based on type
                if field_type == int:
                    pattern = rf'{field_name}[^:]*[:\s]*(\d+)'
                    match = re.search(pattern, combined_text, re.IGNORECASE)
                    if match:
                        data[field_name] = int(match.group(1))
                elif field_type == float:
                    pattern = rf'{field_name}[^:]*[:\s]*(\d+\.?\d*)'
                    match = re.search(pattern, combined_text, re.IGNORECASE)
                    if match:
                        data[field_name] = float(match.group(1))
                else:
                    pattern = rf'{field_name}[^:]*[:\s]*["\']?([^"\',\s}}]+)["\']?'
                    match = re.search(pattern, combined_text, re.IGNORECASE)
                    if match:
                        data[field_name] = match.group(1).strip()
            
            return data if data else None
        except Exception as e:
            logger.debug(f"Regex repair failed: {e}")
            return None


# ============================================================================
# Validation Strategies
# ============================================================================

class StrictValidationStrategy(IValidationStrategy):
    """
    Strict validation: fail fast on any error.
    
    Single Responsibility: Enforce strict validation with no repair.
    """
    
    @property
    def mode_name(self) -> str:
        return "strict"
    
    async def validate(
        self,
        data: Dict[str, Any],
        schema: Type[BaseModel],
        validation_rules: Dict[str, Any],
        llm: Optional[ILLMProvider] = None,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """Validate strictly - no repair."""
        try:
            validated = schema(**data)
            return {
                "validated": validated.model_dump(),
                "warnings": [],
                "repairs": {},
                "outcome": "strict_pass",
            }
        except ValidationError as e:
            # Re-raise with better message
            error_msg = f"Strict validation failed: {e}"
            raise ValueError(error_msg) from e


class RetryValidationStrategy(IValidationStrategy):
    """
    Retry validation: use LLM repair loop with error feedback.
    
    Single Responsibility: Implement LLM-based repair with retries.
    """
    
    @property
    def mode_name(self) -> str:
        return "retry"
    
    async def validate(
        self,
        data: Dict[str, Any],
        schema: Type[BaseModel],
        validation_rules: Dict[str, Any],
        llm: Optional[ILLMProvider] = None,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """Validate with LLM repair retries."""
        if llm is None:
            raise ValueError("RETRY mode requires llm parameter")
        
        last_error = None
        for attempt in range(max_retries):
            try:
                validated = schema(**data)
                return {
                    "validated": validated.model_dump(),
                    "warnings": [f"Repaired after {attempt} attempt(s)"] if attempt > 0 else [],
                    "repairs": {},
                    "outcome": f"repaired_after_{attempt}_retries" if attempt > 0 else "strict_pass",
                }
            except ValidationError as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Try LLM repair
                    data = await self._repair_with_llm(data, e, schema, llm)
        
        # Re-raise with better message
        error_msg = f"Validation failed after {max_retries} retries: {last_error}"
        raise ValueError(error_msg) from last_error
    
    async def _repair_with_llm(
        self,
        data: Dict[str, Any],
        error: ValidationError,
        schema: Type[BaseModel],
        llm: ILLMProvider,
    ) -> Dict[str, Any]:
        """Repair data using LLM with validation error feedback."""
        error_msg = self._format_errors(error)
        prompt = (
            f"Fix this JSON to match the schema. Validation errors:\n{error_msg}\n\n"
            f"Current data: {json.dumps(data, indent=2)}\n\n"
            f"Schema: {schema.__name__}\n\n"
            f"Return ONLY valid JSON matching the schema."
        )
        
        from langchain_core.messages import HumanMessage
        response = await llm.invoke([HumanMessage(content=prompt)])
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        
        return data
    
    def _format_errors(self, error: ValidationError) -> str:
        """Format validation errors for LLM."""
        errors = []
        for err in error.errors():
            field = ".".join(str(loc) for loc in err.get("loc", ["unknown"]))
            msg = err.get("msg", "")
            errors.append(f"{field}: {msg}")
        return "\n".join(errors)


class BestEffortValidationStrategy(IValidationStrategy):
    """
    Best-effort validation: apply safe mechanical repairs.
    
    Single Responsibility: Apply schema-driven repairs without LLM.
    """
    
    @property
    def mode_name(self) -> str:
        return "best_effort"
    
    async def validate(
        self,
        data: Dict[str, Any],
        schema: Type[BaseModel],
        validation_rules: Dict[str, Any],
        llm: Optional[ILLMProvider] = None,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """Validate with best-effort repairs."""
        repairs = {}
        warnings = []
        
        # Clean extraction artifacts
        if "extraction_status" in data and len(data) == 1:
            warnings.append("Extraction returned only status marker")
            data = {}
        
        # Remove undefined/null values
        data = {k: v for k, v in data.items() 
               if k != "extraction_status" and v is not None}
        
        # Try validation with repairs
        for attempt in range(2):
            try:
                validated = schema(**data)
                return {
                    "validated": validated.model_dump(),
                    "warnings": warnings,
                    "repairs": repairs,
                    "outcome": f"repaired_{attempt}_times" if attempt > 0 else "strict_pass",
                }
            except ValidationError as e:
                if attempt < 1:
                    data, new_repairs = self._repair_fields_safe(data, e, schema)
                    repairs.update(new_repairs)
                else:
                    # Re-raise with better message
                    error_msg = f"Validation failed after best-effort repair: {e}"
                    raise ValueError(error_msg) from e
        
        return {
            "validated": data,
            "warnings": warnings,
            "repairs": repairs,
            "outcome": "repaired_twice",
        }
    
    def _repair_fields_safe(
        self,
        data: Dict[str, Any],
        error: ValidationError,
        schema: Type[BaseModel],
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Apply safe, schema-driven repairs."""
        repairs = {}
        
        # Get schema fields
        if hasattr(schema, 'model_fields'):
            schema_fields = schema.model_fields
        else:
            return data, repairs
        
        # Apply repairs based on validation errors
        for err in error.errors():
            field_path = err.get("loc", [])
            if not field_path:
                continue
            
            field_name = str(field_path[0])
            error_type = err.get("type", "")
            
            # Type coercion
            if error_type == "type_error" and field_name in data:
                value = data[field_name]
                field_info = schema_fields.get(field_name)
                if field_info:
                    field_type = getattr(field_info, 'annotation', None)
                    try:
                        if field_type == int:
                            data[field_name] = int(float(str(value)))
                            repairs[field_name] = "coerced_to_int"
                        elif field_type == float:
                            data[field_name] = float(str(value))
                            repairs[field_name] = "coerced_to_float"
                    except (ValueError, TypeError):
                        pass
            
            # Use schema defaults
            if error_type == "missing" and field_name in schema_fields:
                field_info = schema_fields[field_name]
                if hasattr(field_info, 'default') and field_info.default is not None:
                    data[field_name] = field_info.default
                    repairs[field_name] = "used_schema_default"
        
        return data, repairs

