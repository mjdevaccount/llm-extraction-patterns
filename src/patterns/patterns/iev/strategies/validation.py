"""Validation Strategies for SOLID Design.

Single Responsibility: Each strategy validates in one specific way.
Open/Closed: New validation modes can be added without modifying existing code.
"""

import json
import re
import logging
from typing import Any, Dict, Optional, Type, Tuple

from pydantic import BaseModel, ValidationError

from ..abstractions import IValidationStrategy, ILLMProvider

logger = logging.getLogger(__name__)


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
            raise ValidationError(
                f"Strict validation failed: {e}",
                model=schema
            )


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
        
        raise ValidationError(
            f"Validation failed after {max_retries} retries: {last_error}",
            model=schema
        )
    
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
                    raise ValidationError(
                        f"Validation failed after best-effort repair: {e}",
                        model=schema
                    )
        
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

