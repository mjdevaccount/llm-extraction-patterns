"""JSON Repair Strategies for SOLID Design.

Single Responsibility: Each strategy fixes JSON in one specific way.
Open/Closed: New strategies can be added without modifying existing code.
"""

import json
import re
import logging
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from ..abstractions import IJSONRepairStrategy, ILLMProvider

logger = logging.getLogger(__name__)


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

