"""Node-based IEV implementation (alternative to LangGraph version).

This provides the original node-based API for those who prefer it.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Callable

from pydantic import BaseModel, ValidationError

from ...core.llm_client import LLMClient
from ...core.types import NodeStatus, NodeMetrics, NodeExecutionError

logger = logging.getLogger(__name__)


class BaseNode:
    """Base class for all pipeline nodes."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.metrics = NodeMetrics(status=NodeStatus.RUNNING)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node with the given state.

        Args:
            state: Current pipeline state

        Returns:
            Updated state dictionary

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError


class IntelligenceNode(BaseNode):
    """Node for free-form reasoning and analysis using LLMs."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: str,
        required_state_keys: Optional[List[str]] = None,
        name: str = "intelligence",
        description: str = "Free-form reasoning phase",
    ):
        super().__init__(name, description)
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.required_state_keys = required_state_keys or []

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute intelligence analysis on the state.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with 'analysis' key

        Raises:
            ValueError: If required state keys are missing
        """
        start = datetime.now()
        self.metrics.input_keys = self.required_state_keys

        try:
            # Validate inputs
            missing = [k for k in self.required_state_keys if k not in state]
            if missing:
                raise ValueError(f"Missing required keys: {missing}")

            # Format prompt
            format_dict = {k: state[k] for k in self.required_state_keys}
            prompt = self.prompt_template.format(**format_dict)

            logger.info(f"[{self.name}] Running intelligence node")

            # Invoke LLM using new client interface
            analysis = await self.llm_client.generate(
                system="You are a helpful assistant that analyzes information.",
                user=prompt,
            )

            # Store results
            state["analysis"] = analysis

            self.metrics.output_keys = ["analysis"]
            self.metrics.status = NodeStatus.SUCCESS

            return state

        except Exception as e:
            self.metrics.status = NodeStatus.FAILED
            self.metrics.error_message = str(e)
            logger.exception(f"[{self.name}] Error: {e}")
            raise

        finally:
            elapsed = (datetime.now() - start).total_seconds() * 1000
            self.metrics.duration_ms = elapsed


class ExtractionNode(BaseNode):
    """Node for extracting structured data from text using LLMs with JSON repair."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: str,
        output_schema: Type[BaseModel],
        required_state_keys: Optional[List[str]] = None,
        name: str = "extraction",
        description: str = "Structured data extraction",
    ):
        super().__init__(name, description)
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.output_schema = output_schema
        self.required_state_keys = required_state_keys or ["analysis"]

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute extraction on the state.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with 'extracted' key containing parsed data

        Raises:
            ValueError: If required state keys are missing
        """
        start = datetime.now()
        self.metrics.input_keys = self.required_state_keys

        try:
            # Validate inputs
            missing = [k for k in self.required_state_keys if k not in state]
            if missing:
                raise ValueError(f"Missing required keys: {missing}")

            # Format prompt with schema
            format_dict = {k: state[k] for k in self.required_state_keys}
            schema_json = self.output_schema.model_json_schema()
            prompt = self.prompt_template.format(**format_dict)
            prompt += f"\n\nExtract data matching this schema (return valid JSON only):\n{json.dumps(schema_json, indent=2)}"

            logger.info(f"[{self.name}] Running extraction node")

            # Invoke LLM using new client interface
            response_text = await self.llm_client.generate(
                system="You are a helpful assistant that extracts structured data.",
                user=prompt,
            )

            # Extract and repair JSON
            json_text = self._extract_json(response_text)
            json_data = self._repair_json(json_text)

            # Parse with Pydantic
            parsed = self.output_schema.model_validate(json_data)
            state["extracted"] = parsed

            self.metrics.output_keys = ["extracted"]
            self.metrics.status = NodeStatus.SUCCESS

            return state

        except ValidationError as e:
            self.metrics.status = NodeStatus.FAILED
            self.metrics.error_message = f"Validation error: {e}"
            logger.exception(f"[{self.name}] Validation error: {e}")
            raise

        except Exception as e:
            self.metrics.status = NodeStatus.FAILED
            self.metrics.error_message = str(e)
            logger.exception(f"[{self.name}] Error: {e}")
            raise

        finally:
            elapsed = (datetime.now() - start).total_seconds() * 1000
            self.metrics.duration_ms = elapsed

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text, handling code blocks and markdown."""
        # Try to find JSON in code blocks first
        json_block = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        if json_block:
            return json_block.group(1)

        # Try to find JSON object/array directly
        json_match = re.search(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)
        if json_match:
            return json_match.group(1)

        return text.strip()

    def _repair_json(self, json_text: str) -> Dict[str, Any]:
        """Repair common JSON issues."""
        # Try direct parse first
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass

        # Common repairs
        json_text = re.sub(r",\s*}", "}", json_text)
        json_text = re.sub(r",\s*]", "]", json_text)
        json_text = re.sub(r"'([^']*)':", r'"\1":', json_text)
        json_text = re.sub(r":\s*'([^']*)'", r': "\1"', json_text)

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.warning(f"[{self.name}] Could not repair JSON: {e}")
            raise


class ValidationNode(BaseNode):
    """Node for validating extracted data against rules and schema."""

    def __init__(
        self,
        output_schema: Type[BaseModel],
        validation_rules: Optional[Dict[str, Callable[[BaseModel], bool]]] = None,
        required_state_keys: Optional[List[str]] = None,
        name: str = "validation",
        description: str = "Data validation",
    ):
        super().__init__(name, description)
        self.output_schema = output_schema
        self.validation_rules = validation_rules or {}
        self.required_state_keys = required_state_keys or ["extracted"]

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation on the state.

        Args:
            state: Current pipeline state

        Returns:
            Updated state with 'validated' key containing validated data

        Raises:
            ValueError: If required state keys are missing or validation fails
        """
        start = datetime.now()
        self.metrics.input_keys = self.required_state_keys

        try:
            # Validate inputs
            missing = [k for k in self.required_state_keys if k not in state]
            if missing:
                raise ValueError(f"Missing required keys: {missing}")

            extracted = state["extracted"]

            # Ensure it matches the schema
            if not isinstance(extracted, self.output_schema):
                # Try to convert
                try:
                    extracted = self.output_schema.model_validate(extracted)
                except ValidationError as e:
                    raise ValueError(f"Schema validation failed: {e}") from e

            # Apply custom validation rules
            failed_rules = []
            for rule_name, rule_func in self.validation_rules.items():
                try:
                    if not rule_func(extracted):
                        failed_rules.append(rule_name)
                        self.metrics.warnings.append(f"Validation rule '{rule_name}' failed")
                except Exception as e:
                    logger.warning(f"[{self.name}] Rule '{rule_name}' raised exception: {e}")
                    failed_rules.append(rule_name)

            if failed_rules:
                error_msg = f"Validation rules failed: {', '.join(failed_rules)}"
                self.metrics.warnings.append(error_msg)
                logger.warning(f"[{self.name}] {error_msg}")

            # Store validated result
            state["validated"] = extracted

            self.metrics.output_keys = ["validated"]
            self.metrics.status = NodeStatus.SUCCESS

            return state

        except Exception as e:
            self.metrics.status = NodeStatus.FAILED
            self.metrics.error_message = str(e)
            logger.exception(f"[{self.name}] Error: {e}")
            raise

        finally:
            elapsed = (datetime.now() - start).total_seconds() * 1000
            self.metrics.duration_ms = elapsed

