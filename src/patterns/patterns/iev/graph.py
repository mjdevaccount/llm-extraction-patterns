"""IEV Pattern: Intelligence-Extraction-Validation state machine using LangGraph."""

import json
import logging
import re
import time
from typing import Any, Dict, Literal, TypedDict, Optional, Type
from collections import defaultdict

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    StateGraph = None
    END = None

from pydantic import BaseModel, ValidationError

from ...core.llm_client import LLMClient
from .abstractions import IValidationStrategy

logger = logging.getLogger(__name__)


class IEVState(TypedDict):
    """State for IEV pattern."""

    input: str
    analysis: str
    extracted: Any
    validated: Any
    verification_status: Literal["PENDING", "APPROVED", "REJECTED"]
    error: str
    metrics: Dict[str, Any]  # Track execution metrics


class MetricsCollector:
    """Collect metrics from IEV nodes."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def record(self, node_name: str, duration_ms: float, status: str, details: Optional[Dict] = None):
        """Record node execution."""
        self.metrics[node_name].append({
            "timestamp": time.time(),
            "duration_ms": duration_ms,
            "status": status,
            "details": details or {},
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        for node_name, executions in self.metrics.items():
            durations = [e["duration_ms"] for e in executions]
            summary[node_name] = {
                "count": len(executions),
                "avg_ms": sum(durations) / len(durations) if durations else 0,
                "min_ms": min(durations) if durations else 0,
                "max_ms": max(durations) if durations else 0,
                "success_rate": len([e for e in executions if e["status"] == "success"]) / len(executions) if executions else 0,
            }
        return summary


def create_iev_graph(
    llm_client: LLMClient,
    output_schema: Type[BaseModel],
    intelligence_prompt: str,
    extraction_prompt: str,
    verification_prompt: str,
    validation_strategy: Optional[IValidationStrategy] = None,
):
    """
    Create IEV pattern graph.

    Args:
        llm_client: LLM client instance
        output_schema: Pydantic schema for extraction
        intelligence_prompt: Prompt template for intelligence phase
        extraction_prompt: Prompt template for extraction phase
        verification_prompt: Prompt template for verification phase
        validation_strategy: Optional strategy for sophisticated validation (SOLID pattern)

    Returns:
        Configured LangGraph StateGraph (if langgraph installed) or simple executor
    """
    
    metrics = MetricsCollector()
    
    if StateGraph is None:
        # Fallback: return a simple executor function
        logger.warning("LangGraph not installed. Using basic executor (no visualization/inspection).")
        logger.warning("Install with: pip install langgraph")
        
        async def simple_executor(input_text: str):
            state = {
                "input": input_text,
                "verification_status": "PENDING",
                "metrics": {},
            }
            state = await intelligence_node(state)
            state = await extraction_node(state)
            state = await verification_node(state)
            state["metrics"] = metrics.get_summary()
            return state
        return simple_executor

    graph = StateGraph(IEVState)

    async def intelligence_node(state: IEVState) -> Dict[str, Any]:
        """Intelligence phase: Analyze input."""
        logger.info("[IEV] Intelligence phase")
        start_time = time.time()
        try:
            prompt = intelligence_prompt.format(input=state["input"])
            analysis = await llm_client.generate(
                system="You are a helpful assistant that analyzes information.",
                user=prompt,
            )
            duration = (time.time() - start_time) * 1000
            metrics.record("intelligence", duration, "success")
            return {"analysis": analysis}
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            metrics.record("intelligence", duration, "error", {"error": str(e)})
            logger.exception(f"[IEV] Intelligence error: {e}")
            return {"error": str(e)}

    async def extraction_node(state: IEVState) -> Dict[str, Any]:
        """Extraction phase: Extract structured data with repair strategies."""
        logger.info("[IEV] Extraction phase")
        start_time = time.time()
        try:
            if "error" in state and state["error"]:
                return {"error": state["error"]}

            analysis = state.get("analysis", "")
            prompt = extraction_prompt.format(analysis=analysis)

            # Add schema to prompt
            schema_json = output_schema.model_json_schema()
            prompt += f"\n\nExtract data matching this schema (return valid JSON only):\n{json.dumps(schema_json, indent=2)}"

            response_text = await llm_client.generate(
                system="You are a helpful assistant that extracts structured data.",
                user=prompt,
            )

            # Extract and repair JSON
            json_text = _extract_json(response_text)
            json_data = _repair_json(json_text)

            # Parse with Pydantic
            parsed = output_schema.model_validate(json_data)

            duration = (time.time() - start_time) * 1000
            metrics.record("extraction", duration, "success", {"schema": output_schema.__name__})
            return {"extracted": parsed}
        except ValidationError as e:
            duration = (time.time() - start_time) * 1000
            metrics.record("extraction", duration, "error", {"error": f"Validation: {e}"})
            logger.exception(f"[IEV] Extraction validation error: {e}")
            return {"error": f"Validation error: {e}"}
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            metrics.record("extraction", duration, "error", {"error": str(e)})
            logger.exception(f"[IEV] Extraction error: {e}")
            return {"error": str(e)}

    async def verification_node(state: IEVState) -> Dict[str, Any]:
        """Verification phase: Verify extracted data using optional validation strategy."""
        logger.info("[IEV] Verification phase")
        start_time = time.time()
        try:
            if "error" in state and state["error"]:
                duration = (time.time() - start_time) * 1000
                metrics.record("verification", duration, "error", {"error": state["error"]})
                return {
                    "verification_status": "REJECTED",
                    "error": state["error"],
                }

            extracted = state.get("extracted")
            if extracted is None:
                duration = (time.time() - start_time) * 1000
                metrics.record("verification", duration, "error", {"error": "No extracted data"})
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
                
                duration = (time.time() - start_time) * 1000
                metrics.record("verification", duration, "success", {
                    "strategy": validation_strategy.mode_name,
                    "outcome": status,
                })
                
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
            verification_result = await llm_client.generate(
                system="You are a helpful assistant that verifies data quality and safety.",
                user=prompt,
            )
            verification_result = verification_result.upper()

            if "APPROVED" in verification_result:
                status = "APPROVED"
                validated = extracted
            else:
                status = "REJECTED"
                validated = None

            duration = (time.time() - start_time) * 1000
            metrics.record("verification", duration, "success", {"outcome": status})
            
            return {
                "verification_status": status,
                "validated": validated,
            }
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            metrics.record("verification", duration, "error", {"error": str(e)})
            logger.exception(f"[IEV] Verification error: {e}")
            return {
                "verification_status": "REJECTED",
                "error": str(e),
            }

    # Add nodes
    graph.add_node("intelligence", intelligence_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("verification", verification_node)

    # Define edges
    graph.set_entry_point("intelligence")
    graph.add_edge("intelligence", "extraction")
    graph.add_edge("extraction", "verification")
    graph.add_edge("verification", END)

    compiled_graph = graph.compile()
    
    # Wrap to add metrics
    async def invoke_with_metrics(input_text: str) -> Dict[str, Any]:
        result = compiled_graph.invoke({"input": input_text})
        result["metrics"] = metrics.get_summary()
        return result
    
    return invoke_with_metrics


def _extract_json(text: str) -> str:
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


def _repair_json(json_text: str) -> Dict[str, Any]:
    """Repair common JSON issues (basic strategies)."""
    # Try direct parse first
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass

    # Common repairs
    json_text = re.sub(r",\s*}", "}", json_text)  # Remove trailing commas in objects
    json_text = re.sub(r",\s*\]", "]", json_text)  # Remove trailing commas in arrays
    json_text = re.sub(r"'([^']*)':", r'"\1":', json_text)  # Single to double quotes (keys)
    json_text = re.sub(r":\s*'([^']*)'(,|$|\})", r': "\1"\2', json_text)  # Single to double quotes (values)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Could not repair JSON: {e}")
        raise
