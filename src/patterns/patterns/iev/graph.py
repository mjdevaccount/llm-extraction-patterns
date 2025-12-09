"""IEV Pattern: Intelligence-Extraction-Validation state machine using LangGraph."""

import json
import logging
import re
from typing import Any, Dict, Literal, TypedDict

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    # Fallback if langgraph not installed
    StateGraph = None
    END = None

from pydantic import BaseModel, ValidationError

from ...core.llm_client import LLMClient

logger = logging.getLogger(__name__)


class IEVState(TypedDict):
    """State for IEV pattern."""

    input: str
    analysis: str
    extracted: Any
    validated: Any
    verification_status: Literal["PENDING", "APPROVED", "REJECTED"]
    error: str


def create_iev_graph(
    llm_client: LLMClient,
    output_schema: type[BaseModel],
    intelligence_prompt: str,
    extraction_prompt: str,
    verification_prompt: str,
):
    """Create IEV pattern graph.

    Args:
        llm_client: LLM client instance
        output_schema: Pydantic schema for extraction
        intelligence_prompt: Prompt template for intelligence phase
        extraction_prompt: Prompt template for extraction phase
        verification_prompt: Prompt template for verification phase

    Returns:
        Configured LangGraph StateGraph (if langgraph installed) or simple executor
    """
    if StateGraph is None:
        # Fallback: return a simple executor function
        async def simple_executor(input_text: str):
            state = {"input": input_text, "verification_status": "PENDING"}
            state = await intelligence_node(state)
            state = await extraction_node(state)
            state = await verification_node(state)
            return state
        return simple_executor

    graph = StateGraph(IEVState)

    async def intelligence_node(state: IEVState) -> Dict[str, Any]:
        """Intelligence phase: Analyze input."""
        logger.info("[IEV] Intelligence phase")
        try:
            prompt = intelligence_prompt.format(input=state["input"])
            # Use the new LLM client interface
            analysis = await llm_client.generate(
                system="You are a helpful assistant that analyzes information.",
                user=prompt,
            )

            return {
                "analysis": analysis,
            }
        except Exception as e:
            logger.exception(f"[IEV] Intelligence error: {e}")
            return {"error": str(e)}

    async def extraction_node(state: IEVState) -> Dict[str, Any]:
        """Extraction phase: Extract structured data."""
        logger.info("[IEV] Extraction phase")
        try:
            if "error" in state and state["error"]:
                return {"error": state["error"]}

            analysis = state.get("analysis", "")
            prompt = extraction_prompt.format(analysis=analysis)

            # Add schema to prompt
            schema_json = output_schema.model_json_schema()
            prompt += f"\n\nExtract data matching this schema (return valid JSON only):\n{json.dumps(schema_json, indent=2)}"

            # Use the new LLM client interface
            response_text = await llm_client.generate(
                system="You are a helpful assistant that extracts structured data.",
                user=prompt,
            )

            # Extract and repair JSON
            json_text = _extract_json(response_text)
            json_data = _repair_json(json_text)

            # Parse with Pydantic
            parsed = output_schema.model_validate(json_data)

            return {
                "extracted": parsed,
            }
        except ValidationError as e:
            logger.exception(f"[IEV] Extraction validation error: {e}")
            return {"error": f"Validation error: {e}"}
        except Exception as e:
            logger.exception(f"[IEV] Extraction error: {e}")
            return {"error": str(e)}

    async def verification_node(state: IEVState) -> Dict[str, Any]:
        """Verification phase: Verify extracted data."""
        logger.info("[IEV] Verification phase")
        try:
            if "error" in state and state["error"]:
                return {
                    "verification_status": "REJECTED",
                    "error": state["error"],
                }

            extracted = state.get("extracted")
            if extracted is None:
                return {
                    "verification_status": "REJECTED",
                    "error": "No extracted data to verify",
                }

            # Convert to dict if Pydantic model
            if isinstance(extracted, BaseModel):
                extracted_dict = extracted.model_dump()
            else:
                extracted_dict = extracted

            prompt = verification_prompt.format(extracted=json.dumps(extracted_dict, indent=2, default=str))
            # Use the new LLM client interface
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

    # Add nodes
    graph.add_node("intelligence", intelligence_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node("verification", verification_node)

    # Define edges
    graph.set_entry_point("intelligence")
    graph.add_edge("intelligence", "extraction")
    graph.add_edge("extraction", "verification")
    graph.add_edge("verification", END)

    return graph.compile()


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
        logger.warning(f"Could not repair JSON: {e}")
        raise

