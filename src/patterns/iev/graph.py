"""
IEV Pattern: Intelligence-Extraction-Validation using WorkflowBuilder.

This is the clean version using core.graph_builder.
"""

from typing import Any, Dict, Literal, Optional, Type, TypedDict

from pydantic import BaseModel

from core.llm_client import LLMClient
from core.abstractions import IValidationStrategy
from core.retry_strategies import IRetryStrategy, CompositeRetryStrategy
from core.graph_builder import create_linear_workflow

from .nodes.intelligence_node import intelligence_node
from .nodes.extraction_node import extraction_node
from .nodes.verification_node import verification_node


class IEVState(TypedDict):
    """State for IEV pattern."""
    input: str
    analysis: str
    extracted: Any
    validated: Any
    verification_status: Literal["PENDING", "APPROVED", "REJECTED"]
    error: str
    metrics: Dict[str, Any]
    extraction_attempts: int


def create_iev_graph(
    llm_client: LLMClient,
    output_schema: Type[BaseModel],
    intelligence_prompt: str,
    extraction_prompt: str,
    verification_prompt: str,
    validation_strategy: Optional[IValidationStrategy] = None,
    retry_strategy: Optional[IRetryStrategy] = None,
    max_extraction_retries: int = 4,
):
    """
    Create IEV pattern graph.
    
    Uses core.graph_builder for workflow construction.
    """
    if retry_strategy is None:
        retry_strategy = CompositeRetryStrategy()
    
    # Define nodes with their parameters
    return create_linear_workflow(
        IEVState,
        [
            ("intelligence", intelligence_node, {
                "llm_client": llm_client,
                "intelligence_prompt": intelligence_prompt,
            }),
            ("extraction", extraction_node, {
                "llm_client": llm_client,
                "output_schema": output_schema,
                "extraction_prompt": extraction_prompt,
                "retry_strategy": retry_strategy,
                "max_extraction_retries": max_extraction_retries,
            }),
            ("verification", verification_node, {
                "llm_client": llm_client,
                "output_schema": output_schema,
                "verification_prompt": verification_prompt,
                "validation_strategy": validation_strategy,
            }),
        ]
    )

