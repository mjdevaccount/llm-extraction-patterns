"""Tests for IEV pattern (node-based implementation)."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from patterns.core import create_llm_client
from patterns.patterns.iev import IntelligenceNode, ExtractionNode, ValidationNode
from patterns.core.types import NodeStatus
from pydantic import BaseModel, Field


class TestSchema(BaseModel):
    """Test schema for extraction."""

    name: str
    value: int = Field(gt=0)


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="Test analysis response")
    return llm


@pytest.fixture
def intelligence_node(mock_llm_client):
    """Create an IntelligenceNode instance."""
    return IntelligenceNode(
        mock_llm_client,
        "Analyze: {text}",
        required_state_keys=["text"],
        name="test_intelligence",
    )


@pytest.mark.asyncio
async def test_intelligence_node_success(intelligence_node, mock_llm_client):
    """Test successful intelligence node execution."""
    state = {"text": "Test input"}

    result = await intelligence_node.execute(state)

    assert "analysis" in result
    assert result["analysis"] == "Test analysis response"
    assert intelligence_node.metrics.status == NodeStatus.SUCCESS
    assert intelligence_node.metrics.duration_ms > 0


@pytest.mark.asyncio
async def test_intelligence_node_missing_keys(intelligence_node):
    """Test intelligence node with missing required keys."""
    state = {}

    with pytest.raises(ValueError, match="Missing required keys"):
        await intelligence_node.execute(state)

    assert intelligence_node.metrics.status == NodeStatus.FAILED


@pytest.mark.asyncio
async def test_extraction_node_success(mock_llm_client):
    """Test successful extraction node execution."""
    import json

    json_data = {"name": "test", "value": 42}
    mock_llm_client.generate = AsyncMock(return_value=json.dumps(json_data))

    extract = ExtractionNode(
        mock_llm_client,
        "Extract: {analysis}",
        output_schema=TestSchema,
        required_state_keys=["analysis"],
    )

    state = {"analysis": "Some analysis text"}

    result = await extract.execute(state)

    assert "extracted" in result
    assert isinstance(result["extracted"], TestSchema)
    assert result["extracted"].name == "test"
    assert result["extracted"].value == 42
    assert extract.metrics.status == NodeStatus.SUCCESS


@pytest.mark.asyncio
async def test_validation_node_success():
    """Test successful validation node execution."""
    extracted = TestSchema(name="test", value=42)
    state = {"extracted": extracted}

    validate = ValidationNode(
        output_schema=TestSchema,
        validation_rules={
            "value_positive": lambda x: x.value > 0,
        },
    )

    result = await validate.execute(state)

    assert "validated" in result
    assert result["validated"] == extracted
    assert validate.metrics.status == NodeStatus.SUCCESS


@pytest.mark.asyncio
async def test_full_iev_pipeline(mock_llm_client):
    """Test complete IEV pipeline."""
    import json

    # Setup mock responses
    analysis_response = "Analysis: Deal with Acme Corp, $10M swaps, next week"
    json_response = json.dumps({
        "name": "test",
        "value": 42,
    })

    call_count = 0

    async def mock_generate(system, user):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return analysis_response
        else:
            return json_response

    mock_llm_client.generate = mock_generate

    # Create nodes
    intel = IntelligenceNode(
        mock_llm_client,
        "Analyze: {email}",
        required_state_keys=["email"],
    )

    extract = ExtractionNode(
        mock_llm_client,
        "Extract: {analysis}",
        output_schema=TestSchema,
        required_state_keys=["analysis"],
    )

    validate = ValidationNode(output_schema=TestSchema)

    # Execute pipeline
    state = {"email": "Test email"}
    state = await intel.execute(state)
    assert "analysis" in state

    state = await extract.execute(state)
    assert "extracted" in state

    state = await validate.execute(state)
    assert "validated" in state

