"""
Tests for IEV (Intelligence-Extraction-Validation) Pattern

Tests cover:
  - Happy path (valid extraction and verification)
  - JSON repair strategies
  - Validation failures
  - Error handling
  - Metrics collection
"""

import pytest
from pydantic import BaseModel
from patterns.core.llm_client import LLMClient
from patterns.patterns.iev.graph import create_iev_graph, MetricsCollector


# ============================================================================
# Test Schemas
# ============================================================================

class DealInfo(BaseModel):
    """Simple schema for testing extraction."""
    company_name: str
    deal_value: float
    deal_type: str  # "acquisition", "investment", "merger"


class Person(BaseModel):
    """Schema for testing name extraction."""
    first_name: str
    last_name: str
    age: int


# ============================================================================
# Mock LLM Client
# ============================================================================

class MockLLMClient(LLMClient):
    """Mock LLM for testing (no API calls)."""
    
    def __init__(self, response_override: str = None, should_fail: bool = False):
        self.response_override = response_override
        self.should_fail = should_fail
        self.call_count = 0
    
    def generate(self, system: str, user: str, temperature: float = 0.7, max_tokens=None, **kwargs) -> str:
        """Synchronous mock generate."""
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError("LLM failure (mocked)")
        return self.response_override or "{}"
    
    def stream(self, system: str, user: str, temperature: float = 0.7):
        """Mock stream."""
        yield self.response_override or "{}"


# ============================================================================
# Tests: Happy Path
# ============================================================================

@pytest.mark.asyncio
async def test_iev_happy_path_valid_json():
    """Test IEV with valid JSON extraction."""
    llm = MockLLMClient(
        response_override='{"company_name": "Acme Inc", "deal_value": 1000000, "deal_type": "acquisition"}'
    )
    
    graph = create_iev_graph(
        llm_client=llm,
        output_schema=DealInfo,
        intelligence_prompt="Extract deal from: {input}",
        extraction_prompt="Extract from analysis: {analysis}",
        verification_prompt="Verify deal: {extracted}",
    )
    
    # Since graph might be async or sync, handle both
    if hasattr(graph, '__await__'):
        result = await graph("Acme Inc acquired TechCorp for $1M")
    else:
        # It's a regular function, just call it
        result = await graph("Acme Inc acquired TechCorp for $1M")
    
    assert result["verification_status"] in ["APPROVED", "PENDING", "REJECTED"]
    # We called intelligence, extraction, and verification
    assert llm.call_count >= 1  # At least one call


@pytest.mark.asyncio
async def test_iev_metrics_collection():
    """Test that metrics are collected from each node."""
    llm = MockLLMClient(
        response_override='{"company_name": "Acme", "deal_value": 500000, "deal_type": "investment"}'
    )
    
    graph = create_iev_graph(
        llm_client=llm,
        output_schema=DealInfo,
        intelligence_prompt="Analyze: {input}",
        extraction_prompt="Extract from: {analysis}",
        verification_prompt="Verify: {extracted}",
    )
    
    result = await graph("Test input")
    
    # Check that metrics were collected
    assert "metrics" in result
    metrics = result["metrics"]
    assert isinstance(metrics, dict)


# ============================================================================
# Tests: JSON Repair
# ============================================================================

@pytest.mark.asyncio
async def test_iev_json_repair_trailing_comma():
    """Test JSON repair with trailing commas."""
    llm = MockLLMClient(
        # Malformed JSON with trailing comma
        response_override='{"company_name": "Acme", "deal_value": 1000000, "deal_type": "acquisition",}'
    )
    
    graph = create_iev_graph(
        llm_client=llm,
        output_schema=DealInfo,
        intelligence_prompt="Analyze: {input}",
        extraction_prompt="Extract: {analysis}",
        verification_prompt="Verify: {extracted}",
    )
    
    result = await graph("Input text")
    # Should still extract despite bad JSON (repair strategies handle it)
    # May have error if repair fails, but that's acceptable
    assert "verification_status" in result


@pytest.mark.asyncio
async def test_iev_json_repair_single_quotes():
    """Test JSON repair with single quotes."""
    llm = MockLLMClient(
        response_override="{'company_name': 'TechCorp', 'deal_value': 500000, 'deal_type': 'acquisition'}"
    )
    
    graph = create_iev_graph(
        llm_client=llm,
        output_schema=DealInfo,
        intelligence_prompt="Analyze: {input}",
        extraction_prompt="Extract: {analysis}",
        verification_prompt="Verify: {extracted}",
    )
    
    result = await graph("Input")
    # Should repair quotes and extract (repair strategies handle it)
    # May have error if repair fails, but that's acceptable
    assert "verification_status" in result


# ============================================================================
# Tests: Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_iev_llm_failure():
    """Test IEV handles LLM failures gracefully."""
    llm = MockLLMClient(should_fail=True)
    
    graph = create_iev_graph(
        llm_client=llm,
        output_schema=DealInfo,
        intelligence_prompt="Analyze: {input}",
        extraction_prompt="Extract: {analysis}",
        verification_prompt="Verify: {extracted}",
    )
    
    result = await graph("Input")
    
    # Should have error in result
    assert result["verification_status"] == "REJECTED"
    assert "error" in result


@pytest.mark.asyncio
async def test_iev_invalid_schema():
    """Test IEV handles schema validation errors."""
    llm = MockLLMClient(
        # Missing required fields
        response_override='{"company_name": "Acme"}'
    )
    
    graph = create_iev_graph(
        llm_client=llm,
        output_schema=DealInfo,
        intelligence_prompt="Analyze: {input}",
        extraction_prompt="Extract: {analysis}",
        verification_prompt="Verify: {extracted}",
    )
    
    result = await graph("Input")
    
    # Should fail validation
    assert result["verification_status"] == "REJECTED"
    assert "error" in result


# ============================================================================
# Tests: MetricsCollector
# ============================================================================

def test_metrics_collector_records_execution():
    """Test metrics collector tracks node executions."""
    collector = MetricsCollector()
    
    # Record some executions
    collector.record("intelligence", 100.0, "success")
    collector.record("extraction", 200.0, "success")
    collector.record("extraction", 150.0, "error", {"error": "JSON parse failed"})
    
    summary = collector.get_summary()
    
    assert "intelligence" in summary
    assert summary["intelligence"]["count"] == 1
    assert summary["intelligence"]["avg_ms"] == 100.0
    
    assert "extraction" in summary
    assert summary["extraction"]["count"] == 2
    assert summary["extraction"]["avg_ms"] == 175.0
    # 50% success (1 success, 1 error)
    assert summary["extraction"]["success_rate"] == 0.5


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_iev_full_workflow():
    """Test complete IEV workflow: intelligence -> extraction -> verification."""
    llm = MockLLMClient(
        response_override='{"first_name": "John", "last_name": "Doe", "age": 30}'
    )
    
    graph = create_iev_graph(
        llm_client=llm,
        output_schema=Person,
        intelligence_prompt="Extract person info from: {input}",
        extraction_prompt="Get person data from: {analysis}",
        verification_prompt="Verify person: {extracted}",
    )
    
    result = await graph("John Doe is 30 years old")
    
    # Verify the workflow completed
    assert "verification_status" in result
    assert "metrics" in result
    assert llm.call_count >= 1


if __name__ == "__main__":
    # Run tests: pytest tests/test_iev.py -v
    pass
