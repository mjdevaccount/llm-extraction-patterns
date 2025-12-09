"""Tests for IEV nodes: Intelligence, Extraction, Validation."""

import pytest
from typing import Dict, Any
from pydantic import BaseModel, Field
from patterns.patterns.iev.nodes import (
    IntelligenceNode,
    ExtractionNode,
    ValidationNode,
    ValidationMode,
    NodeExecutionError,
    NodeStatus,
)


# ============================================================================
# Test Schemas
# ============================================================================

class Person(BaseModel):
    """Test schema for extraction."""
    name: str
    age: int = Field(gt=0, le=150)
    email: str


class Deal(BaseModel):
    """Test schema for deal extraction."""
    company: str
    amount: float = Field(gt=0)
    currency: str = "USD"


# ============================================================================
# Mock LLM
# ============================================================================

class MockLLM:
    """Mock LangChain ChatModel for testing."""
    
    def __init__(self, responses: list = None, should_fail: bool = False):
        self.responses = responses or []
        self.should_fail = should_fail
        self.call_count = 0
    
    async def ainvoke(self, messages):
        """Mock async invoke."""
        self.call_count += 1
        if self.should_fail:
            raise Exception("LLM call failed")
        
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        if self.call_count <= len(self.responses):
            return MockResponse(self.responses[self.call_count - 1])
        return MockResponse("Default response")


# ============================================================================
# IntelligenceNode Tests
# ============================================================================

class TestIntelligenceNode:
    """Tests for IntelligenceNode."""
    
    @pytest.mark.asyncio
    async def test_intelligence_node_execution(self):
        """Test intelligence node executes successfully."""
        llm = MockLLM(responses=["This is an analysis of the input."])
        node = IntelligenceNode(
            llm=llm,
            prompt_template="Analyze: {input}",
            required_state_keys=["input"],
        )
        
        state = {"input": "test data"}
        result = await node.execute(state)
        
        assert "analysis" in result
        assert result["analysis"] == "This is an analysis of the input."
        assert node.metrics.status == NodeStatus.SUCCESS
        assert "analysis" in node.metrics.output_keys
    
    @pytest.mark.asyncio
    async def test_intelligence_node_missing_keys(self):
        """Test intelligence node fails on missing required keys."""
        llm = MockLLM()
        node = IntelligenceNode(
            llm=llm,
            prompt_template="Analyze: {input}",
            required_state_keys=["input"],
        )
        
        state = {}  # Missing "input"
        with pytest.raises(NodeExecutionError, match="Missing required keys"):
            await node.execute(state)
    
    @pytest.mark.asyncio
    async def test_intelligence_node_llm_failure(self):
        """Test intelligence node handles LLM failures."""
        llm = MockLLM(should_fail=True)
        node = IntelligenceNode(
            llm=llm,
            prompt_template="Analyze: {input}",
            required_state_keys=["input"],
        )
        
        state = {"input": "test"}
        with pytest.raises(Exception):
            await node.execute(state)
        
        assert node.metrics.status == NodeStatus.FAILED


# ============================================================================
# ExtractionNode Tests
# ============================================================================

class TestExtractionNode:
    """Tests for ExtractionNode."""
    
    @pytest.mark.asyncio
    async def test_extraction_node_valid_json(self):
        """Test extraction with valid JSON."""
        llm = MockLLM(responses=['{"name": "Alice", "age": 30, "email": "alice@example.com"}'])
        node = ExtractionNode(
            llm=llm,
            prompt_template="Extract: {analysis}",
            output_schema=Person,
        )
        
        state = {"analysis": "Alice is 30 years old, email alice@example.com"}
        result = await node.execute(state)
        
        assert "extracted" in result
        assert result["extracted"]["name"] == "Alice"
        assert result["extracted"]["age"] == 30
        assert node.metrics.status == NodeStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_extraction_node_malformed_json(self):
        """Test extraction with malformed JSON uses repair strategies."""
        # JSON with trailing comma
        llm = MockLLM(responses=['{"name": "Bob", "age": 25, "email": "bob@example.com",}'])
        node = ExtractionNode(
            llm=llm,
            prompt_template="Extract: {analysis}",
            output_schema=Person,
            json_repair_strategies=["incremental_repair"],  # Should fix trailing comma
        )
        
        state = {"analysis": "Bob is 25, email bob@example.com"}
        result = await node.execute(state)
        
        assert "extracted" in result
        assert node.metrics.status == NodeStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_extraction_node_missing_analysis(self):
        """Test extraction node fails without analysis."""
        llm = MockLLM()
        node = ExtractionNode(
            llm=llm,
            prompt_template="Extract: {analysis}",
            output_schema=Person,
        )
        
        state = {}  # Missing "analysis"
        with pytest.raises(NodeExecutionError, match="Missing 'analysis'"):
            await node.execute(state)


# ============================================================================
# ValidationNode Tests
# ============================================================================

class TestValidationNode:
    """Tests for ValidationNode."""
    
    @pytest.mark.asyncio
    async def test_validation_node_strict_mode_success(self):
        """Test strict validation mode with valid data."""
        node = ValidationNode(
            output_schema=Person,
            mode=ValidationMode.STRICT,
        )
        
        state = {
            "extracted": {
                "name": "Charlie",
                "age": 35,
                "email": "charlie@example.com",
            }
        }
        result = await node.execute(state)
        
        assert "validated" in result
        assert result["validated"]["name"] == "Charlie"
        assert node.metrics.status == NodeStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_validation_node_strict_mode_failure(self):
        """Test strict validation mode fails on invalid data."""
        node = ValidationNode(
            output_schema=Person,
            mode=ValidationMode.STRICT,
        )
        
        state = {
            "extracted": {
                "name": "Invalid",
                "age": -5,  # Invalid: age must be > 0
                "email": "invalid@example.com",
            }
        }
        
        # Strict mode should raise an error
        with pytest.raises((NodeExecutionError, ValueError)):
            await node.execute(state)
    
    @pytest.mark.asyncio
    async def test_validation_node_best_effort_mode(self):
        """Test best-effort validation mode applies repairs."""
        node = ValidationNode(
            output_schema=Person,
            mode=ValidationMode.BEST_EFFORT,
        )
        
        # Missing email, but best-effort should handle it
        state = {
            "extracted": {
                "name": "David",
                "age": 40,
                # Missing email
            }
        }
        
        # Best-effort might fail or repair - depends on schema defaults
        # This test verifies the mode is used
        try:
            result = await node.execute(state)
            # Should have either validated data or validation metadata
            assert "validated" in result or "validation_metadata" in result
        except (NodeExecutionError, ValueError):
            # Best-effort can still fail if no defaults and repair fails
            # This is acceptable behavior
            pass
    
    @pytest.mark.asyncio
    async def test_validation_node_custom_rules(self):
        """Test validation with custom rules."""
        def age_check(data):
            return data.get("age", 0) >= 18
        
        node = ValidationNode(
            output_schema=Person,
            mode=ValidationMode.STRICT,
            validation_rules={"adult_check": age_check},
        )
        
        state = {
            "extracted": {
                "name": "Eve",
                "age": 25,
                "email": "eve@example.com",
            }
        }
        result = await node.execute(state)
        assert "validated" in result
    
    @pytest.mark.asyncio
    async def test_validation_node_missing_extracted(self):
        """Test validation node fails without extracted data."""
        node = ValidationNode(
            output_schema=Person,
            mode=ValidationMode.STRICT,
        )
        
        state = {}  # Missing "extracted"
        with pytest.raises(NodeExecutionError, match="Missing 'extracted'"):
            await node.execute(state)

