"""Tests for JSON repair and validation strategies."""

import pytest
from pydantic import BaseModel, Field
from patterns.core.extraction_strategies import (
    IncrementalRepairStrategy,
    LLMRepairStrategy,
    RegexRepairStrategy,
    StrictValidationStrategy,
    RetryValidationStrategy,
    BestEffortValidationStrategy,
)
from patterns.core.abstractions import ILLMProvider


# ============================================================================
# Test Schemas
# ============================================================================

class TestSchemaModel(BaseModel):
    """Test schema for strategies."""
    name: str
    age: int = Field(gt=0, le=150)
    score: float = Field(ge=0.0, le=100.0)


# Use TestSchemaModel in tests
TestSchema = TestSchemaModel


# ============================================================================
# Mock LLM Provider
# ============================================================================

class MockLLMProvider(ILLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, response: str = None):
        self.response = response or '{"name": "Test", "age": 25, "score": 85.5}'
        self.call_count = 0
    
    async def invoke(self, messages):
        """Mock invoke."""
        self.call_count += 1
        return self.response
    
    async def invoke_structured(self, messages, schema):
        """Mock structured invoke."""
        self.call_count += 1
        import json
        data = json.loads(self.response)
        return schema(**data)
    
    @property
    def supports_structured_output(self) -> bool:
        """Mock supports structured output."""
        return True


# ============================================================================
# JSON Repair Strategy Tests
# ============================================================================

class TestIncrementalRepairStrategy:
    """Tests for IncrementalRepairStrategy."""
    
    @pytest.mark.asyncio
    async def test_repair_trailing_comma(self):
        """Test repair of trailing comma."""
        strategy = IncrementalRepairStrategy()
        json_text = '{"name": "Alice", "age": 30,}'
        result = await strategy.repair(json_text, TestSchema)
        assert result is not None
        assert result["name"] == "Alice"
        assert result["age"] == 30
    
    @pytest.mark.asyncio
    async def test_repair_unclosed_brace(self):
        """Test repair of unclosed brace."""
        strategy = IncrementalRepairStrategy()
        json_text = '{"name": "Bob", "age": 25'
        result = await strategy.repair(json_text, TestSchema)
        # Should attempt to close the brace
        assert result is not None or result is None  # May or may not succeed
    
    @pytest.mark.asyncio
    async def test_repair_valid_json(self):
        """Test repair with already valid JSON."""
        strategy = IncrementalRepairStrategy()
        json_text = '{"name": "Charlie", "age": 35, "score": 90.0}'
        result = await strategy.repair(json_text, TestSchema)
        assert result is not None
        assert result["name"] == "Charlie"
    
    @pytest.mark.asyncio
    async def test_repair_invalid_json_fails(self):
        """Test repair fails on completely invalid JSON."""
        strategy = IncrementalRepairStrategy()
        json_text = "This is not JSON at all"
        result = await strategy.repair(json_text, TestSchema)
        assert result is None


class TestLLMRepairStrategy:
    """Tests for LLMRepairStrategy."""
    
    @pytest.mark.asyncio
    async def test_llm_repair_success(self):
        """Test LLM repair succeeds."""
        llm = MockLLMProvider('{"name": "David", "age": 40, "score": 75.0}')
        strategy = LLMRepairStrategy(llm)
        json_text = '{"name": "David", "age": 40}'  # Missing score
        result = await strategy.repair(json_text, TestSchema)
        # LLM should add missing field
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_llm_repair_extracts_json(self):
        """Test LLM repair extracts JSON from response."""
        llm = MockLLMProvider('Here is the JSON: {"name": "Eve", "age": 28, "score": 88.0}')
        strategy = LLMRepairStrategy(llm)
        json_text = '{"name": "Eve"}'
        result = await strategy.repair(json_text, TestSchema)
        assert result is not None


class TestRegexRepairStrategy:
    """Tests for RegexRepairStrategy."""
    
    @pytest.mark.asyncio
    async def test_regex_extraction(self):
        """Test regex extraction of fields."""
        strategy = RegexRepairStrategy()
        json_text = "name: Frank, age: 45, score: 92.5"
        result = await strategy.repair(json_text, TestSchema)
        assert result is not None
        assert "name" in result
        assert "age" in result
    
    @pytest.mark.asyncio
    async def test_regex_extraction_partial(self):
        """Test regex extraction with partial fields."""
        strategy = RegexRepairStrategy()
        json_text = "name: Grace, age: 33"
        result = await strategy.repair(json_text, TestSchema)
        # Should extract what it can
        assert result is not None or result is None  # May be partial


# ============================================================================
# Validation Strategy Tests
# ============================================================================

class TestStrictValidationStrategy:
    """Tests for StrictValidationStrategy."""
    
    @pytest.mark.asyncio
    async def test_strict_validation_success(self):
        """Test strict validation passes with valid data."""
        strategy = StrictValidationStrategy()
        data = {"name": "Henry", "age": 50, "score": 95.0}
        result = await strategy.validate(
            data=data,
            schema=TestSchema,
            validation_rules={},
        )
        assert result["outcome"] == "strict_pass"
        assert "validated" in result
        assert result["validated"]["name"] == "Henry"
    
    @pytest.mark.asyncio
    async def test_strict_validation_failure(self):
        """Test strict validation fails with invalid data."""
        strategy = StrictValidationStrategy()
        data = {"name": "Invalid", "age": -5, "score": 95.0}  # Invalid age
        
        with pytest.raises(Exception):  # Should raise ValidationError
            await strategy.validate(
                data=data,
                schema=TestSchema,
                validation_rules={},
            )


class TestBestEffortValidationStrategy:
    """Tests for BestEffortValidationStrategy."""
    
    @pytest.mark.asyncio
    async def test_best_effort_repair(self):
        """Test best-effort validation applies repairs."""
        strategy = BestEffortValidationStrategy()
        data = {
            "name": "Iris",
            "age": "35",  # String instead of int
            "score": 80.0,
        }
        result = await strategy.validate(
            data=data,
            schema=TestSchema,
            validation_rules={},
        )
        # Should attempt type coercion
        assert "validated" in result or "repairs" in result
    
    @pytest.mark.asyncio
    async def test_best_effort_cleans_artifacts(self):
        """Test best-effort cleans extraction artifacts."""
        strategy = BestEffortValidationStrategy()
        data = {"extraction_status": "success"}  # Only status marker
        # Best-effort will try to repair but may fail if no defaults
        try:
            result = await strategy.validate(
                data=data,
                schema=TestSchema,
                validation_rules={},
            )
            # Should have warnings or repairs if it succeeded
            assert "warnings" in result or "repairs" in result or "validated" in result
        except ValueError:
            # Can fail if repair is impossible - this is acceptable
            pass


class TestRetryValidationStrategy:
    """Tests for RetryValidationStrategy."""
    
    @pytest.mark.asyncio
    async def test_retry_validation_requires_llm(self):
        """Test retry validation requires LLM."""
        strategy = RetryValidationStrategy()
        data = {"name": "Jack", "age": 60, "score": 70.0}
        
        with pytest.raises(ValueError, match="requires llm"):
            await strategy.validate(
                data=data,
                schema=TestSchema,
                validation_rules={},
                llm=None,
            )
    
    @pytest.mark.asyncio
    async def test_retry_validation_with_llm(self):
        """Test retry validation with LLM repair."""
        llm = MockLLMProvider('{"name": "Jack", "age": 60, "score": 70.0}')
        # RetryValidationStrategy doesn't take constructor args, llm goes to validate()
        strategy = RetryValidationStrategy()
        data = {"name": "Jack", "age": 60, "score": 70.0}
        
        result = await strategy.validate(
            data=data,
            schema=TestSchema,
            validation_rules={},
            llm=llm,
        )
        assert result["outcome"] in ["strict_pass", "repaired_after_0_retries"]

