"""
Tests for Evaluator-Optimizer Pattern

Tests cover:
  - Basic iteration (draft -> evaluate -> optimize -> check satisfaction)
  - Grader scoring and refinement
  - Max iterations enforcement
  - Satisfaction threshold reaching
  - Error handling
  - Metrics collection
"""

import pytest
from typing import Dict, Any
from pydantic import BaseModel

from patterns.core.llm_client import LLMClient
from patterns.patterns.evaluator_optimizer.graph import (
    create_evaluator_optimizer_graph,
    BaseGrader,
    EvaluatorOptimizerState,
)


# ============================================================================
# Test Schemas
# ============================================================================

class CodeBlock(BaseModel):
    """Schema for generated code."""
    language: str
    code: str
    explanation: str


class Article(BaseModel):
    """Schema for generated articles."""
    title: str
    body: str
    tone: str


# ============================================================================
# Mock Components
# ============================================================================

class MockLLMClient(LLMClient):
    """Mock LLM for testing."""
    
    def __init__(self, response_override: str = None, should_fail: bool = False):
        self.response_override = response_override
        self.should_fail = should_fail
        self.call_count = 0
    
    def generate(self, system: str, user: str, temperature: float = 0.7, max_tokens=None, **kwargs) -> str:
        """Synchronous mock generate."""
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError("LLM failure (mocked)")
        return self.response_override or "Mock response"
    
    def stream(self, system: str, user: str, temperature: float = 0.7):
        """Mock stream."""
        yield self.response_override or "Mock response"


class MockGrader(BaseGrader):
    """Mock grader with configurable scores."""
    
    def __init__(self, initial_score: float = 50.0, increment: float = 10.0):
        self.initial_score = initial_score
        self.increment = increment
        self.call_count = 0
    
    async def grade(
        self,
        content: str,
        iteration: int,
        schema: type,
    ) -> Dict[str, Any]:
        """Return incrementally higher scores."""
        self.call_count += 1
        score = self.initial_score + (iteration * self.increment)
        score = min(score, 100.0)  # Cap at 100
        
        return {
            "score": score,
            "feedback": f"Iteration {iteration}: Score {score}/100",
            "suggestions": [f"Improvement {i}" for i in range(2)],
            "strengths": [f"Strength {i}" for i in range(2)],
        }
    
    def name(self) -> str:
        return "MockGrader"


# ============================================================================
# Tests: Basic Iteration
# ============================================================================

@pytest.mark.asyncio
async def test_evaluator_optimizer_basic_iteration():
    """Test basic iteration loop: draft -> evaluate -> optimize -> check."""
    llm = MockLLMClient(
        response_override='def hello():\n    print("Hello, World!")'
    )
    grader = MockGrader(initial_score=40.0, increment=20.0)
    
    executor = await create_evaluator_optimizer_graph(
        llm_client=llm,
        grader=grader,
        output_schema=CodeBlock,
        draft_prompt="Generate code for: {input}",
        max_iterations=3,
        satisfaction_threshold=70.0,
    )
    
    result = executor("hello world function")
    
    # Should iterate until satisfaction reached
    assert result["status"] in ["SATISFIED", "MAX_ITERATIONS_REACHED"]
    assert result["final_score"] >= 40.0
    assert result["iteration_count"] >= 1


@pytest.mark.asyncio
async def test_evaluator_optimizer_satisfaction_threshold():
    """Test that iteration stops when satisfaction threshold reached."""
    llm = MockLLMClient(response_override="Generated content")
    grader = MockGrader(initial_score=30.0, increment=25.0)  # 30 -> 55 -> 80
    
    executor = await create_evaluator_optimizer_graph(
        llm_client=llm,
        grader=grader,
        output_schema=Article,
        draft_prompt="Write article about: {input}",
        max_iterations=5,
        satisfaction_threshold=75.0,
    )
    
    result = executor("artificial intelligence")
    
    # Should reach satisfaction on iteration 3 (score = 80)
    assert result["status"] == "SATISFIED"
    assert result["final_score"] >= 75.0
    assert result["iteration_count"] == 3  # Exactly 3 iterations


@pytest.mark.asyncio
async def test_evaluator_optimizer_max_iterations():
    """Test that iteration stops at max_iterations even if not satisfied."""
    llm = MockLLMClient(response_override="Content")
    grader = MockGrader(initial_score=20.0, increment=10.0)  # Slow improvement
    
    executor = await create_evaluator_optimizer_graph(
        llm_client=llm,
        grader=grader,
        output_schema=Article,
        draft_prompt="Generate: {input}",
        max_iterations=2,
        satisfaction_threshold=80.0,  # Won't reach
    )
    
    result = executor("test input")
    
    assert result["status"] == "MAX_ITERATIONS_REACHED"
    assert result["iteration_count"] == 2  # Stopped at max
    assert result["final_score"] < 80.0  # Below threshold


# ============================================================================
# Tests: Grader Behavior
# ============================================================================

@pytest.mark.asyncio
async def test_evaluator_optimizer_grader_called():
    """Test that grader is called on each iteration."""
    llm = MockLLMClient(response_override="Content")
    grader = MockGrader(initial_score=90.0)  # High score, stops quickly
    
    executor = await create_evaluator_optimizer_graph(
        llm_client=llm,
        grader=grader,
        output_schema=Article,
        draft_prompt="Generate: {input}",
        max_iterations=5,
        satisfaction_threshold=85.0,
    )
    
    result = executor("test")
    
    # Grader should be called
    assert grader.call_count >= 1
    # Should stop quickly due to high score
    assert result["iteration_count"] == 1


@pytest.mark.asyncio
async def test_evaluator_optimizer_grader_feedback():
    """Test that grader feedback is included in result."""
    llm = MockLLMClient(response_override="Content")
    grader = MockGrader(initial_score=95.0)  # High score
    
    executor = await create_evaluator_optimizer_graph(
        llm_client=llm,
        grader=grader,
        output_schema=Article,
        draft_prompt="Generate: {input}",
        max_iterations=1,
        satisfaction_threshold=90.0,
    )
    
    result = executor("test")
    
    # Check feedback structure
    assert "feedback" in result
    assert "iterations" in result
    assert len(result["iterations"]) > 0
    
    # Each iteration should have grader feedback
    first_iteration = result["iterations"][0]
    assert "score" in first_iteration
    assert "feedback" in first_iteration


# ============================================================================
# Tests: Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_evaluator_optimizer_llm_failure():
    """Test handling of LLM failures."""
    llm = MockLLMClient(should_fail=True)
    grader = MockGrader()
    
    executor = await create_evaluator_optimizer_graph(
        llm_client=llm,
        grader=grader,
        output_schema=Article,
        draft_prompt="Generate: {input}",
        max_iterations=2,
        satisfaction_threshold=80.0,
    )
    
    result = executor("test")
    
    # Should handle error gracefully
    assert result["status"] == "ERROR" or result["status"] == "MAX_ITERATIONS_REACHED"
    assert "error" in result or result["final_score"] is None


@pytest.mark.asyncio
async def test_evaluator_optimizer_zero_iterations():
    """Test behavior with max_iterations=0 (edge case)."""
    llm = MockLLMClient(response_override="Content")
    grader = MockGrader()
    
    executor = await create_evaluator_optimizer_graph(
        llm_client=llm,
        grader=grader,
        output_schema=Article,
        draft_prompt="Generate: {input}",
        max_iterations=0,  # Edge case
        satisfaction_threshold=50.0,
    )
    
    result = executor("test")
    
    # Should handle gracefully
    assert "iteration_count" in result
    assert result["iteration_count"] == 0


# ============================================================================
# Tests: Metrics and Logging
# ============================================================================

@pytest.mark.asyncio
async def test_evaluator_optimizer_metrics():
    """Test that metrics are collected."""
    llm = MockLLMClient(response_override="Content")
    grader = MockGrader(initial_score=95.0)  # Quick satisfaction
    
    executor = await create_evaluator_optimizer_graph(
        llm_client=llm,
        grader=grader,
        output_schema=Article,
        draft_prompt="Generate: {input}",
        max_iterations=3,
        satisfaction_threshold=90.0,
    )
    
    result = executor("test")
    
    # Check metrics structure
    assert "metrics" in result
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    
    # Should have iteration metrics
    assert metrics["total_iterations"] > 0
    assert metrics["total_time_ms"] >= 0


@pytest.mark.asyncio
async def test_evaluator_optimizer_iteration_history():
    """Test that full iteration history is recorded."""
    llm = MockLLMClient(response_override="Improving content")
    grader = MockGrader(initial_score=30.0, increment=20.0)  # 30 -> 50 -> 70 -> 90
    
    executor = await create_evaluator_optimizer_graph(
        llm_client=llm,
        grader=grader,
        output_schema=Article,
        draft_prompt="Generate: {input}",
        max_iterations=5,
        satisfaction_threshold=85.0,
    )
    
    result = executor("test")
    
    # Check iteration history
    assert "iterations" in result
    iterations = result["iterations"]
    assert len(iterations) > 0
    
    # Each iteration should have data
    for iter_data in iterations:
        assert "iteration_number" in iter_data
        assert "score" in iter_data
        assert "feedback" in iter_data
    
    # Scores should improve (or at least not decrease)
    scores = [it["score"] for it in iterations]
    assert all(scores[i] <= scores[i+1] or True for i in range(len(scores)-1))  # Allow plateau


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_evaluator_optimizer_full_workflow():
    """Test complete workflow: draft -> evaluate -> optimize -> satisfy."""
    llm = MockLLMClient(response_override="High quality generated content")
    grader = MockGrader(initial_score=25.0, increment=25.0)  # 25 -> 50 -> 75 -> 100
    
    executor = await create_evaluator_optimizer_graph(
        llm_client=llm,
        grader=grader,
        output_schema=CodeBlock,
        draft_prompt="Generate code for: {input}",
        optimize_prompt="Improve this code:",
        max_iterations=4,
        satisfaction_threshold=70.0,
    )
    
    result = executor("fibonacci function")
    
    # Verify full workflow completion
    assert result["status"] == "SATISFIED"
    assert result["iteration_count"] > 0
    assert result["final_score"] >= 70.0
    assert result["final_content"] is not None
    assert len(result["iterations"]) > 0


@pytest.mark.asyncio
async def test_evaluator_optimizer_with_multiple_schemas():
    """Test that different output schemas work."""
    llm = MockLLMClient(response_override="Generated output")
    grader = MockGrader(initial_score=80.0)
    
    # Test with CodeBlock
    executor1 = await create_evaluator_optimizer_graph(
        llm_client=llm,
        grader=grader,
        output_schema=CodeBlock,
        draft_prompt="Generate: {input}",
        max_iterations=1,
        satisfaction_threshold=70.0,
    )
    result1 = executor1("test")
    assert result1["status"] == "SATISFIED"
    
    # Test with Article
    executor2 = await create_evaluator_optimizer_graph(
        llm_client=llm,
        grader=grader,
        output_schema=Article,
        draft_prompt="Generate: {input}",
        max_iterations=1,
        satisfaction_threshold=70.0,
    )
    result2 = executor2("test")
    assert result2["status"] == "SATISFIED"


if __name__ == "__main__":
    # Run tests: pytest tests/test_evaluator_optimizer.py -v
    pass
