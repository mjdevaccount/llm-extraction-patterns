"""
Evaluator-Optimizer Pattern Graph Definition

State machine:
  1. DRAFT: Generate initial output
  2. EVALUATE: Score and critique output
  3. OPTIMIZE: Refine based on feedback
  4. CHECK: Did feedback improve score? Loop back to OPTIMIZE or finish
"""

import json
import logging
from typing import Any, Dict, Literal, TypedDict, Optional
from abc import ABC, abstractmethod

from pydantic import BaseModel
from ...core.llm_client import LLMClient

logger = logging.getLogger(__name__)


class EvaluatorOptimizerState(TypedDict):
    """State for Evaluator-Optimizer pattern."""
    input: str
    draft: str
    evaluation: Dict[str, Any]
    score: float
    feedback: str
    optimized: str
    iteration: int
    max_iterations: int
    satisfaction_threshold: float
    status: Literal["PENDING", "SATISFIED", "MAX_ITERATIONS", "ERROR"]
    error: Optional[str]


class BaseGrader(ABC):
    """
    Abstract grader for evaluating outputs.
    
    Implement for domain-specific scoring:
      - CodeQualityGrader: Score Python code
      - ContentQualityGrader: Score written content
      - CustomGrader: Your own scoring logic
    """
    
    @abstractmethod
    async def grade(self, output: str, context: str = "") -> Dict[str, Any]:
        """
        Grade output and provide feedback.
        
        Args:
            output: Text to grade
            context: Optional context for grading
        
        Returns:
            Dict with keys:
              - score (float 0-100)
              - feedback (str)
              - issues (list of specific problems)
        """
        pass


class CodeQualityGrader(BaseGrader):
    """Grader for code quality (basic implementation)."""
    
    async def grade(self, output: str, context: str = "") -> Dict[str, Any]:
        """
        Grade code for common issues.
        
        Checks for:
          - Syntax (basic)
          - Style (docstrings, naming)
          - Logic (error handling)
        """
        issues = []
        score = 100.0
        
        # Check for docstrings
        if "def " in output and "\"\"\"" not in output:
            issues.append("Missing docstrings")
            score -= 10
        
        # Check for error handling
        if "try:" not in output and "except" not in output:
            issues.append("No error handling")
            score -= 15
        
        # Check for type hints
        if "->" not in output and "def " in output:
            issues.append("Missing type hints")
            score -= 10
        
        feedback = f"Score: {score}/100"
        if issues:
            feedback += f"\nIssues: {', '.join(issues)}"
        
        return {
            "score": score,
            "feedback": feedback,
            "issues": issues,
        }


async def create_evaluator_optimizer_graph(
    llm_client: LLMClient,
    grader: BaseGrader,
    draft_prompt: str,
    optimization_prompt: str,
    max_iterations: int = 3,
    satisfaction_threshold: float = 80.0,
):
    """
    Create Evaluator-Optimizer graph.
    
    Args:
        llm_client: LLM for draft and optimization
        grader: Grader for evaluation
        draft_prompt: Template for generating draft
        optimization_prompt: Template for refining
        max_iterations: Max refinement loops
        satisfaction_threshold: Score threshold (0-100) to stop iterating
    
    Returns:
        Async function: invoke(input_text) -> final state
    """
    
    async def draft_node(state: EvaluatorOptimizerState) -> Dict[str, Any]:
        """Generate initial draft."""
        logger.info("[EO] Draft phase")
        try:
            prompt = draft_prompt.format(input=state["input"])
            draft = await llm_client.generate(
                system="You are a helpful assistant that generates high-quality output.",
                user=prompt,
            )
            return {"draft": draft}
        except Exception as e:
            logger.exception(f"[EO] Draft error: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def evaluate_node(state: EvaluatorOptimizerState) -> Dict[str, Any]:
        """Evaluate draft and score."""
        logger.info(f"[EO] Evaluate phase (iteration {state['iteration']})") 
        try:
            evaluation = await grader.grade(
                output=state["draft"],
                context=state["input"],
            )
            return {
                "evaluation": evaluation,
                "score": evaluation.get("score", 0.0),
                "feedback": evaluation.get("feedback", ""),
            }
        except Exception as e:
            logger.exception(f"[EO] Evaluation error: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def optimize_node(state: EvaluatorOptimizerState) -> Dict[str, Any]:
        """Refine based on feedback."""
        logger.info(f"[EO] Optimize phase (iteration {state['iteration']})") 
        try:
            prompt = optimization_prompt.format(
                input=state["input"],
                previous=state["draft"],
                feedback=state["feedback"],
            )
            optimized = await llm_client.generate(
                system="You are a helpful assistant that improves work based on feedback.",
                user=prompt,
            )
            return {
                "optimized": optimized,
                "draft": optimized,  # Replace draft for next iteration
                "iteration": state["iteration"] + 1,
            }
        except Exception as e:
            logger.exception(f"[EO] Optimization error: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    async def check_satisfaction_node(state: EvaluatorOptimizerState) -> Dict[str, Any]:
        """Check if output satisfies threshold."""
        logger.info(f"[EO] Satisfaction check: {state['score']:.1f}/{satisfaction_threshold}")
        
        if state["score"] >= satisfaction_threshold:
            logger.info("[EO] Satisfied - stopping")
            return {"status": "SATISFIED"}
        elif state["iteration"] >= state["max_iterations"]:
            logger.info("[EO] Max iterations reached")
            return {"status": "MAX_ITERATIONS"}
        else:
            logger.info("[EO] Need more refinement - continuing")
            return {}  # Continue to next iteration
    
    async def simple_executor(input_text: str) -> EvaluatorOptimizerState:
        """Simple executor without LangGraph."""
        state: EvaluatorOptimizerState = {
            "input": input_text,
            "draft": "",
            "evaluation": {},
            "score": 0.0,
            "feedback": "",
            "optimized": "",
            "iteration": 0,
            "max_iterations": max_iterations,
            "satisfaction_threshold": satisfaction_threshold,
            "status": "PENDING",
            "error": None,
        }
        
        # Draft phase
        state.update(await draft_node(state))
        if state.get("status") == "ERROR":
            return state
        
        # Refinement loop
        while True:
            # Evaluate
            state.update(await evaluate_node(state))
            if state.get("status") == "ERROR":
                return state
            
            # Check satisfaction
            check_result = await check_satisfaction_node(state)
            state.update(check_result)
            if state["status"] in ["SATISFIED", "MAX_ITERATIONS"]:
                break
            
            # Optimize
            state.update(await optimize_node(state))
            if state.get("status") == "ERROR":
                return state
        
        return state
    
    return simple_executor
