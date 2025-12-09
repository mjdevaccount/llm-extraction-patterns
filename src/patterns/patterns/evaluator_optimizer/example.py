"""
Evaluator-Optimizer Example: Code Review with Iterative Refinement

This example uses the Evaluator-Optimizer pattern to:
  1. Generate Python code to solve a problem
  2. Evaluate code quality (docstrings, error handling, type hints)
  3. Iteratively refine until quality score >= 80

Run with:
  PROVIDER=openai python -m patterns.patterns.evaluator_optimizer.example
"""

import asyncio
import os
from patterns.core import create_llm_client
from .graph import create_evaluator_optimizer_graph, CodeQualityGrader


DRAFT_PROMPT = """
Write Python code to solve this problem:

{input}

Provide only the code, no explanation.
"""

OPTIMIZATION_PROMPT = """
Based on this feedback, improve the Python code:

Original Problem:
{input}

Previous Code:
{previous}

Feedback to Address:
{feedback}

Provide the improved code only, no explanation.
"""


async def main():
    # Setup
    provider = os.getenv("PROVIDER", "openai")
    model = os.getenv("MODEL", None)
    
    llm = create_llm_client(provider=provider, model=model)
    grader = CodeQualityGrader()
    
    # Create executor
    executor = await create_evaluator_optimizer_graph(
        llm_client=llm,
        grader=grader,
        draft_prompt=DRAFT_PROMPT,
        optimization_prompt=OPTIMIZATION_PROMPT,
        max_iterations=3,
        satisfaction_threshold=80.0,
    )
    
    # Run example
    problem = "Write a function that parses a JSON string and returns a Python dict."
    
    print("\n" + "="*70)
    print("EVALUATOR-OPTIMIZER EXAMPLE: Code Review")
    print("="*70)
    print(f"\nProblem: {problem}")
    print("\nStarting code review cycle...\n")
    
    result = await executor(problem)
    
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iteration']}")
    print(f"Final Score: {result['score']:.1f}/100")
    print(f"\nFinal Code:\n{result['draft']}")
    
    if result['status'] == 'ERROR':
        print(f"\nError: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())
