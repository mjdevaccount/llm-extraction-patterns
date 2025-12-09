"""
Prompt templates for Evaluator-Optimizer pattern.

These are used by the draft and optimization phases.
"""

# Code generation template
CODE_DRAFT_PROMPT = """
Write high-quality Python code that solves this problem:

{input}

Provide only the code, with proper formatting.
"""

CODE_OPTIMIZATION_PROMPT = """
Improve this Python code based on the feedback:

Original Problem:
{input}

Current Code:
{previous}

Feedback to Address:
{feedback}

Provide the improved code only, addressing all feedback points.
"""

# Content refinement template
CONTENT_DRAFT_PROMPT = """
Write clear, engaging content about:

{input}

Make it informative and well-structured.
"""

CONTENT_OPTIMIZATION_PROMPT = """
Refine this content based on feedback:

Original Topic:
{input}

Current Content:
{previous}

Feedback:
{feedback}

Provide improved content that addresses all feedback.
"""

# Summary refinement template
SUMMARY_DRAFT_PROMPT = """
Write a concise summary of:

{input}

Keep it brief but comprehensive.
"""

SUMMARY_OPTIMIZATION_PROMPT = """
Improve this summary based on feedback:

Original Content:
{input}

Current Summary:
{previous}

Feedback:
{feedback}

Provide a better summary that addresses the feedback.
"""
