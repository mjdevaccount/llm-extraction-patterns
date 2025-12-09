"""Prompts for IEV pattern."""

INTELLIGENCE_PROMPT = """Analyze the following input and provide a comprehensive analysis:

{input}

Consider:
- Key information and entities
- Context and relationships
- Potential ambiguities or risks
- Recommended next steps
"""

EXTRACTION_PROMPT = """Based on the following analysis, extract structured data:

{analysis}

Extract the relevant information as structured JSON matching the provided schema.
"""

VERIFICATION_PROMPT = """Verify the following extracted data:

{extracted}

Check for:
- Completeness: All required fields present
- Correctness: Values are reasonable and valid
- Consistency: Data aligns with the original input
- Safety: No potentially harmful or incorrect actions

Respond with:
- 'APPROVED' if the data is valid and safe to use
- 'REJECTED' with a reason if there are issues
"""

