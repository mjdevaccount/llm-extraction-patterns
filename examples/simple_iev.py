"""Simple IEV example: Extract deal information from text."""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from patterns.core import create_llm_client
from patterns.patterns.iev import IntelligenceNode, ExtractionNode, ValidationNode

# Load environment variables
load_dotenv()


class DealRecord(BaseModel):
    """Schema for extracted deal information."""

    counterparty: str = Field(description="The counterparty name")
    notional: float = Field(gt=0, description="The notional amount in base currency")
    product: str = Field(description="The financial product type")
    execution_date: datetime = Field(description="The execution/settlement date")


INTELLIGENCE_PROMPT = """Analyze this email for deal information:

{email}

Identify:
- Counterparty name
- Notional amount and currency
- Product type (swap, option, bond, etc.)
- Execution or settlement date
- Any other relevant deal details
"""

EXTRACTION_PROMPT = """Based on the following analysis, extract the deal information as structured JSON:

{analysis}

Return only valid JSON matching the provided schema.
"""


async def extract_deal(email: str):
    """Extract deal information from an email.

    Args:
        email: Email text containing deal information

    Returns:
        Validated DealRecord
    """
    # Initialize LLM client
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4")
    llm = create_llm_client(provider=provider, model=model)

    # Initialize nodes
    intel = IntelligenceNode(
        llm,
        INTELLIGENCE_PROMPT,
        required_state_keys=["email"],
        name="deal_intelligence",
    )

    extract = ExtractionNode(
        llm,
        EXTRACTION_PROMPT,
        output_schema=DealRecord,
        required_state_keys=["analysis"],
        name="deal_extraction",
    )

    validate = ValidationNode(
        output_schema=DealRecord,
        validation_rules={
            "date_not_future": lambda x: x.execution_date <= datetime.now(),
        },
        name="deal_validation",
    )

    # Run pipeline
    state = {"email": email}
    state = await intel.execute(state)
    state = await extract.execute(state)
    state = await validate.execute(state)

    return state["validated"]


if __name__ == "__main__":
    email = "We're selling $10M notional of swaps to Acme Corp, settlement next week."
    result = asyncio.run(extract_deal(email))
    print(f"Extracted deal: {result}")
    print(f"Counterparty: {result.counterparty}")
    print(f"Notional: ${result.notional:,.0f}")
    print(f"Product: {result.product}")
    print(f"Execution Date: {result.execution_date}")

