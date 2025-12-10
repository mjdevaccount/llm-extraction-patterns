"""Minimal working example for IEV pattern."""

import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from ...core import create_llm_client
from .graph import create_iev_graph
from .prompts import INTELLIGENCE_PROMPT, EXTRACTION_PROMPT, VERIFICATION_PROMPT

# Load environment variables
load_dotenv()


class DealRecord(BaseModel):
    """Example schema for deal extraction."""

    counterparty: str
    notional: float = Field(gt=0)
    product: str
    execution_date: datetime


async def main():
    """Run IEV pattern example."""
    # Initialize LLM client
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4")
    llm = create_llm_client(provider=provider, model=model)

    # Create IEV graph
    graph = create_iev_graph(
        llm_client=llm,
        output_schema=DealRecord,
        intelligence_prompt=INTELLIGENCE_PROMPT,
        extraction_prompt=EXTRACTION_PROMPT,
        verification_prompt=VERIFICATION_PROMPT,
    )

    # Example input
    input_text = "We're selling $10M notional of swaps to Acme Corp, settlement next week."

    print(f"Input: {input_text}\n")
    print("Running IEV pattern...\n")

    # Execute graph
    if callable(graph) and not hasattr(graph, "invoke"):
        # Simple executor fallback
        result = await graph(input_text)
    else:
        # LangGraph
        result = await graph.ainvoke({"input": input_text, "verification_status": "PENDING"})

    # Display results
    print(f"Analysis: {result.get('analysis', 'N/A')[:200]}...\n")
    if result.get("validated"):
        deal = result["validated"]
        print(f"✓ Validated Deal:")
        print(f"  Counterparty: {deal.counterparty}")
        print(f"  Notional: ${deal.notional:,.0f}")
        print(f"  Product: {deal.product}")
        print(f"  Execution Date: {deal.execution_date}")
    else:
        print(f"✗ Verification Status: {result.get('verification_status', 'UNKNOWN')}")
        if result.get("error"):
            print(f"  Error: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())

