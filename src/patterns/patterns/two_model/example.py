"""
Example: Two-Model Pattern for Valuation Extraction

Demonstrates the two-model pattern:
1. Main LLM (Qwen2.5-14B) generates valuation analysis
2. NuExtract (7B) extracts structured JSON

This pattern is efficient for local models on 16GB GPU:
- Main LLM: ~8GB (Q4 quantization)
- NuExtract: ~5-6GB
- Total: ~14-15GB (fits with headroom)
"""

import asyncio
from typing import Optional
from pydantic import BaseModel, Field

# Example: Import from your project structure
# from patterns.patterns.two_model import TwoModelExtractionNode, NuExtractFormatter
# from patterns.patterns.iev.nodes.intelligence import IntelligenceNode

# For this example, we'll show the pattern structure
# In practice, you'd import from the actual modules


class ValuationResult(BaseModel):
    """Structured output schema for valuation."""
    valuation_methodology: str = Field(description="Method used (e.g., Black-Scholes)")
    input_parameters: dict = Field(description="Input parameters used")
    calculated_values: dict = Field(description="Calculated valuation results")
    risk_metrics: Optional[dict] = Field(default=None, description="Greeks and risk metrics")
    confidence: str = Field(description="Confidence level in the valuation")


async def example_two_model_valuation():
    """
    Example workflow using two-model pattern.
    
    Pattern Flow:
    1. Main LLM generates unstructured valuation analysis
    2. NuExtract formats analysis into structured JSON
    3. Result is validated and ready for downstream use
    """
    
    # In practice, you'd load your models:
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # import torch
    # 
    # main_model = AutoModelForCausalLM.from_pretrained(
    #     "Qwen/Qwen2.5-14B-Instruct",
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )
    # 
    # extractor_model = AutoModelForCausalLM.from_pretrained(
    #     "numind/NuExtract-large",
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto"
    # ).eval()
    
    # For this example, we'll show the pattern structure
    print("=" * 80)
    print("TWO-MODEL PATTERN EXAMPLE")
    print("=" * 80)
    print("\nPattern: Main LLM → Unstructured Text → Extractor LLM → Structured JSON")
    print("\n1. Main LLM (Qwen2.5-14B) generates analysis:")
    print("   - Temperature: 0.7 (creative reasoning)")
    print("   - Output: Unstructured text with valuation details")
    
    print("\n2. NuExtract (7B) formats to structured JSON:")
    print("   - Temperature: 0.0 (deterministic extraction)")
    print("   - Input: Unstructured text + JSON schema")
    print("   - Output: Structured JSON matching schema")
    
    print("\n3. Result: Validated Pydantic model ready for use")
    print("=" * 80)
    
    # Example usage pattern (commented out - requires actual models):
    """
    from patterns.patterns.two_model import TwoModelExtractionNode, NuExtractFormatter
    
    # Create formatter
    formatter = NuExtractFormatter()
    
    # Create node
    extraction_node = TwoModelExtractionNode(
        main_llm=main_model,  # Your main LLM
        extractor_llm=extractor_model,  # Your extractor LLM
        extractor_formatter=formatter,
        output_schema=ValuationResult,
        main_prompt_template="Value this option: {input}",
        required_state_keys=["input"],
        name="valuation_extraction"
    )
    
    # Execute
    state = await extraction_node.execute({
        "input": """
        European Call Option on Apple Stock
        - Current underlying price: $150
        - Strike price: $155
        - Time to expiration: 90 days
        - Risk-free rate: 4.5% annualized
        - Volatility (implied): 22%
        - No dividends
        - Use Black-Scholes methodology
        """
    })
    
    # Use structured result
    result = state["extracted"]
    print(f"Valuation: ${result['calculated_values']['final_valuation']}")
    print(f"Method: {result['valuation_methodology']}")
    """
    
    print("\nExample pattern structure shown above.")
    print("To use, implement with your actual LLM models.")


if __name__ == "__main__":
    asyncio.run(example_two_model_valuation())

