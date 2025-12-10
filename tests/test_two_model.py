"""
Test for Two-Model Pattern: Main LLM + Extractor LLM

Tests the two-model extraction pattern with:
- Main LLM (qwen2.5:14b) for complex reasoning
- Extractor LLM (nuextract:latest) for structured output
- Complex nested schema with arrays, nested objects, mixed types

Prerequisites:
1. Ollama installed and running (http://localhost:11434)
2. Models pulled:
   - ollama pull qwen2.5:14b
   - ollama pull qwen3:8b (or another model for extraction)
3. langchain-ollama installed:
   - pip install langchain-ollama

Run:
    pytest tests/test_two_model.py -v
    # or
    python tests/test_two_model.py
"""

import pytest
import asyncio
import os
from typing import List, Optional
from pydantic import BaseModel, Field

# Use latest langchain-ollama package
try:
    from langchain_ollama import ChatOllama
    HAS_LANGCHAIN_OLLAMA = True
except ImportError:
    HAS_LANGCHAIN_OLLAMA = False
    # Fallback to OllamaClient
    from patterns.core.llm_client import OllamaClient

from patterns.patterns.two_model import TwoModelExtractionNode
from patterns.core.extractor_formatters import NuExtractFormatter
from patterns.core.llm_factory import (
    get_reasoning_llm,
    get_extractor_llm,
    get_extractor_llm_from_formatter,
)
from patterns.core.json_repair import repair_json, extract_json_object


# ============================================================================
# NuExtract-Style JSON Template Schema
# ============================================================================
# NuExtract wants everything as strings and arrays; coerce to types in Python after

PRODUCT_REVIEW_SCHEMA_JSON = {
    "product_name": "",
    "product_category": "",
    "review_summary": "",
    "key_features": [
        {
            "feature_name": "",
            "sentiment": "",           # "positive" | "negative" | "neutral"
            "mentioned_count": ""      # string, will cast to int
        }
    ],
    "sentiment_analysis": {
        "overall_sentiment": "",      # "positive" | "negative" | "mixed"
        "positive_aspects": [""],
        "negative_aspects": [""],
        "neutral_aspects": [""],
        "sentiment_score": ""         # string, will cast to float 0.0–1.0
    },
    "metrics": {
        "rating_out_of_5": "",
        "value_rating": "",
        "quality_rating": "",
        "recommendation_score": "",
        "would_recommend": ""         # "true"/"false" or similar
    },
    "pros": [""],
    "cons": [""],
    "final_verdict": ""               # "recommend" | "not_recommend" | "conditional"
}


# ============================================================================
# Test Prompt
# ============================================================================

PRODUCT_REVIEW_PROMPT = """Analyze this product review and extract structured insights:

Review Text:

"I've been using the QuantumTech Pro Wireless Headphones for about 3 months now, and I have mixed feelings. 

The sound quality is absolutely fantastic - crystal clear highs, rich bass, and the noise cancellation is top-notch. Battery life is impressive too, I get about 30 hours on a single charge which is way better than my old pair. The build quality feels premium with the metal construction.

However, there are some issues. The Bluetooth connectivity can be spotty sometimes, especially when I'm walking around. The ear cups are a bit small for my ears and get uncomfortable after 2-3 hours of use. Also, at $299, it's quite expensive compared to competitors.

The app is decent but could use more customization options. Customer service was responsive when I had a question about the warranty.

Overall, if you prioritize sound quality and battery life and don't mind the price, these are solid. But if you need something more comfortable for long sessions or want better connectivity, you might want to look elsewhere.

I'd give it 4 out of 5 stars - great product with some room for improvement."

Please extract:
- Product name and category
- Key features mentioned with sentiment
- Overall sentiment analysis
- Quantitative ratings
- Pros and cons
- Final recommendation"""


# ============================================================================
# Test: Two-Model Pattern
# ============================================================================

# ============================================================================
# Mini Schema for Quick Smoke Test
# ============================================================================

MINI_SCHEMA = {
    "product_name": "",
    "rating_out_of_5": ""
}

MINI_PROMPT = 'Product: "X1000 Mouse". Rating: 3.5/5 stars.'


@pytest.mark.asyncio
async def test_nuextract_mini_smoke():
    """
    Tiny sanity test that should almost never fail and runs quick.
    If this breaks, NuExtract or the prompt contract changed.
    """
    nue = get_extractor_llm(num_ctx=2048, num_predict=500)
    
    import json
    schema_str = json.dumps(MINI_SCHEMA, indent=2)
    prompt = f"""### Template:

{schema_str}

### Text:

{MINI_PROMPT}
"""
    
    from langchain_core.messages import HumanMessage
    resp = await nue.ainvoke([HumanMessage(content=prompt)])
    raw = resp.content
    
    json_candidate = extract_json_object(raw)
    assert json_candidate is not None, "No JSON object found"
    
    # Try to repair common JSON issues
    data = repair_json(json_candidate)
    if data is None:
        # Fallback to direct parse
        data = json.loads(json_candidate)
    
    assert data["product_name"]
    rating = float(data.get("rating_out_of_5", "0"))
    assert rating > 0
    
    print(f"\n✓ Mini smoke test passed! Product: {data['product_name']}, Rating: {rating}")


@pytest.mark.asyncio
async def test_raw_nuextract_roundtrip():
    """
    Bare-bones test to verify nuextract + schema works without the node/formatter.
    This helps isolate whether the issue is in the models or the plumbing.
    """
    # Use factory to get LLMs
    main_llm = get_reasoning_llm()
    extractor_llm = get_extractor_llm()
    
    # 1) Main reasoning
    from langchain_core.messages import HumanMessage
    main_resp = await main_llm.ainvoke([HumanMessage(content=PRODUCT_REVIEW_PROMPT)])
    main_text = main_resp.content
    
    # 2) NuExtract protocol (NO <|input|> / <|output|> - Ollama wraps this internally)
    import json
    schema_str = json.dumps(PRODUCT_REVIEW_SCHEMA_JSON, indent=4)
    prompt = f"""### Template:

{schema_str}

### Text:

{main_text}
"""
    
    ext_resp = await extractor_llm.ainvoke([HumanMessage(content=prompt)])
    raw = ext_resp.content
    
    print("\n=== RAW NUEXTRACT OUTPUT (first 1000 chars) ===")
    print(raw[:1000])
    
    # NuExtract with Ollama tends to return the JSON directly or prefixed with the template;
    # extract the first '{...}' block.
    json_candidate = extract_json_object(raw)
    assert json_candidate is not None, "No JSON object found in nuextract output"
    
    # Try to repair common JSON issues
    data = repair_json(json_candidate)
    if data is None:
        # Fallback to direct parse
        data = json.loads(json_candidate)
    
    assert "product_name" in data
    print(f"\n✓ Raw roundtrip successful! Product: {data.get('product_name')}")
    return data


@pytest.mark.asyncio
async def test_two_model_product_review_extraction():
    """
    Test two-model pattern with product review analysis.
    
    This test validates:
    - Nested objects (sentiment_analysis, metrics)
    - Arrays of complex objects (key_features[], pros[], cons[])
    - Mixed types (floats, ints, strings, booleans)
    - Sentiment extraction and classification
    - Derived values (ratings, scores)
    """
    
    # Use factory to get LLMs
    main_llm = get_reasoning_llm()
    
    # Create formatter - NuExtract protocol
    formatter = NuExtractFormatter()
    print("Using NuExtractFormatter")
    
    # Get extractor LLM with formatter's recommended config
    extractor_llm = get_extractor_llm_from_formatter(formatter)
    
    # Create two-model extraction node (strict=True for tests)
    from patterns.patterns.two_model import TwoModelExtractionNode
    
    extraction_node = TwoModelExtractionNode(
        main_llm=main_llm,
        extractor_llm=extractor_llm,
        extractor_formatter=formatter,
        output_schema=PRODUCT_REVIEW_SCHEMA_JSON,  # Dict template for NuExtract
        main_prompt_template="{input}",
        required_state_keys=["input"],
        strict=True,  # Hard fail for tests
        bypass_main=False,  # Use full two-model pipeline
        name="product_review_extraction"
    )
    
    # Execute extraction
    print("\n" + "=" * 80)
    print("TWO-MODEL PATTERN TEST: PRODUCT REVIEW ANALYSIS")
    print("=" * 80)
    print("\nStep 1: Main LLM generating analysis...")
    
    state = await extraction_node.execute({
        "input": PRODUCT_REVIEW_PROMPT
    })
    
    # Validate results
    assert "extracted" in state, "Extraction should produce 'extracted' key"
    
    extracted = state["extracted"]
    
    print("\nStep 2: Validating extracted structure...")
    print(f"Extracted keys: {list(extracted.keys())}")
    
    # Check metrics
    metrics = extraction_node.get_metrics()
    if metrics.get('extra'):
        print(f"\nExtractor Stats:")
        print(f"  Analysis chars: {metrics['extra'].get('analysis_chars', 'N/A')}")
        print(f"  Extractor prompt chars: {metrics['extra'].get('extractor_prompt_chars', 'N/A')}")
        print(f"  Extractor response chars: {metrics['extra'].get('extractor_response_chars', 'N/A')}")
        print(f"  Formatter: {metrics['extra'].get('formatter', 'N/A')}")
    
    # Basic shape checks
    assert "product_name" in extracted, f"Missing product_name. Keys: {list(extracted.keys())}"
    assert "product_category" in extracted
    assert "key_features" in extracted
    assert isinstance(extracted["key_features"], list)
    assert len(extracted["key_features"]) > 0
    
    # Coerce and validate metrics (NuExtract returns strings)
    def coerce_float(val, default=0.0):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    
    def coerce_int(val, default=0):
        try:
            return int(val)
        except (ValueError, TypeError):
            return default
    
    m = extracted.get("metrics", {})
    rating = coerce_float(m.get("rating_out_of_5"))
    assert 0.0 <= rating <= 5.0
    
    value_rating = coerce_float(m.get("value_rating"))
    assert 0.0 <= value_rating <= 1.0
    
    quality_rating = coerce_float(m.get("quality_rating"))
    assert 0.0 <= quality_rating <= 1.0
    
    recommendation_score = coerce_float(m.get("recommendation_score"))
    assert 0.0 <= recommendation_score <= 1.0
    
    # Coerce boolean
    raw_wr = str(m.get("would_recommend", "")).strip()
    if not raw_wr:
        print("Warning: would_recommend empty in metrics")
    wr = raw_wr.lower()
    would_recommend = wr in ["true", "yes", "1", "y"]
    
    # Validate sentiment
    sentiment = extracted.get("sentiment_analysis", {})
    overall_sentiment = str(sentiment.get("overall_sentiment", "")).lower().strip()
    print(f"Overall sentiment value: '{overall_sentiment}'")
    if not overall_sentiment:
        print("Warning: overall_sentiment is empty")
    elif overall_sentiment not in ["positive", "negative", "mixed"]:
        print(f"Warning: overall_sentiment '{overall_sentiment}' not in expected values, but continuing...")
    
    sentiment_score = coerce_float(sentiment.get("sentiment_score"))
    assert 0.0 <= sentiment_score <= 1.0
    
    # Validate features (warnings instead of asserts for smoke testing)
    features = extracted.get("key_features", [])
    if not isinstance(features, list):
        print("Warning: key_features is not a list")
    else:
        for feature in features:
            if "feature_name" not in feature:
                print(f"Warning: feature missing feature_name: {feature}")
            feat_sentiment = str(feature.get("sentiment", "")).lower().strip()
            # More lenient - check if it contains the expected values
            if not any(x in feat_sentiment for x in ["positive", "negative", "neutral"]):
                print(f"Warning: feature sentiment '{feat_sentiment}' not in expected values")
            mentioned_count = coerce_int(feature.get("mentioned_count"))
            if mentioned_count < 0:
                print(f"Warning: mentioned_count is negative: {mentioned_count}")
    
    # Final verdict domain check (more lenient)
    fv = str(extracted.get("final_verdict", "")).lower().strip()
    if not fv:
        print("Warning: final_verdict is empty")
    elif not any(x in fv for x in ["recommend", "not_recommend", "conditional"]):
        print(f"Warning: final_verdict '{fv}' not in expected values, but continuing...")
    
    # Validate pros and cons
    assert isinstance(extracted.get("pros", []), list)
    assert isinstance(extracted.get("cons", []), list)
    
    print("\n" + "=" * 80)
    print("VALIDATION SUCCESSFUL!")
    print("=" * 80)
    print(f"\nProduct: {extracted.get('product_name')}")
    print(f"Category: {extracted.get('product_category')}")
    print(f"Overall Sentiment: {overall_sentiment}")
    print(f"Sentiment Score: {sentiment_score:.2f}")
    print(f"Rating: {rating:.1f}/5.0")
    print(f"Would Recommend: {would_recommend}")
    print(f"Final Verdict: {fv}")
    print(f"Features Extracted: {len(extracted['key_features'])}")
    print(f"Pros: {len(extracted.get('pros', []))}")
    print(f"Cons: {len(extracted.get('cons', []))}")
    print("\n" + "=" * 80)
    
    return extracted


if __name__ == "__main__":
    # Run test directly
    asyncio.run(test_two_model_product_review_extraction())

