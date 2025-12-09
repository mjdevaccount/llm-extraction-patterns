"""Extraction Node for SOLID Design.

Single Responsibility: Extract JSON from text and repair if needed.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Type
from datetime import datetime

from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from .base_nodes import BaseNode, NodeExecutionError, NodeStatus
from ..abstractions import ILLMProvider
from ..strategies import (
    IncrementalRepairStrategy,
    LLMRepairStrategy,
    RegexRepairStrategy,
)
from ..llm_adapter import LangChainLLMAdapter

logger = logging.getLogger(__name__)


class ExtractionNode(BaseNode):
    """
    Structured output extraction node.
    
    Single Responsibility: Extract JSON from text and repair if needed.
    
    Temperature: 0.1 (deterministic, consistent)
    
    Input Requirements:
        - state["analysis"]: Unstructured text from intelligence node
    
    Output:
        - Adds `extracted` key with parsed/repaired JSON as dict
        - Adds `extraction_warnings` if repairs were needed
    
    Repair Strategies (applied in order):
        1. Direct JSON parse
        2. Incremental repair (close braces, remove trailing commas)
        3. LLM repair (ask model to fix JSON)
        4. Regex extraction (pull out key fields manually)
    
    Design Pattern: Strategy
        Multiple repair strategies tried in sequence.
        Different strategies can be enabled/disabled.
    
    Example:
        extraction = ExtractionNode(
            llm=extraction_llm,  # Low-temperature LLM
            prompt_template="Extract JSON from: {analysis}\nSchema: {schema}",
            output_schema=AdoptionPrediction,
            name="extraction"
        )
        
        state = await extraction.execute({"analysis": "Market will grow..."})
        # state["extracted"] = {
        #     "adoption_timeline_months": 18,
        #     "disruption_score": 8.5,
        #     ...
        # }
    """
    
    def __init__(
        self,
        llm: Any,  # LangChain ChatModel with low temperature
        prompt_template: str,
        output_schema: Type[BaseModel],  # Pydantic model
        json_repair_strategies: List[str] = None,
        name: str = "extraction",
        description: str = "Extract structured data from analysis",
    ):
        """
        Initialize extraction node.
        
        Args:
            llm: LangChain ChatModel (should have temperature=0.1)
            prompt_template: Template for extraction prompt
            output_schema: Pydantic BaseModel defining expected JSON structure
            json_repair_strategies: List of repair strategies to use
                Options: ["incremental_repair", "llm_repair", "regex_fallback"]
            name: Node identifier
            description: Human-readable description
        """
        super().__init__(name=name, description=description)
        self.llm = llm
        self.prompt_template = prompt_template
        self.output_schema = output_schema
        self.json_repair_strategies = json_repair_strategies or [
            "incremental_repair",
            "llm_repair",
            "regex_fallback",
        ]
        
        # SOLID Refactoring: Initialize strategy objects
        llm_adapter = LangChainLLMAdapter(llm)
        strategy_map = {
            "incremental_repair": IncrementalRepairStrategy(),
            "llm_repair": LLMRepairStrategy(llm_adapter),
            "regex_fallback": RegexRepairStrategy(),
        }
        self._repair_strategies = []
        for strategy_name in self.json_repair_strategies:
            if strategy_name in strategy_map:
                self._repair_strategies.append(strategy_map[strategy_name])
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute extraction.
        
        Process:
            1. Format prompt with analysis text
            2. Invoke LLM for JSON extraction
            3. Parse JSON (with repair if needed)
            4. Return extracted data
        
        Args:
            state: Must have "analysis" key
        
        Returns:
            State with added keys:
                - extracted: Parsed JSON dict
                - extraction_warnings: List of warnings (if repairs applied)
        
        Raises:
            NodeExecutionError: If extraction fails
        """
        start_time = datetime.now()
        self.metrics.status = NodeStatus.RUNNING
        self.metrics.input_keys = ["analysis"]
        
        try:
            if "analysis" not in state:
                raise NodeExecutionError(
                    node_name=self.name,
                    reason="Missing 'analysis' from intelligence node",
                    state=state
                )
            
            # Format prompt
            schema_str = self.output_schema.__name__
            prompt_text = self.prompt_template.format(
                analysis=state["analysis"][:1500],  # Limit context
                schema=schema_str
            )
            
            logger.info(f"[{self.name}] Invoking LLM for extraction")
            
            # December 2025 Standard: Use with_structured_output() when available
            # This provides automatic parsing, validation, and retry logic
            warnings = []
            data = None
            
            # Try with_structured_output() first (December 2025 best practice)
            try:
                if hasattr(self.llm, 'with_structured_output'):
                    structured_llm = self.llm.with_structured_output(self.output_schema)
                    validated_model = await structured_llm.ainvoke(
                        [HumanMessage(content=prompt_text)]
                    )
                    # Convert Pydantic model to dict
                    data = validated_model.model_dump() if hasattr(validated_model, 'model_dump') else dict(validated_model)
                    logger.info(f"[{self.name}] Structured output extraction successful (with_structured_output)")
                else:
                    # Fallback: Manual parsing with repair strategies
                    raise AttributeError("LLM doesn't support with_structured_output")
            except (AttributeError, Exception) as e:
                # Fallback to manual parsing with repair strategies
                logger.info(f"[{self.name}] Using manual parsing (with_structured_output not available: {e})")
                
                # Invoke LLM manually
                response = await self.llm.ainvoke(
                    [HumanMessage(content=prompt_text)]
                )
                json_text = response.content
                
                # Strategy 1: Direct parse
                try:
                    data = json.loads(json_text)
                    logger.info(f"[{self.name}] Direct JSON parse successful")
                except json.JSONDecodeError as e:
                    logger.warning(f"[{self.name}] Direct parse failed: {e}")
                    
                    # SOLID Refactoring: Use strategy pattern
                    data = await self._repair_with_strategies(json_text, warnings)
                    
                    if data is None:
                        raise NodeExecutionError(
                            node_name=self.name,
                            reason="All JSON repair strategies failed",
                            state=state
                        )
            
            # Store results
            state["extracted"] = data
            if warnings:
                state["extraction_warnings"] = warnings
                self.metrics.warnings = warnings
            
            # Update metrics
            self.metrics.output_keys = ["extracted"]
            self.metrics.status = NodeStatus.SUCCESS
            
            logger.info(
                f"[{self.name}] Extraction complete "
                f"({len(data)} fields)" + (
                    f" with {len(warnings)} repairs" if warnings else ""
                )
            )
            
            return state
        
        except Exception as e:
            self.metrics.status = NodeStatus.FAILED
            self.metrics.error_message = str(e)
            raise await self.on_error(e, state)
        
        finally:
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.duration_ms = elapsed
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        return "analysis" in state
    
    async def _repair_with_strategies(
        self, 
        json_text: str, 
        warnings: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        SOLID Refactoring: Use strategy pattern for JSON repair.
        
        Single Responsibility: Orchestrate repair strategies.
        """
        for strategy in self._repair_strategies:
            try:
                data = await strategy.repair(json_text, self.output_schema)
                if data:
                    warnings.append(f"JSON required {strategy.name}")
                    logger.info(f"[{self.name}] {strategy.name} successful")
                    return data
            except Exception as e:
                logger.debug(f"[{self.name}] {strategy.name} failed: {e}")
        return None
    
    async def on_error(self, error: Exception, state: Dict[str, Any]) -> Exception:
        """
        Custom error handling for extraction failures.
        
        Log additional context and re-raise.
        """
        logger.error(
            f"[{self.name}] Extraction failed\n"
            f"Analysis length: {len(state.get('analysis', ''))}\n"
            f"Error: {error}"
        )
        return error

