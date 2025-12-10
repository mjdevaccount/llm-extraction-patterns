"""
Two-Model Extraction Node

Implements the two-model pattern:
1. Main LLM generates unstructured analysis
2. Extractor LLM converts to structured JSON

Single Responsibility: Orchestrate two-model extraction workflow.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime

from pydantic import BaseModel, ValidationError
from langchain_core.messages import HumanMessage

from core.types import BaseNode, NodeExecutionError, NodeStatus
from core.llm_helpers import invoke_llm, is_llm_client
from core.extractor_formatters import IExtractorFormatter
from core.json_repair import repair_json

logger = logging.getLogger(__name__)


class TwoModelExtractionNode(BaseNode):
    """
    Two-model extraction node.
    
    Single Responsibility: Orchestrate extraction using main LLM + extractor LLM.
    
    Pattern:
        1. Main LLM (smart, large) generates unstructured analysis
        2. Extractor LLM (small, specialized) formats to structured JSON
    
    Input Requirements:
        - state["analysis"]: Unstructured text (optional, can generate from main_llm)
        - OR state must have keys for main_llm prompt
    
    Output:
        - Adds `extracted` key with parsed JSON as dict
        - Adds `extraction_warnings` if issues occurred
    
    Design Pattern: Strategy
        Different extractor formatters can be swapped (NuExtract, etc.)
    
    Example:
        # Main LLM for reasoning
        main_llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        
        # Extractor LLM for formatting
        extractor_llm = AutoModelForCausalLM.from_pretrained("numind/NuExtract-large")
        
        # Formatter for mapping
        formatter = NuExtractFormatter()
        
        # Can use Pydantic model for strict validation
        node = TwoModelExtractionNode(
            main_llm=main_llm,
            extractor_llm=extractor_llm,
            extractor_formatter=formatter,
            output_schema=ValuationResult,  # Pydantic model
            main_prompt_template="Analyze: {input}",
            strict=True,  # Hard fail on errors (for tests)
            name="two_model_extraction"
        )
        
        # Or use dict template for NuExtract-style schemas
        node2 = TwoModelExtractionNode(
            main_llm=main_llm,
            extractor_llm=extractor_llm,
            extractor_formatter=formatter,
            output_schema=MY_SCHEMA_JSON,  # Dict template
            strict=False,  # Soft fail (for production)
            bypass_main=True,  # Skip main LLM, use state["input"] directly
        )
        
        state = await node.execute({"input": "Value this option..."})
        # state["extracted"] = {"valuation": 123.45, ...}
    """
    
    def __init__(
        self,
        main_llm: Any,  # LLM for reasoning (high temperature)
        extractor_llm: Any,  # LLM for extraction (low temperature)
        extractor_formatter: IExtractorFormatter,
        output_schema: Union[Type[BaseModel], Dict[str, Any]],  # Pydantic model or dict template
        main_prompt_template: Optional[str] = None,
        required_state_keys: Optional[List[str]] = None,
        strict: bool = True,  # Hard fail on extraction errors (True) or soft fail (False)
        bypass_main: bool = False,  # Skip main LLM, use state["input"] or state["analysis"] directly
        name: str = "two_model_extraction",
        description: str = "Two-model extraction: reasoning + formatting",
    ):
        """
        Initialize two-model extraction node.
        
        Args:
            main_llm: LLM for generating unstructured analysis
                Should have temperature=0.7-0.8 for reasoning
            extractor_llm: LLM for extracting structured JSON
                Should have temperature=0.0 for deterministic extraction
            extractor_formatter: Formatter that maps main output to extractor input
            output_schema: Pydantic BaseModel (for strict validation) or dict template
                (for NuExtract-style string templates). Formatter handles conversion.
            main_prompt_template: Template for main LLM prompt (if generating analysis)
                If None, expects state["analysis"] to already exist
            required_state_keys: Keys required for main prompt (if generating)
            strict: If True, raise on extraction errors. If False, soft-fail and store warnings.
            bypass_main: If True, skip main LLM and use state["input"] or state["analysis"] directly
            name: Node identifier
            description: Human-readable description
        """
        super().__init__(name=name, description=description)
        self.main_llm = main_llm
        self.extractor_llm = extractor_llm
        self.extractor_formatter = extractor_formatter
        self.output_schema = output_schema
        self.main_prompt_template = main_prompt_template
        self.required_state_keys = required_state_keys or []
        self.strict = strict
        self.bypass_main = bypass_main
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute two-model extraction.
        
        Process:
            1. Generate or retrieve unstructured analysis from main LLM
            2. Format prompt for extractor LLM using formatter
            3. Invoke extractor LLM
            4. Extract and parse JSON
            5. Validate against schema
        
        Args:
            state: Must have "analysis" OR keys for main_prompt_template
        
        Returns:
            State with added keys:
                - extracted: Parsed JSON dict
                - extraction_warnings: List of warnings (if any)
        
        Raises:
            NodeExecutionError: If extraction fails
        """
        start_time = datetime.now()
        self.metrics.status = NodeStatus.RUNNING
        
        try:
            # Step 1: Get unstructured analysis
            analysis = await self._get_analysis(state)
            self.metrics.input_keys = list(state.keys())
            
            logger.info(f"[{self.name}] Step 1: Got analysis ({len(analysis)} chars)")
            
            # Step 2: Format extractor prompt
            extractor_prompt = self.extractor_formatter.format_prompt(
                unstructured_text=analysis,
                schema=self.output_schema,
            )
            
            logger.info(f"[{self.name}] Step 2: Formatted extractor prompt ({len(extractor_prompt)} chars)")
            
            # Step 3: Invoke extractor LLM
            logger.info(f"[{self.name}] Step 3: Invoking extractor LLM ({self.extractor_formatter.name})")
            
            extractor_response = await self._invoke_extractor(
                extractor_prompt
            )
            
            logger.info(f"[{self.name}] Step 4: Got extractor response ({len(extractor_response)} chars)")
            
            # Step 4: Extract JSON from response
            json_str = self.extractor_formatter.extract_json_from_response(
                extractor_response
            )
            
            # Step 5: Parse JSON and validate
            try:
                json_data = json.loads(json_str)
            except json.JSONDecodeError as parse_error:
                # Try JSON repair
                json_data = repair_json(json_str)
                
                if json_data is None:
                    if self.strict:
                        logger.error(f"[{self.name}] JSON parse failed: {parse_error}")
                        logger.error(f"[{self.name}] JSON string (first 500 chars): {json_str[:500]}")
                        raise
                    # Soft-fail path
                    state.setdefault("extraction_warnings", []).append(f"JSON parse failed: {parse_error}")
                    state["extracted_raw"] = json_str
                    state["extraction_error"] = str(parse_error)
                    state["extracted"] = {}
                    logger.warning(f"[{self.name}] Extraction failed (soft-fail mode): {parse_error}")
                    return state
                else:
                    logger.info(f"[{self.name}] JSON repair successful")
            
            # Validate against schema (if Pydantic model)
            try:
                if isinstance(self.output_schema, type) and issubclass(self.output_schema, BaseModel):
                    validated = self.output_schema.model_validate(json_data)
                    data = validated.model_dump() if hasattr(validated, 'model_dump') else dict(validated)
                else:
                    # Dict schema - just use the JSON data as-is
                    data = json_data
                
                # Store results
                state["extracted"] = data
                
            except ValidationError as validation_error:
                if self.strict:
                    raise
                # Soft-fail path: validation failed but JSON parsed
                state.setdefault("extraction_warnings", []).append(f"Schema validation failed: {validation_error}")
                state["extracted"] = json_data  # Use unvalidated data
                logger.warning(f"[{self.name}] Schema validation failed (soft-fail mode): {validation_error}")
            
            # Update metrics
            self.metrics.output_keys = ["extracted"]
            self.metrics.status = NodeStatus.SUCCESS
            
            # Capture extractor-specific stats
            if not hasattr(self.metrics, 'extra'):
                self.metrics.extra = {}
            self.metrics.extra.update({
                "analysis_chars": len(analysis),
                "extractor_prompt_chars": len(extractor_prompt),
                "extractor_response_chars": len(extractor_response),
                "formatter": self.extractor_formatter.name,
            })
            
            # Store raw extractor text for debugging if enabled
            import os
            if os.getenv("DEBUG_EXTRACTOR", "0") == "1":
                state["_debug_extractor_raw"] = extractor_response
                state["_debug_extractor_prompt"] = extractor_prompt
            
            logger.info(f"[{self.name}] Extraction complete")
            
            return state
        
        except Exception as e:
            self.metrics.status = NodeStatus.FAILED
            self.metrics.error_message = str(e)
            raise await self.on_error(e, state)
        
        finally:
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.duration_ms = elapsed
    
    async def _get_analysis(self, state: Dict[str, Any]) -> str:
        """
        Get unstructured analysis from state or generate from main LLM.
        
        Args:
            state: Current workflow state
        
        Returns:
            Unstructured text analysis
        """
        # Bypass main LLM if flag is set
        if self.bypass_main:
            if "analysis" in state:
                return state["analysis"]
            elif "input" in state:
                return str(state["input"])
            else:
                raise NodeExecutionError(
                    node_name=self.name,
                    reason="bypass_main=True but no 'analysis' or 'input' in state",
                    state=state
                )
        
        # If analysis already exists, use it
        if "analysis" in state:
            return state["analysis"]
        
        # Otherwise, generate from main LLM
        if self.main_prompt_template is None:
            raise NodeExecutionError(
                node_name=self.name,
                reason="No 'analysis' in state and no main_prompt_template provided",
                state=state
            )
        
        # Validate required keys
        if not self.validate_input(state):
            missing = [k for k in self.required_state_keys if k not in state]
            raise NodeExecutionError(
                node_name=self.name,
                reason=f"Missing required keys for main LLM: {missing}",
                state=state
            )
        
        # Format prompt
        format_dict = {k: state.get(k) for k in self.required_state_keys}
        prompt_text = self.main_prompt_template.format(**format_dict)
        
        # Invoke main LLM
        logger.info(f"[{self.name}] Generating analysis from main LLM")
        messages = [HumanMessage(content=prompt_text)]
        analysis = await invoke_llm(
            self.main_llm,
            messages,
            system="You are a helpful assistant that analyzes information."
        )
        
        # Store in state for potential reuse
        state["analysis"] = analysis
        
        return analysis
    
    async def _invoke_extractor(
        self,
        prompt: str
    ) -> str:
        """
        Invoke extractor LLM with formatted prompt.
        
        Args:
            prompt: Formatted prompt for extractor
        
        Returns:
            Extractor response text
        """
        messages = [HumanMessage(content=prompt)]
        
        response = await invoke_llm(
            self.extractor_llm,
            messages,
            system="You are a specialized assistant that extracts structured data."
        )
        
        return response
    
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        Validate that state has required keys.
        
        If analysis exists, no validation needed.
        Otherwise, check required_state_keys.
        """
        if "analysis" in state:
            return True
        
        if self.main_prompt_template is None:
            return False
        
        return all(k in state for k in self.required_state_keys)
    
    async def on_error(self, error: Exception, state: Dict[str, Any]) -> Exception:
        """
        Custom error handling for extraction failures.
        
        Log additional context and re-raise.
        """
        logger.error(
            f"[{self.name}] Two-model extraction failed\n"
            f"Formatter: {self.extractor_formatter.name}\n"
            f"Analysis length: {len(state.get('analysis', ''))}\n"
            f"Error: {error}"
        )
        return error

