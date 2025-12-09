"""Intelligence Node for SOLID Design.

Single Responsibility: Generate thoughtful analysis without structure constraints.
"""

import logging
from typing import Any, Dict, List
from datetime import datetime

from langchain_core.messages import HumanMessage

from .base_node import BaseNode, NodeExecutionError, NodeStatus
from ..llm_adapter import invoke_llm

logger = logging.getLogger(__name__)


class IntelligenceNode(BaseNode):
    """
    Free-form reasoning node.
    
    Single Responsibility: Generate thoughtful analysis without structure constraints.
    
    Temperature: 0.7-0.8 (creative reasoning, explore alternatives)
    
    Input Requirements:
        - State must have all keys in `required_state_keys`
    
    Output:
        - Adds `analysis` key to state with raw text response
        - Updates `messages` with conversation history
    
    Design Pattern: Strategy
        Different reasoning templates can be swapped without changing node logic.
    
    Example:
        intelligence = IntelligenceNode(
            llm=reasoning_llm,  # High-temperature LLM
            prompt_template="Analyze: {event}\nContext: {scenario}",
            required_state_keys=["event", "scenario"],
            name="reasoning"
        )
        
        state = await intelligence.execute({
            "event": {"name": "AI breakthrough"},
            "scenario": "market impact analysis"
        })
        # state["analysis"] = "AI adoption will follow S-curve..."
    """
    
    def __init__(
        self,
        llm: Any,  # LangChain ChatModel
        prompt_template: str,
        required_state_keys: List[str] = None,
        name: str = "intelligence",
        description: str = "Free-form reasoning phase",
    ):
        """
        Initialize intelligence node.
        
        Args:
            llm: LLMClient (cloud) or LangChain ChatModel (local/brittle). Should have high temperature.
            prompt_template: Template with {key} placeholders for state values
            required_state_keys: Keys that must exist in state
            name: Node identifier
            description: Human-readable description
        
        Note:
            LLM should be configured with temperature=0.7-0.8 for creative reasoning.
            Works with both LLMClient (simple cloud APIs) and LangChain ChatModel (local/brittle LLMs).
        """
        super().__init__(name=name, description=description)
        self.llm = llm
        self.prompt_template = prompt_template
        self.required_state_keys = required_state_keys or []
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute intelligence analysis.
        
        Process:
            1. Format prompt with state values
            2. Invoke LLM (high temperature for reasoning)
            3. Capture response as analysis
            4. Update message history
        
        Args:
            state: Must have all keys in required_state_keys
        
        Returns:
            State with added keys:
                - analysis: LLM response text
                - messages: Updated conversation history
        
        Raises:
            NodeExecutionError: If LLM fails or validation fails
        """
        start_time = datetime.now()
        self.metrics.status = NodeStatus.RUNNING
        self.metrics.input_keys = self.required_state_keys
        
        try:
            # Validate input
            if not self.validate_input(state):
                missing = [k for k in self.required_state_keys if k not in state]
                raise NodeExecutionError(
                    node_name=self.name,
                    reason=f"Missing required keys: {missing}",
                    state=state
                )
            
            # Format prompt
            format_dict = {k: state.get(k) for k in self.required_state_keys}
            prompt_text = self.prompt_template.format(**format_dict)
            
            logger.info(f"[{self.name}] Invoking LLM for reasoning")
            
            # Invoke LLM (works with both LLMClient and LangChain ChatModel)
            messages = state.get("messages", [])
            messages.append(HumanMessage(content=prompt_text))
            
            response_text = await invoke_llm(self.llm, messages, system="You are a helpful assistant that analyzes information.")
            
            # Store results
            state["analysis"] = response_text
            # Update messages if using LangChain-style LLM
            if hasattr(self.llm, 'ainvoke'):
                from langchain_core.messages import AIMessage
                response = AIMessage(content=response_text)
                state["messages"] = messages + [response]
            else:
                state["messages"] = messages
            
            # Update metrics
            self.metrics.output_keys = ["analysis", "messages"]
            self.metrics.status = NodeStatus.SUCCESS
            
            logger.info(
                f"[{self.name}] Analysis complete "
                f"({len(response.content)} chars)"
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
        """
        Validate that state has all required keys.
        
        Args:
            state: Current workflow state
        
        Returns:
            True if all required keys present
        """
        return all(k in state for k in self.required_state_keys)

