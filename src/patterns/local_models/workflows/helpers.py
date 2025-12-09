"""
Workflow Helpers: Pre-built Workflow Patterns

This module provides higher-level workflow patterns built on top of the
base workflow abstraction. These helpers encode common patterns like
tool-calling loops and conditional branching.

Patterns:
  - ToolCallingWorkflow: Orchestrate tool-calling loops (Examples 06, 10)
  - ConditionalWorkflow: Handle branching logic (Examples 11, 12)
  - SimpleQAWorkflow: Simple question-answer (Example 13)

All helpers preserve observability (metrics, visualization, error handling).
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from abc import ABC
from dataclasses import dataclass, field

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from .workflow import Workflow
from ..nodes.base_nodes import BaseNode, NodeExecutionError, NodeMetrics, NodeStatus


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class ToolCall:
    """Record of a single tool call."""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_result: str
    duration_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConditionalBranchExecution:
    """Record of a conditional branch execution."""
    decision: str
    branch_key: str
    branch_output: Dict[str, Any]
    duration_ms: float
    success: bool
    error: Optional[str] = None


# ============================================================================
# TOOL CALLING WORKFLOW
# ============================================================================

class ToolCallingWorkflow(Workflow):
    """
    Orchestrate tool-calling loops.
    
    Automatically handles:
    - Tool binding to LLM
    - Iteration loops with max iteration safety
    - Tool result collection and history
    - Graceful error handling
    - Comprehensive metrics
    
    Pattern:
        User Query
            ↓
        LLM Decision: Use tool?
            ├─ No  → Final Response
            └─ Yes → Call Tool
                    ↓
                Tool Result
                    ↓
                [Back to decision]
    
    Example:
        workflow = ToolCallingWorkflow(
            name="web-research",
            llm=llm,
            tools=[search_tool, summarize_tool],
            max_iterations=10,
        )
        
        result = await workflow.invoke(
            query="What are the latest AI breakthroughs?"
        )
        
        print(f"Final Response: {result['final_response']}")
        print(f"Tool Calls: {len(result['tool_calls'])}")
        print(result['metrics'])
    """
    
    def __init__(
        self,
        name: str,
        llm: BaseLanguageModel,
        tools: List[Tool],
        max_iterations: int = 10,
        timeout_seconds: Optional[float] = 300,
        **kwargs
    ):
        """
        Initialize ToolCallingWorkflow.
        
        Args:
            name: Workflow name
            llm: Language model for decision making
            tools: List of tools available to the LLM
            max_iterations: Maximum tool calls before stopping
            timeout_seconds: Total timeout for workflow execution
            **kwargs: Additional Workflow arguments
        """
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(tools)
        
        # Initialize parent Workflow
        super().__init__(
            name=name,
            state_schema=dict,  # Use dict for flexibility
            nodes=[],  # ToolCallingWorkflow doesn't use traditional nodes
            edges=[],
            **kwargs
        )
        
        # Metrics
        self.tool_calls: List[ToolCall] = []
        self.start_time: Optional[float] = None
    
    async def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Run tool-calling loop.
        
        Args:
            query: User's query
            **kwargs: Additional arguments
        
        Returns:
            {
                "query": str,
                "tool_calls": [ToolCall],
                "final_response": str,
                "iterations": int,
                "metrics": {
                    "total_duration_ms": float,
                    "tool_call_count": int,
                    "average_tool_duration_ms": float,
                    "success_rate": float,
                    "errors": List[str],
                }
            }
        """
        self.start_time = time.time()
        self.tool_calls = []
        errors: List[str] = []
        
        try:
            # Initialize message history
            messages = [
                HumanMessage(content=query)
            ]
            
            iteration = 0
            final_response = None
            
            # Tool-calling loop
            while iteration < self.max_iterations:
                iteration += 1
                
                # Check timeout
                elapsed = time.time() - self.start_time
                if self.timeout_seconds and elapsed > self.timeout_seconds:
                    errors.append(f"Timeout after {elapsed:.1f}s")
                    break
                
                # Get LLM response
                try:
                    response = await asyncio.to_thread(
                        self.llm_with_tools.invoke,
                        messages
                    )
                except Exception as e:
                    errors.append(f"LLM error (iteration {iteration}): {str(e)}")
                    break
                
                # Add response to history
                messages.append(response)
                
                # Check if LLM wants to use tools
                tool_calls = response.tool_calls
                
                if not tool_calls:
                    # No tools -> final response
                    final_response = response.content
                    break
                
                # Execute each tool call
                for tool_call in tool_calls:
                    tool_start = time.time()
                    tool_name = tool_call["name"]
                    tool_input = tool_call["args"]
                    
                    try:
                        # Find and execute tool
                        tool = next(
                            (t for t in self.tools if t.name == tool_name),
                            None
                        )
                        
                        if not tool:
                            raise ValueError(f"Tool not found: {tool_name}")
                        
                        # Execute
                        result = await asyncio.to_thread(
                            tool.func,
                            **tool_input
                        )
                        
                        tool_duration = (time.time() - tool_start) * 1000
                        
                        # Record call
                        self.tool_calls.append(ToolCall(
                            tool_name=tool_name,
                            tool_input=tool_input,
                            tool_result=str(result),
                            duration_ms=tool_duration,
                            success=True,
                        ))
                        
                        # Add result to messages
                        messages.append(ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"],
                        ))
                        
                    except Exception as e:
                        tool_duration = (time.time() - tool_start) * 1000
                        error_msg = f"Tool error ({tool_name}): {str(e)}"
                        errors.append(error_msg)
                        
                        # Record failed call
                        self.tool_calls.append(ToolCall(
                            tool_name=tool_name,
                            tool_input=tool_input,
                            tool_result="",
                            duration_ms=tool_duration,
                            success=False,
                            error=error_msg,
                        ))
            
            # Handle max iterations reached
            if iteration >= self.max_iterations and not final_response:
                final_response = "Reached maximum iterations. No final response generated."
                errors.append(f"Max iterations ({self.max_iterations}) reached")
            
            # Calculate metrics
            total_duration = (time.time() - self.start_time) * 1000
            tool_durations = [tc.duration_ms for tc in self.tool_calls if tc.success]
            avg_tool_duration = (
                sum(tool_durations) / len(tool_durations)
                if tool_durations else 0
            )
            success_count = sum(1 for tc in self.tool_calls if tc.success)
            success_rate = (
                success_count / len(self.tool_calls)
                if self.tool_calls else 100.0
            )
            
            return {
                "query": query,
                "tool_calls": self.tool_calls,
                "final_response": final_response or "No response generated",
                "iterations": iteration,
                "metrics": {
                    "workflow_name": self.name,
                    "total_duration_ms": total_duration,
                    "tool_call_count": len(self.tool_calls),
                    "successful_calls": success_count,
                    "failed_calls": len(self.tool_calls) - success_count,
                    "average_tool_duration_ms": avg_tool_duration,
                    "success_rate": success_rate,
                    "max_iterations": self.max_iterations,
                    "errors": errors,
                }
            }
        
        except Exception as e:
            total_duration = (time.time() - self.start_time) * 1000
            error_msg = f"Workflow error: {str(e)}"
            errors.append(error_msg)
            
            return {
                "query": query,
                "tool_calls": self.tool_calls,
                "final_response": None,
                "iterations": iteration,
                "metrics": {
                    "workflow_name": self.name,
                    "total_duration_ms": total_duration,
                    "tool_call_count": len(self.tool_calls),
                    "successful_calls": sum(1 for tc in self.tool_calls if tc.success),
                    "failed_calls": sum(1 for tc in self.tool_calls if not tc.success),
                    "errors": errors,
                    "status": "failed",
                }
            }


# ============================================================================
# CONDITIONAL WORKFLOW
# ============================================================================

class ConditionalWorkflow(Workflow):
    """
    Handle branching logic with conditional execution.
    
    Automatically handles:
    - Decision node selection
    - Branch execution (sequential or parallel)
    - Merge node for result combination
    - Per-branch metrics
    - Error handling per branch
    
    Pattern:
        Input
          ↓
        [Decision] → Branch Key
          ├─ "A" → NodeA1 → NodeA2 → Result A
          ├─ "B" → NodeB1 → NodeB2 → Result B
          └─ "C" → NodeC1 → NodeC2 → Result C
          ↓
        [Merge]
          ↓
        Final Output
    
    Example:
        decision_node = IntelligenceNode(
            llm=llm,
            prompt_template="Classify request: {request}",
        )
        
        branches = {
            "urgent": [UrgentProcessingNode(), NotificationNode()],
            "normal": [NormalProcessingNode()],
            "low": [DeferredProcessingNode()],
        }
        
        workflow = ConditionalWorkflow(
            name="request-router",
            state_schema=RequestState,
            decision_node=decision_node,
            branches=branches,
            merge_node=MergeResultsNode(),
        )
        
        result = await workflow.invoke({"request": "urgent query"})
    """
    
    def __init__(
        self,
        name: str,
        state_schema: type,
        decision_node: BaseNode,
        branches: Dict[str, List[BaseNode]],
        merge_node: Optional[BaseNode] = None,
        parallel: bool = False,
        **kwargs
    ):
        """
        Initialize ConditionalWorkflow.
        
        Args:
            name: Workflow name
            state_schema: State TypedDict schema
            decision_node: Node that outputs branch key
            branches: Dict mapping branch keys to node lists
            merge_node: Optional node to merge branch results
            parallel: Whether to execute branches in parallel
            **kwargs: Additional Workflow arguments
        """
        self.decision_node = decision_node
        self.branches = branches
        self.merge_node = merge_node
        self.parallel = parallel
        self.branch_executions: List[ConditionalBranchExecution] = []
        
        # Flatten all nodes for parent Workflow
        all_nodes = [decision_node] + [n for branch in branches.values() for n in branch]
        if merge_node:
            all_nodes.append(merge_node)
        
        # Initialize parent Workflow
        super().__init__(
            name=name,
            state_schema=state_schema,
            nodes=all_nodes,
            edges=[],  # Edges are dynamic
            **kwargs
        )
    
    async def invoke(self, state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run conditional workflow.
        
        Args:
            state: Initial state dict
            **kwargs: Additional arguments
        
        Returns:
            {
                "state": {...},
                "decision": str,
                "branch_executed": str,
                "branch_executions": [ConditionalBranchExecution],
                "metrics": {...}
            }
        """
        start_time = time.time()
        self.branch_executions = []
        errors: List[str] = []
        
        try:
            # 1. Run decision node
            try:
                decision_state = await self.decision_node.execute(state)
                decision = decision_state.get("decision") or "unknown"
            except Exception as e:
                errors.append(f"Decision node error: {str(e)}")
                return {
                    "state": state,
                    "decision": None,
                    "branch_executed": None,
                    "error": str(e),
                    "metrics": {"total_duration_ms": (time.time() - start_time) * 1000}
                }
            
            # 2. Get branch
            branch_key = str(decision)
            
            if branch_key not in self.branches:
                error_msg = f"Unknown branch: {branch_key}"
                errors.append(error_msg)
                return {
                    "state": state,
                    "decision": decision,
                    "branch_executed": None,
                    "error": error_msg,
                    "metrics": {"total_duration_ms": (time.time() - start_time) * 1000}
                }
            
            # 3. Execute branch nodes
            branch_nodes = self.branches[branch_key]
            branch_start = time.time()
            branch_state = decision_state.copy()
            
            for node in branch_nodes:
                try:
                    branch_state = await node.execute(branch_state)
                except Exception as e:
                    error_msg = f"Branch node error ({node.name}): {str(e)}"
                    errors.append(error_msg)
                    break
            
            branch_duration = (time.time() - branch_start) * 1000
            branch_success = len([e for e in errors if "node error" in e]) == 0
            
            self.branch_executions.append(ConditionalBranchExecution(
                decision=str(decision),
                branch_key=branch_key,
                branch_output=branch_state,
                duration_ms=branch_duration,
                success=branch_success,
            ))
            
            # 4. Run merge node if present
            final_state = branch_state
            if self.merge_node:
                try:
                    final_state = await self.merge_node.execute(branch_state)
                except Exception as e:
                    error_msg = f"Merge node error: {str(e)}"
                    errors.append(error_msg)
            
            # 5. Return result
            total_duration = (time.time() - start_time) * 1000
            
            return {
                "state": final_state,
                "decision": decision,
                "branch_executed": branch_key,
                "branch_executions": self.branch_executions,
                "metrics": {
                    "workflow_name": self.name,
                    "total_duration_ms": total_duration,
                    "branch_duration_ms": branch_duration,
                    "decision": decision,
                    "branch_executed": branch_key,
                    "success": len(errors) == 0,
                    "errors": errors,
                }
            }
        
        except Exception as e:
            total_duration = (time.time() - start_time) * 1000
            return {
                "state": state,
                "decision": None,
                "branch_executed": None,
                "error": str(e),
                "metrics": {
                    "workflow_name": self.name,
                    "total_duration_ms": total_duration,
                    "errors": [str(e)] + errors,
                }
            }


# ============================================================================
# SIMPLE QA WORKFLOW
# ============================================================================

class SimpleQAWorkflow(Workflow):
    """
    Simple question-answer workflow.
    
    For basic Q&A patterns where you just need one LLM call.
    (Example 13)
    
    Example:
        workflow = SimpleQAWorkflow(
            name="qa",
            llm=llm,
            system_prompt="You are a helpful assistant.",
        )
        
        result = await workflow.invoke("What is Python?")
        print(result["answer"])
    """
    
    def __init__(
        self,
        name: str,
        llm: BaseLanguageModel,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs
    ):
        """
        Initialize SimpleQAWorkflow.
        
        Args:
            name: Workflow name
            llm: Language model
            system_prompt: System message for LLM
            **kwargs: Additional arguments
        """
        self.llm = llm
        self.system_prompt = system_prompt
        
        super().__init__(
            name=name,
            state_schema=dict,
            nodes=[],
            edges=[],
            **kwargs
        )
    
    async def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Run Q&A workflow.
        
        Args:
            query: User query
            **kwargs: Additional arguments
        
        Returns:
            {
                "query": str,
                "answer": str,
                "metrics": {...}
            }
        """
        start_time = time.time()
        
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=query),
            ]
            
            response = await asyncio.to_thread(
                self.llm.invoke,
                messages
            )
            
            duration = (time.time() - start_time) * 1000
            
            return {
                "query": query,
                "answer": response.content,
                "metrics": {
                    "workflow_name": self.name,
                    "total_duration_ms": duration,
                    "success": True,
                }
            }
        
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            
            return {
                "query": query,
                "answer": None,
                "error": str(e),
                "metrics": {
                    "workflow_name": self.name,
                    "total_duration_ms": duration,
                    "success": False,
                }
            }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ToolCallingWorkflow",
    "ConditionalWorkflow",
    "SimpleQAWorkflow",
    "ToolCall",
    "ConditionalBranchExecution",
]
