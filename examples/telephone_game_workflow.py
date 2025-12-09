"""
Game of Telephone - 5 Hops Test with Workflow Orchestrator

A message passes through 5 nodes using the Workflow orchestrator.
Tests the workflow system with local Ollama (qwen2.5:8b or qwen3:8b).

This version uses the Workflow class to properly orchestrate nodes,
with state transformation between hops to handle local model brittleness.

Usage:
    python examples/telephone_game_workflow.py
"""

import asyncio
import os
from typing import Dict, Any, TypedDict

from patterns.core.llm_client import OllamaClient
from patterns.patterns.iev.nodes import IntelligenceNode, BaseNode
from patterns.patterns.iev.workflows import Workflow
from patterns.patterns.iev.nodes.base_node import NodeStatus


class TelephoneState(TypedDict, total=False):
    """State schema for telephone game workflow."""
    initial_message: str
    previous_message: str
    analysis: str  # IntelligenceNode output
    hop_1_message: str
    hop_2_message: str
    hop_3_message: str
    hop_4_message: str
    hop_5_message: str
    final_message: str


class StateTransformerNode(BaseNode):
    """
    Transforms state between hops.
    
    For local models, we need explicit state transformation to ensure
    data flows correctly between nodes.
    """
    
    def __init__(self, source_key: str, target_key: str, name: str):
        """
        Initialize transformer node.
        
        Args:
            source_key: State key to read from (e.g., "analysis")
            target_key: State key to write to (e.g., "previous_message")
            name: Node identifier
        """
        super().__init__(name=name, description=f"Transform {source_key} -> {target_key}")
        self.source_key = source_key
        self.target_key = target_key
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform state by copying source_key to target_key."""
        if self.source_key in state:
            state[self.target_key] = state[self.source_key]
            # Also store in hop-specific key for tracking
            # Extract hop number from the previous intelligence node name
            # trans_1 comes after hop_1_intel, so store as hop_1_message
            if "trans_" in self.name:
                trans_num = self.name.split("_")[-1]
                # This transformer processes the output from hop_{trans_num}_intel
                state[f"hop_{trans_num}_message"] = state[self.source_key]
        
        self.metrics.status = NodeStatus.SUCCESS
        self.metrics.output_keys = [self.target_key]
        return state
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate that source_key exists in state."""
        return self.source_key in state


async def run_telephone_game_workflow(initial_message: str = "The quick brown fox jumps over the lazy dog"):
    """Run the telephone game using Workflow orchestrator."""
    
    print("=" * 60)
    print("GAME OF TELEPHONE - 5 HOPS (Workflow Orchestrator)")
    print("=" * 60)
    print(f"\nInitial message: {initial_message}\n")
    
    # Initialize Ollama client with local model
    # Set OLLAMA_MODEL env var to your model (e.g., "qwen3:8b", "qwen2.5:8b", "mistral:7b", etc.)
    model_name = os.getenv("OLLAMA_MODEL", "qwen3:8b")  # Default to qwen3:8b as user mentioned
    llm = OllamaClient(
        model=model_name,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    
    print(f"Using model: {model_name}")
    print(f"Ollama URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}")
    print(f"Note: Make sure '{model_name}' is available in Ollama (run: ollama list)\n")
    
    # Create nodes: 5 intelligence nodes + 4 transformers (between hops)
    nodes = []
    
    # Hop 1: Intelligence node
    hop1_intel = IntelligenceNode(
        llm=llm,
        prompt_template=(
            "You are playing a game of telephone. "
            "The starting message is: {initial_message}\n\n"
            "Repeat the message exactly, but add ONE brief word or phrase (2-3 words max) at the end. "
            "Keep the core message completely intact. "
            "Example: If you hear 'Hello world', you might say 'Hello world - interesting!'"
        ),
        required_state_keys=["initial_message"],
        name="hop_1_intel",
        description="Telephone hop 1 intelligence"
    )
    nodes.append(hop1_intel)
    
    # Transformer 1: analysis -> previous_message
    trans1 = StateTransformerNode("analysis", "previous_message", "trans_1")
    nodes.append(trans1)
    
    # Hops 2-5: Intelligence + Transformer pairs
    for i in range(2, 6):
        # Intelligence node
        hop_intel = IntelligenceNode(
            llm=llm,
            prompt_template=(
                "You are playing a game of telephone. "
                "The previous person said: {previous_message}\n\n"
                "Repeat what they said exactly, but add ONE brief word or phrase (2-3 words max) at the end. "
                "Keep the core message completely intact. "
                "Example: If you hear 'Hello world - interesting!', you might say 'Hello world - interesting! - cool!'"
            ),
            required_state_keys=["previous_message"],
            name=f"hop_{i}_intel",
            description=f"Telephone hop {i} intelligence"
        )
        nodes.append(hop_intel)
        
        # Transformer: analysis -> previous_message (for next hop)
        if i < 5:  # Don't need transformer after last hop
            trans = StateTransformerNode("analysis", "previous_message", f"trans_{i}")
            nodes.append(trans)
    
    # Define edges: intel -> trans -> next_intel
    edges = [
        ("hop_1_intel", "trans_1"),
        ("trans_1", "hop_2_intel"),
        ("hop_2_intel", "trans_2"),
        ("trans_2", "hop_3_intel"),
        ("hop_3_intel", "trans_3"),
        ("trans_3", "hop_4_intel"),
        ("hop_4_intel", "trans_4"),
        ("trans_4", "hop_5_intel"),
    ]
    
    # Create workflow
    workflow = Workflow(
        name="telephone_game",
        state_schema=TelephoneState,
        nodes=nodes,
        edges=edges,
    )
    
    print("Workflow structure:")
    print(workflow.visualize())
    print()
    
    # Execute workflow
    initial_state = {"initial_message": initial_message}
    
    try:
        print("-" * 60)
        print("EXECUTING WORKFLOW:")
        print("-" * 60)
        
        result = await workflow.invoke(initial_state)
        
        # Extract messages from each hop
        print("\n" + "-" * 60)
        print("MESSAGE PROGRESSION:")
        print("-" * 60)
        
        messages = [initial_message]
        
        # Track messages through hops
        # Check for hop-specific messages first, then fall back to analysis
        for i in range(1, 6):
            hop_key = f"hop_{i}_message"
            if hop_key in result:
                messages.append(result[hop_key])
                print(f"Hop {i}: {result[hop_key]}")
            elif i == 1 and "analysis" in result:
                # First hop's analysis (if transformer didn't store it)
                messages.append(result["analysis"])
                print(f"Hop {i}: {result['analysis']}")
        
        # If we didn't get all messages, check analysis field (last hop's output)
        if len(messages) < 6 and "analysis" in result:
            # The last hop's output is in analysis
            if result["analysis"] not in messages:
                messages.append(result["analysis"])
                print(f"Hop 5: {result['analysis']}")
        
        # Final message
        if "hop_5_message" in result:
            final_msg = result["hop_5_message"]
        elif "analysis" in result:
            final_msg = result["analysis"]
        else:
            final_msg = messages[-1] if len(messages) > 1 else initial_message
        
        print("\n" + "-" * 60)
        print("FINAL RESULT:")
        print("-" * 60)
        print(f"\nInitial: {messages[0]}")
        print(f"Final:   {final_msg}")
        
        # Show all messages
        print("\n" + "-" * 60)
        print("ALL MESSAGES:")
        print("-" * 60)
        for i, msg in enumerate(messages):
            print(f"Hop {i}: {msg}")
        
        # Show metrics
        print("\n" + "-" * 60)
        print("WORKFLOW METRICS:")
        print("-" * 60)
        metrics = workflow.get_metrics()
        print(f"Overall status: {metrics.get('overall_status', 'unknown')}")
        print(f"Total duration: {metrics.get('total_duration_ms', 0):.2f}ms")
        
        if "nodes" in metrics:
            print("\nNode execution times:")
            for node_name, node_metrics in metrics["nodes"].items():
                duration = node_metrics.get("duration_ms", 0)
                status = node_metrics.get("status", "unknown")
                print(f"  {node_name}: {duration:.2f}ms ({status})")
        
        print("\n" + "=" * 60)
        print("SUCCESS: Message passed through all 5 hops via Workflow!")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test with default message
    result = asyncio.run(run_telephone_game_workflow())
    
    # Or test with custom message
    # result = asyncio.run(run_telephone_game_workflow("Python is a great programming language"))

