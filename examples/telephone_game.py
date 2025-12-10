"""
Game of Telephone - 5 Hops Test

A message passes through 5 nodes, each node processes and passes it along.
Tests the workflow orchestrator with local Ollama (qwen2.5:8b or qwen3:8b).

Usage:
    python examples/telephone_game.py
"""

import asyncio
import os
from typing import Dict, Any

from patterns.core.llm_client import OllamaClient
from patterns.patterns.iev.nodes import IntelligenceNode


async def run_telephone_game(initial_message: str = "The quick brown fox jumps over the lazy dog"):
    """Run the telephone game with a message passing through 5 hops."""
    
    print("=" * 60)
    print("GAME OF TELEPHONE - 5 HOPS")
    print("=" * 60)
    print(f"\nInitial message: {initial_message}\n")
    
    # Initialize Ollama client with local model
    # Set OLLAMA_MODEL env var to your model name (e.g., "qwen3:8b", "qwen2.5:8b", etc.)
    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:8b")
    llm = OllamaClient(
        model=model_name,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    
    print(f"Using model: {model_name}")
    print(f"Ollama URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}\n")
    
    # Create 5 intelligence nodes (each is a "hop")
    # Each node receives the previous message and passes it along
    
    # Hop 1: receives initial_message
    hop1 = IntelligenceNode(
        llm=llm,
        prompt_template=(
            "You are playing a game of telephone. "
            "The starting message is: {initial_message}\n\n"
            "Repeat the message exactly, but add ONE brief word or phrase (2-3 words max) at the end. "
            "Keep the core message completely intact. "
            "Example: If you hear 'Hello world', you might say 'Hello world - interesting!'"
        ),
        required_state_keys=["initial_message"],
        name="hop_1",
        description="Telephone hop 1 of 5"
    )
    
    # Hops 2-5: receive previous_message from previous hop
    hops = [hop1]
    for i in range(2, 6):
        hop = IntelligenceNode(
            llm=llm,
            prompt_template=(
                "You are playing a game of telephone. "
                "The previous person said: {previous_message}\n\n"
                "Repeat what they said exactly, but add ONE brief word or phrase (2-3 words max) at the end. "
                "Keep the core message completely intact. "
                "Example: If you hear 'Hello world - interesting!', you might say 'Hello world - interesting! - cool!'"
            ),
            required_state_keys=["previous_message"],
            name=f"hop_{i}",
            description=f"Telephone hop {i} of 5"
        )
        hops.append(hop)
    
    # Execute hops sequentially, passing message along
    print("-" * 60)
    print("MESSAGE PROGRESSION:")
    print("-" * 60)
    
    messages = [initial_message]
    state = {"initial_message": initial_message}
    
    try:
        for i, hop in enumerate(hops, 1):
            print(f"\nHop {i} processing...")
            
            # Execute hop
            state = await hop.execute(state)
            
            # IntelligenceNode outputs "analysis"
            if "analysis" in state:
                message = state["analysis"]
                messages.append(message)
                print(f"Hop {i} output: {message}")
                
                # Prepare for next hop: analysis becomes previous_message
                state["previous_message"] = message
            else:
                print(f"⚠️  Hop {i} did not produce 'analysis' in state")
                print(f"   State keys: {list(state.keys())}")
                break
        
        # Show final result
        print("\n" + "-" * 60)
        print("FINAL RESULT:")
        print("-" * 60)
        print(f"\nInitial: {messages[0]}")
        print(f"Final:   {messages[-1]}")
        
        # Show all messages
        print("\n" + "-" * 60)
        print("ALL MESSAGES:")
        print("-" * 60)
        for i, msg in enumerate(messages):
            print(f"Hop {i}: {msg}")
        
        print("\n" + "=" * 60)
        print("SUCCESS: Message passed through all 5 hops!")
        print("=" * 60)
        
        return state
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None




if __name__ == "__main__":
    # Test with default message
    result = asyncio.run(run_telephone_game())
    
    # Or test with custom message
    # result = asyncio.run(run_telephone_game("Python is a great programming language"))

