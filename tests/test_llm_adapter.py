"""Tests for LLM adapter (unified LLMClient and LangChain interface)."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from patterns.patterns.iev.nodes.llm_adapter import invoke_llm, is_llm_client
from patterns.core.llm_client import LLMClient


# ============================================================================
# Mock LLMClient
# ============================================================================

class MockLLMClient(LLMClient):
    """Mock LLMClient for testing."""
    
    def __init__(self, response: str = "Mock response"):
        self.response = response
        self.call_count = 0
    
    def generate(self, system: str, user: str, temperature: float = 0.7, max_tokens=None, **kwargs) -> str:
        """Synchronous generate."""
        self.call_count += 1
        return self.response
    
    def stream(self, system: str, user: str, temperature: float = 0.7):
        """Stream generator."""
        yield self.response


# ============================================================================
# Mock LangChain ChatModel
# ============================================================================

class MockLangChainLLM:
    """Mock LangChain ChatModel for testing."""
    
    def __init__(self, response: str = "LangChain response"):
        self.response = response
        self.call_count = 0
    
    async def ainvoke(self, messages):
        """Async invoke."""
        self.call_count += 1
        class MockResponse:
            def __init__(self, content):
                self.content = content
        return MockResponse(self.response)


# ============================================================================
# Tests
# ============================================================================

class TestLLMAdapter:
    """Tests for unified LLM adapter."""
    
    def test_is_llm_client_detection(self):
        """Test LLMClient detection."""
        llm_client = MockLLMClient()
        langchain_llm = MockLangChainLLM()
        
        assert is_llm_client(llm_client) is True
        assert is_llm_client(langchain_llm) is False
        assert is_llm_client(None) is False
    
    @pytest.mark.asyncio
    async def test_invoke_llm_with_llm_client(self):
        """Test invoke_llm with LLMClient."""
        llm = MockLLMClient(response="LLMClient response")
        messages = [HumanMessage(content="Test prompt")]
        
        result = await invoke_llm(llm, messages, system="System prompt")
        
        assert result == "LLMClient response"
        assert llm.call_count == 1
    
    @pytest.mark.asyncio
    async def test_invoke_llm_with_langchain(self):
        """Test invoke_llm with LangChain ChatModel."""
        llm = MockLangChainLLM(response="LangChain response")
        messages = [HumanMessage(content="Test prompt")]
        
        result = await invoke_llm(llm, messages)
        
        assert result == "LangChain response"
        assert llm.call_count == 1
    
    @pytest.mark.asyncio
    async def test_invoke_llm_with_multiple_messages(self):
        """Test invoke_llm handles multiple messages."""
        llm = MockLLMClient(response="Combined response")
        messages = [
            HumanMessage(content="First message"),
            AIMessage(content="Assistant response"),
            HumanMessage(content="Second message"),
        ]
        
        result = await invoke_llm(llm, messages, system="System")
        
        assert result == "Combined response"
        # Should combine user messages
        assert llm.call_count == 1
    
    @pytest.mark.asyncio
    async def test_invoke_llm_system_prompt(self):
        """Test invoke_llm uses system prompt for LLMClient."""
        llm = MockLLMClient(response="Response")
        messages = [HumanMessage(content="User message")]
        
        result = await invoke_llm(llm, messages, system="Custom system prompt")
        
        assert result == "Response"
        # System prompt should be passed to LLMClient.generate()

