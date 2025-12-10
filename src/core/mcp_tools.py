"""MCP (Model Context Protocol) tool integration."""

from typing import Any, Optional, Dict, List
from pydantic import BaseModel
import json
import httpx
import os


class ToolCall(BaseModel):
    """Representation of a tool call."""

    tool_name: str
    args: Dict[str, Any]
    result: Optional[str] = None
    error: Optional[str] = None


class MCPToolkit:
    """Interface to your aistack-mcp server.

    Your MCP server exposes tools like:
    - file_read(path: str)
    - file_write(path: str, content: str)
    - get_market_data(ticker: str, date: str)
    - execute_trade(action: str, qty: int, price: float)
    - etc.
    """

    def __init__(self, mcp_server_url: str = "http://localhost:8000"):
        self.base_url = mcp_server_url.rstrip("/")
        self._tools_cache: Optional[List[str]] = None

    def list_tools(self) -> List[str]:
        """Get available tools from MCP server."""
        if self._tools_cache is not None:
            return self._tools_cache

        try:
            response = httpx.get(f"{self.base_url}/tools", timeout=5.0)
            response.raise_for_status()
            self._tools_cache = response.json().get("tools", [])
            return self._tools_cache
        except Exception as e:
            print(f"Warning: Could not fetch tools from MCP: {e}")
            return []

    def call(self, tool_name: str, **kwargs) -> str:
        """Call a tool on the MCP server.

        Usage:
            result = toolkit.call("file_read", path="/path/to/file")
            data = toolkit.call("get_market_data", ticker="NVDA", date="2025-12-01")
        """
        try:
            response = httpx.post(
                f"{self.base_url}/invoke",
                json={"tool": tool_name, "args": kwargs},
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json().get("result", "")
            return str(result)
        except Exception as e:
            raise RuntimeError(f"MCP tool '{tool_name}' failed: {str(e)}")

    def call_with_fallback(self, tool_name: str, fallback_fn: callable, **kwargs) -> str:
        """Try MCP, fall back to local function if MCP is unavailable.

        Useful for development without MCP server running.
        """
        try:
            return self.call(tool_name, **kwargs)
        except Exception as e:
            print(f"MCP unavailable ({tool_name}), using fallback: {e}")
            return fallback_fn(**kwargs)


# ============================================================
# Singleton instance (shared across all patterns)
# ============================================================

_mcp_instance: Optional[MCPToolkit] = None


def get_mcp_toolkit() -> MCPToolkit:
    """Get or create the global MCP toolkit."""
    global _mcp_instance
    if _mcp_instance is None:
        mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
        _mcp_instance = MCPToolkit(mcp_server_url=mcp_url)
    return _mcp_instance
