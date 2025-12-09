"""Conversation memory and context management."""

from typing import List
from pydantic import BaseModel
from datetime import datetime


class Message(BaseModel):
    """Single message in conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ConversationMemory:
    """Short-term context manager.

    Tracks the conversation history for use in graphs.
    """

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.messages: List[Message] = []

    def add_message(self, role: str, content: str):
        """Add a message to memory."""
        self.messages.append(Message(role=role, content=content))

        # Prune old messages if exceeding max
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history :]

    def get_context(self) -> str:
        """Format memory as a context string for prompts."""
        if not self.messages:
            return ""

        context = []
        for msg in self.messages:
            role_label = msg.role.upper()
            context.append(f"{role_label}: {msg.content}")

        return "\n".join(context)

    def get_recent(self, n: int = 3) -> List[Message]:
        """Get the last n messages."""
        return self.messages[-n:]

    def clear(self):
        """Clear all history."""
        self.messages = []
