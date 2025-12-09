"""
Abstractions and Interfaces for SOLID Design

This module defines abstract interfaces and base classes that enforce
SOLID principles throughout the workflow system.

Key Abstractions:
  - ILLMProvider: Abstract interface for LLM interactions (DIP)
  - IJSONRepairStrategy: Strategy interface for JSON repair (OCP, SRP)
  - IValidationStrategy: Strategy interface for validation modes (OCP, SRP)
  - IStateValidator: Interface for input validation (ISP)
  - IMetricsCollector: Interface for metrics collection (ISP)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, ValidationError


# ============================================================================
# LLM Abstraction (Dependency Inversion Principle)
# ============================================================================

class ILLMProvider(ABC):
    """
    Abstract interface for LLM interactions.
    
    This abstraction allows nodes to depend on an interface rather than
    concrete LLM implementations (Dependency Inversion Principle).
    
    Implementations can wrap LangChain, OpenAI, Anthropic, etc.
    """
    
    @abstractmethod
    async def invoke(self, messages: List[Any]) -> str:
        """
        Invoke LLM with messages and return text response.
        
        Args:
            messages: List of message objects (LangChain format)
        
        Returns:
            Text response from LLM
        """
        pass
    
    @abstractmethod
    async def invoke_structured(
        self, 
        messages: List[Any], 
        schema: Type[BaseModel]
    ) -> BaseModel:
        """
        Invoke LLM with structured output.
        
        Args:
            messages: List of message objects
            schema: Pydantic model for structured output
        
        Returns:
            Pydantic model instance
        """
        pass
    
    @property
    @abstractmethod
    def supports_structured_output(self) -> bool:
        """Whether this LLM provider supports structured output."""
        pass


# ============================================================================
# JSON Repair Strategy (Open/Closed Principle, Single Responsibility)
# ============================================================================

class IJSONRepairStrategy(ABC):
    """
    Strategy interface for JSON repair.
    
    Each repair strategy has a single responsibility: repair JSON in one way.
    New strategies can be added without modifying ExtractionNode (OCP).
    """
    
    @abstractmethod
    async def repair(
        self, 
        json_text: str, 
        schema: Type[BaseModel]
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to repair JSON text.
        
        Args:
            json_text: Broken or malformed JSON text
            schema: Expected Pydantic schema
        
        Returns:
            Repaired data dict, or None if repair failed
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""
        pass


# ============================================================================
# Validation Strategy (Open/Closed Principle, Single Responsibility)
# ============================================================================

class IValidationStrategy(ABC):
    """
    Strategy interface for validation modes.
    
    Each validation strategy implements one validation approach.
    New modes can be added without modifying ValidationNode (OCP).
    """
    
    @abstractmethod
    async def validate(
        self,
        data: Dict[str, Any],
        schema: Type[BaseModel],
        validation_rules: Dict[str, Any],
        llm: Optional[ILLMProvider] = None,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Validate and optionally repair data.
        
        Args:
            data: Data to validate
            schema: Pydantic schema
            validation_rules: Custom validation rules
            llm: Optional LLM for repair (for RETRY mode)
            max_retries: Max retry attempts
        
        Returns:
            Dict with keys:
                - validated: Validated data dict
                - warnings: List of warnings
                - repairs: Dict of field repairs
                - outcome: Validation outcome string
        
        Raises:
            ValidationError: If validation fails and cannot be repaired
        """
        pass
    
    @property
    @abstractmethod
    def mode_name(self) -> str:
        """Validation mode identifier."""
        pass


# ============================================================================
# State Validation (Interface Segregation Principle)
# ============================================================================

class IStateValidator(ABC):
    """
    Interface for input state validation.
    
    Separates validation concern from execution (ISP).
    """
    
    @abstractmethod
    def validate(self, state: Dict[str, Any]) -> bool:
        """
        Validate that state has required keys.
        
        Args:
            state: Current workflow state
        
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_required_keys(self) -> List[str]:
        """Get list of required state keys."""
        pass


# ============================================================================
# Metrics Collection (Interface Segregation Principle)
# ============================================================================

class IMetricsCollector(ABC):
    """
    Interface for metrics collection.
    
    Separates metrics concern from execution (ISP).
    """
    
    @abstractmethod
    def record_execution(
        self,
        node_name: str,
        duration_ms: float,
        input_keys: List[str],
        output_keys: List[str],
        status: str,
        warnings: Optional[List[str]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record node execution metrics."""
        pass
    
    @abstractmethod
    def get_metrics(self, node_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for node(s).
        
        Args:
            node_name: Specific node, or None for all nodes
        
        Returns:
            Metrics dict
        """
        pass


# ============================================================================
# Node Responsibility Interfaces (Interface Segregation Principle)
# ============================================================================

class IExecutable(ABC):
    """
    Interface for node execution.
    
    Separates execution concern from other responsibilities (ISP).
    Nodes that only need to execute can implement just this.
    """
    
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute node's work.
        
        Args:
            state: Current workflow state
        
        Returns:
            Updated state dict
        """
        pass


class IValidatable(ABC):
    """
    Interface for input validation.
    
    Separates validation concern from execution (ISP).
    Nodes can implement validation independently.
    """
    
    @abstractmethod
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        Validate that state has required keys.
        
        Args:
            state: Current workflow state
        
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_required_keys(self) -> List[str]:
        """Get list of required state keys."""
        pass


class IErrorHandler(ABC):
    """
    Interface for error handling.
    
    Separates error handling concern from execution (ISP).
    Nodes can implement custom error handling independently.
    """
    
    @abstractmethod
    async def on_error(
        self,
        error: Exception,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle errors during execution.
        
        Args:
            error: The exception that was raised
            state: State at time of error
        
        Returns:
            Updated state, or raise to propagate error
        """
        pass


class IMetricsProvider(ABC):
    """
    Interface for metrics provision.
    
    Separates metrics concern from execution (ISP).
    Nodes can provide metrics independently.
    """
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get execution metrics for this node.
        
        Returns:
            Dict with execution stats
        """
        pass

