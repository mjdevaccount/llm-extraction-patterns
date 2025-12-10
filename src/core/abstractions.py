"""
Core Abstractions for SOLID Design

All interfaces and abstractions used across patterns.
Moved from pattern-specific folders to core for reusability.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel


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
# Retry Strategy (Open/Closed Principle, Single Responsibility)
# ============================================================================

class IRetryStrategy(ABC):
    """
    Abstract interface for extraction retry strategies.
    
    Implement to add new retry approaches:
      - PromptRefinementRetry: Refine prompt based on error
      - LLMAssistedRepairRetry: Ask LLM to fix its output
      - SchemaGuidedRetry: Extract with schema constraints
    """
    
    @abstractmethod
    async def retry(
        self,
        llm: Any,  # LLMClient or ILLMProvider
        original_prompt: str,
        failed_output: str,
        error_message: str,
        schema: Type[BaseModel],
        attempt_number: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to extract valid JSON after initial failure.
        
        Args:
            llm: LLM client to use
            original_prompt: Original extraction prompt
            failed_output: LLM output that failed to parse
            error_message: Error from JSON parsing
            schema: Expected Pydantic schema
            attempt_number: Which retry attempt (1, 2, 3, ...)
        
        Returns:
            Valid dict matching schema, or None if retry failed
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""
        pass
    
    @property
    @abstractmethod
    def max_retries(self) -> int:
        """Maximum number of retries for this strategy."""
        pass

