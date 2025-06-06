from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on the prompt."""
        pass
    
    @abstractmethod
    def refine(self, text: str, context: str, **kwargs) -> str:
        """Refine the generated text based on context."""
        pass
    
    @abstractmethod
    def condense(self, text: str, **kwargs) -> str:
        """Condense the text while maintaining key information."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the LLM model."""
        pass 