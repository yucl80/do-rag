from typing import Dict, Type
from .base import BaseLLM
from .openai_llm import OpenAILLM
from .ollama_llm import OllamaLLM
from config import LLM_CONFIGS, DEFAULT_LLM_PROVIDER

class LLMFactory:
    """Factory class for creating LLM instances."""
    
    _providers: Dict[str, Type[BaseLLM]] = {
        "openai": OpenAILLM,
        "ollama": OllamaLLM
    }
    
    @classmethod
    def create(cls, provider: str = None) -> BaseLLM:
        """Create an LLM instance for the specified provider.
        
        Args:
            provider: Name of the LLM provider. If None, uses the default provider.
            
        Returns:
            An instance of the specified LLM provider
            
        Raises:
            ValueError: If the provider is not supported
        """
        provider = provider or DEFAULT_LLM_PROVIDER
        
        if provider not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
        provider_class = cls._providers[provider]
        config = LLM_CONFIGS[provider]
        
        return provider_class(**config)
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLM]) -> None:
        """Register a new LLM provider.
        
        Args:
            name: Name of the provider
            provider_class: Provider class that implements BaseLLM
        """
        cls._providers[name] = provider_class 