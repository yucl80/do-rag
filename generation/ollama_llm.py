import requests
from typing import List, Optional
from .base import BaseLLM

class OllamaLLM(BaseLLM):
    """Ollama LLM implementation."""
    
    def __init__(self, 
                 model_name: str = "llama2",
                 base_url: str = "http://localhost:11434",
                 **kwargs):
        """Initialize Ollama LLM.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            **kwargs: Additional configuration parameters
        """
        self._model_name = model_name
        self._base_url = base_url.rstrip('/')
        
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                **kwargs) -> str:
        """Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        url = f"{self._base_url}/api/generate"
        
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()["response"]
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Ollama.
        
        Args:
            texts: List of texts to get embeddings for
            
        Returns:
            List of embedding vectors
        """
        url = f"{self._base_url}/api/embeddings"
        
        embeddings = []
        for text in texts:
            payload = {
                "model": self._model_name,
                "prompt": text
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            embeddings.append(response.json()["embedding"])
            
        return embeddings
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def provider_name(self) -> str:
        return "ollama" 