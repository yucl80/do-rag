from openai import OpenAI
from typing import Dict, Any
from .base import BaseLLM

class OpenAILLM(BaseLLM):
    """OpenAI implementation of the BaseLLM interface."""
    
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        """
        Initialize the OpenAI LLM.
        
        Args:
            api_key: OpenAI API key (optional, will use environment variable if not provided)
            model: Model name to use
        """
        self.client = OpenAI(api_key=api_key)  # OpenAI client will automatically use OPENAI_API_KEY from env
        self.model = model
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI's API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
    
    def refine(self, text: str, context: str, **kwargs) -> str:
        """Refine the generated text based on context."""
        prompt = f"""Refine the following answer for factual correctness based on context:

Answer: {text}
Context: {context}

Please ensure the refined answer:
1. Is factually consistent with the context
2. Maintains all key information
3. Is clear and well-structured"""
        
        return self.generate(prompt, **kwargs)
    
    def condense(self, text: str, **kwargs) -> str:
        """Condense the text while maintaining key information."""
        prompt = f"""Condense and polish this answer in professional technical style while maintaining all key information:

{text}"""
        
        return self.generate(prompt, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the OpenAI model."""
        return {
            "provider": "OpenAI",
            "model": self.model,
            "type": "chat"
        } 