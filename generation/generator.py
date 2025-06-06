from typing import Optional, List, Dict, Any, Tuple
from .base import BaseLLM
import logging
import time
import re
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    stop_sequences: Optional[List[str]] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    max_context_length: int = 4000
    min_response_length: int = 10
    max_response_length: int = 2000

class Generator:
    """A class that handles text generation using a language model.
    
    This class provides methods for generating text responses using the provided
    language model, with support for various generation parameters and configurations.
    """
    
    def __init__(self, llm: BaseLLM):
        """Initialize the Generator with a language model.
        
        Args:
            llm (BaseLLM): The language model to use for generation
        """
        if not llm:
            raise ValueError("LLM instance is required")
        self.llm = llm
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.config = GenerationConfig()
        
    def _validate_parameters(self, prompt: str, config: GenerationConfig) -> None:
        """Validate generation parameters.
        
        Args:
            prompt: The input prompt
            config: Generation configuration
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
            
        if len(prompt) > config.max_context_length:
            raise ValueError(f"Prompt exceeds maximum length of {config.max_context_length} characters")
            
        if config.temperature < 0.0 or config.temperature > 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
            
        if config.top_p < 0.0 or config.top_p > 1.0:
            raise ValueError("Top_p must be between 0.0 and 1.0")
            
        if config.max_tokens is not None and config.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
            
    def _check_response_quality(self, response: str, config: GenerationConfig) -> Tuple[bool, str]:
        """Check the quality of the generated response.
        
        Args:
            response: The generated response
            config: Generation configuration
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not response:
            return False, "Empty response generated"
            
        if len(response) < config.min_response_length:
            return False, f"Response too short (length: {len(response)})"
            
        if len(response) > config.max_response_length:
            return False, f"Response too long (length: {len(response)})"
            
        # Check for common issues
        if re.search(r'\b(undefined|null|error|exception)\b', response.lower()):
            return False, "Response contains error indicators"
            
        if len(re.findall(r'[.!?]', response)) < 1:
            return False, "Response lacks proper sentence structure"
            
        return True, ""
        
    def _prepare_prompt(self, prompt: str, config: GenerationConfig) -> str:
        """Prepare and normalize the prompt.
        
        Args:
            prompt: The input prompt
            config: Generation configuration
            
        Returns:
            Normalized prompt
        """
        # Remove extra whitespace
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        
        # Truncate if necessary
        if len(prompt) > config.max_context_length:
            self.logger.warning(f"Prompt truncated from {len(prompt)} to {config.max_context_length} characters")
            prompt = prompt[:config.max_context_length]
            
        return prompt
        
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        skip_quality_check: bool = False,
        **kwargs: Any
    ) -> str:
        """Generate text based on the provided prompt.
        
        Args:
            prompt: The input prompt for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness in generation (0.0 to 1.0)
            top_p: Controls diversity via nucleus sampling (0.0 to 1.0)
            stop_sequences: Sequences that stop generation
            skip_quality_check: Whether to skip response quality check
            **kwargs: Additional parameters to pass to the LLM
            
        Returns:
            The generated text
            
        Raises:
            ValueError: If prompt is empty or parameters are invalid
            RuntimeError: If generation fails
        """
        # Create config from parameters
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences
        )
        
        # Validate parameters
        self._validate_parameters(prompt, config)
        
        # Prepare prompt
        prompt = self._prepare_prompt(prompt, config)
        
        try:
            # Prepare generation parameters
            generation_params = {
                "temperature": config.temperature,
                "top_p": config.top_p,
                **(kwargs or {})
            }
            
            if config.max_tokens is not None:
                generation_params["max_tokens"] = config.max_tokens
                
            if config.stop_sequences:
                generation_params["stop"] = config.stop_sequences
                
            # Generate response using the LLM
            response = self.llm.generate(prompt, **generation_params)
            response = response.strip()
            
            # Check response quality
            if not skip_quality_check:
                is_valid, error_message = self._check_response_quality(response, config)
                if not is_valid:
                    raise RuntimeError(f"Response quality check failed: {error_message}")
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error during text generation: {str(e)}")
            raise RuntimeError(f"Generation failed: {str(e)}")
            
    def generate_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any
    ) -> str:
        """Generate text with smart retry logic.
        
        Args:
            prompt: The input prompt for generation
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            **kwargs: Additional parameters to pass to generate()
            
        Returns:
            The generated text
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        config = GenerationConfig(max_retries=max_retries, retry_delay=retry_delay)
        last_error = None
        
        for attempt in range(config.max_retries):
            try:
                # Adjust parameters for retry
                if attempt > 0:
                    # Increase temperature slightly for diversity
                    kwargs['temperature'] = min(1.0, kwargs.get('temperature', 0.7) + 0.1)
                    # Reduce max_tokens if specified
                    if 'max_tokens' in kwargs:
                        kwargs['max_tokens'] = int(kwargs['max_tokens'] * 0.9)
                
                response = self.generate(prompt, **kwargs)
                return response
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < config.max_retries - 1:
                    time.sleep(config.retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                    
        raise RuntimeError(f"All generation attempts failed. Last error: {str(last_error)}")


class AnswerGenerator(Generator):
    """A specialized generator for creating and refining answers using language models.
    
    This class extends the base Generator to provide specific functionality for
    generating, refining, and condensing answers using the provided language model.
    """
    
    def __init__(self, llm: BaseLLM):
        """Initialize the AnswerGenerator with a language model.
        
        Args:
            llm (BaseLLM): The language model to use for generation
        """
        super().__init__(llm)
        
    def generate_answer(self, prompt: str, skip_quality_check: bool = False) -> str:
        """Generate an initial answer based on the prompt.
        
        Args:
            prompt: The input prompt for answer generation
            skip_quality_check: Whether to skip response quality check
            
        Returns:
            The generated answer
        """
        return self.generate_with_retry(
            prompt,
            temperature=0.7,
            max_tokens=1000,
            skip_quality_check=skip_quality_check
        )

    def refine_answer(self, raw_answer: str, context: str) -> str:
        """Refine an answer based on provided context.
        
        Args:
            raw_answer: The initial answer to refine
            context: The context to use for refinement
            
        Returns:
            The refined answer
        """
        prompt = f"""Refine the following answer for factual correctness and completeness based on the provided context.
        Ensure the refined answer:
        1. Is factually accurate according to the context
        2. Includes all relevant information
        3. Maintains a professional technical tone
        4. Is well-structured and clear
        
        Original Answer: {raw_answer}
        
        Context: {context}
        
        Refined Answer:"""
        
        return self.generate_with_retry(
            prompt,
            temperature=0.5,  # Lower temperature for more focused refinement
            max_tokens=1500
        )

    def condense_answer(self, refined_answer: str) -> str:
        """Condense and polish an answer in professional technical style.
        
        Args:
            refined_answer: The answer to condense
            
        Returns:
            The condensed and polished answer
        """
        prompt = f"""Condense and polish this answer in professional technical style.
        The condensed answer should:
        1. Be concise while maintaining all key information
        2. Use clear and precise technical language
        3. Have a logical flow and structure
        4. Be free of redundancy
        
        Original Answer: {refined_answer}
        
        Condensed Answer:"""
        
        return self.generate_with_retry(
            prompt,
            temperature=0.3,  # Very low temperature for consistent style
            max_tokens=800
        )
