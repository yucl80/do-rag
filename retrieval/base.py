from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

class BaseRetriever(ABC):
    """Abstract base class for retrieval implementations.
    
    This class defines the core interface that all retrievers must implement.
    It provides a standardized way to retrieve and structure information from
    various data sources.
    
    Key Features:
    - Abstract interface for information retrieval
    - Standardized method signatures for all retrievers
    - Common contract for structured prompt generation
    - Consistent retriever information reporting
    
    All concrete retriever implementations should inherit from this class and
    implement its abstract methods to ensure consistent behavior across
    different retrieval strategies.
    """
    
    @abstractmethod
    def retrieve(self, query: str, query_embedding: np.ndarray, **kwargs) -> Tuple[List[str], str]:
        """
        Retrieve relevant information for the query.
        
        Args:
            query: The query string
            query_embedding: The embedding vector of the query
            **kwargs: Additional parameters specific to the retriever
            
        Returns:
            Tuple containing:
            - List of relevant text chunks
            - Structured context information
        """
        pass
    
    @abstractmethod
    def get_structured_prompt(self, query: str, relevant_chunks: List[str], 
                            context: str, **kwargs) -> str:
        """
        Generate a structured prompt combining retrieved information.
        
        Args:
            query: The original query string
            relevant_chunks: List of retrieved text chunks
            context: Additional structured context information
            **kwargs: Additional parameters for prompt customization
            
        Returns:
            A structured prompt string that combines all retrieved information
            in a format suitable for downstream processing
        """
        pass
    
    @abstractmethod
    def get_retriever_info(self) -> Dict[str, Any]:
        """
        Get information about the retriever configuration.
        
        Returns:
            Dictionary containing configuration details such as:
            - retriever type
            - model information
            - feature flags
            - configuration parameters
        """
        pass 