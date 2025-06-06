from typing import Optional, Dict, Any, List, Tuple
from generation.base import BaseLLM
from generation.grounded_generator import GroundedGenerator, GroundedAnswer
from retrieval.base import BaseRetriever
from ingest.multimodal_processor import MultimodalProcessor
from kg.builder import KnowledgeGraphBuilder
import numpy as np

class ModularRAGPipeline:
    """A modular RAG pipeline that can work with different LLM and retriever implementations."""
    
    def __init__(self, 
                 llm: BaseLLM,
                 retriever: BaseRetriever,
                 processor: Optional[MultimodalProcessor] = None,
                 kg_builder: Optional[KnowledgeGraphBuilder] = None):
        """
        Initialize the pipeline with modular components.
        
        Args:
            llm: An implementation of BaseLLM
            retriever: An implementation of BaseRetriever
            processor: Optional multimodal document processor
            kg_builder: Optional knowledge graph builder
        """
        self.llm = llm
        self.retriever = retriever
        self.processor = processor
        self.kg_builder = kg_builder
        
    def process_document(self, file_path: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """
        Process a document and prepare it for retrieval.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Tuple containing:
            - Dictionary of processed content
            - Dictionary of content embeddings
        """
        if not self.processor:
            raise ValueError("Document processor is required for document processing")
            
        # Process document
        content = self.processor.process_document(file_path)
        
        # Generate embeddings
        embeddings = self.processor.get_embeddings(content)
        
        # Build knowledge graph if builder is available
        if self.kg_builder:
            # Process text chunks for knowledge graph
            if 'text_chunks' in content:
                for chunk in content['text_chunks']:
                    self.kg_builder.extract_entities_and_relations_with_llm(chunk)
            
            # Process table content for knowledge graph
            if 'tables' in content:
                for table in content['tables']:
                    # Convert table to text representation
                    table_text = self._table_to_text(table)
                    self.kg_builder.extract_entities_and_relations_with_llm(table_text)
            
            # Process image content for knowledge graph
            if 'images' in content:
                for image in content['images']:
                    if 'text' in image:
                        self.kg_builder.extract_entities_and_relations_with_llm(image['text'])
        
        return content, embeddings
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table data to text representation."""
        text_parts = []
        
        # Add sheet name if available
        if 'sheet' in table:
            text_parts.append(f"Table in sheet: {table['sheet']}")
        
        # Add table data
        if 'data' in table:
            for row in table['data']:
                row_text = []
                for key, value in row.items():
                    row_text.append(f"{key}: {value}")
                text_parts.append(" | ".join(row_text))
        
        return "\n".join(text_parts)
    
    def answer_query(self, 
                    query: str, 
                    query_embedding: Any,
                    content_types: Optional[List[str]] = None,
                    **kwargs) -> GroundedAnswer:
        """
        Answer a query using the configured LLM and retriever with grounded generation.
        
        Args:
            query: The query string
            query_embedding: The embedding vector of the query
            content_types: List of content types to search in
            **kwargs: Additional parameters for the pipeline components
            
        Returns:
            GroundedAnswer containing the answer, citations, and follow-up questions
        """
        # Retrieve relevant information
        relevant_content, graph_context = self.retriever.retrieve(
            query, query_embedding, content_types=content_types, **kwargs
        )
        
        # Initialize grounded generator
        grounded_generator = GroundedGenerator(self.llm)
        
        # Generate grounded answer with both content and graph context
        return grounded_generator.generate(
            query, 
            relevant_content,
            graph_context=graph_context
        )
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        info = {
            "llm": self.llm.get_model_info(),
            "has_processor": self.processor is not None,
            "has_kg_builder": self.kg_builder is not None
        }
        
        if self.retriever is not None:
            info["retriever"] = self.retriever.get_retriever_info()
        else:
            info["retriever"] = None
            
        return info 