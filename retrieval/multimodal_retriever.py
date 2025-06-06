from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import spacy
import logging
import time
import os
import json
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from .base import BaseRetriever
from .config import (
    RELATION_WEIGHTS, 
    LEVEL_WEIGHTS,
    CONTENT_WEIGHTS,
    DEFAULT_MODEL_CONFIG,
    RETRIEVAL_CONFIG,
    CACHE_CONFIG,
    LOGGING_CONFIG
)

class MultimodalRetriever(BaseRetriever):
    """A multimodal retrieval system that handles multiple types of content simultaneously.
    
    This retriever is designed for scenarios where information needs to be retrieved
    from multiple modalities (text, images, tables, etc.) and combined into a coherent
    response. It's particularly effective for rich content that spans different formats
    and requires unified retrieval strategies.
    
    Use Cases:
    1. Rich Document Processing:
       - Documents containing text, images, and tables
       - Technical documentation with diagrams
       - Research papers with figures and data tables
    
    2. Visual Question Answering:
       - Questions about images or diagrams
       - Queries requiring visual context
       - Image-based information retrieval
    
    3. Data Analysis and Reporting:
       - Retrieving information from structured tables
       - Combining numerical data with textual context
       - Multi-format data presentation
    
    Key Features:
    - Multi-modal content support (text, images, tables)
    - Specialized retrieval strategies per content type
    - Unified ranking and scoring system
    - Graph-based context enhancement
    - Structured prompt generation for multi-modal content
    
    Supported Content Types:
    - Text: Standard text chunks with embeddings
    - Images: Image features and metadata
    - Tables: Structured data with semantic search
    - (Extensible to other content types)
    
    Performance Considerations:
    - Memory usage scales with number of content types
    - Each content type requires its own embedding model
    - Best for scenarios with diverse content types
    - Efficient for unified retrieval across modalities
    
    Example Usage:
        retriever = MultimodalRetriever(
            graph=knowledge_graph,
            content={
                'text': text_chunks,
                'images': image_features,
                'tables': table_data
            },
            embeddings={
                'text': text_embeddings,
                'image': image_embeddings
            }
        )
        results, context = retriever.retrieve(
            query="Show me the performance metrics and related diagrams",
            query_embedding=query_vector,
            content_types=['text', 'image', 'table']
        )
    """
    
    def __init__(self, 
                 graph: nx.MultiDiGraph,
                 content: Dict[str, Any],
                 embeddings: Dict[str, np.ndarray],
                 model_name: str = DEFAULT_MODEL_CONFIG["text_model"]):
        """
        Initialize the multimodal retriever.
        
        Args:
            graph: Knowledge graph containing relationships between entities
            content: Dictionary containing different types of content:
                    - 'text': List of text chunks
                    - 'images': List of image features
                    - 'tables': List of table data
            embeddings: Dictionary containing embeddings for different content types:
                       - 'text': Text chunk embeddings
                       - 'image': Image feature embeddings
            model_name: Name of the text embedding model to use
        """
        self.graph = graph
        self.content = content
        self.embeddings = embeddings
        self.model_name = model_name
        
        # Initialize models
        try:
            self.nlp = spacy.load(DEFAULT_MODEL_CONFIG["spacy_model"])
        except Exception as e:
            logging.error(f"Failed to load spaCy model: {str(e)}")
            raise
            
        # Setup logging
        self._setup_logging()
        
        # Initialize cache
        self._setup_cache()
        
        # Load weights from configuration
        self.relation_weights = RELATION_WEIGHTS
        self.level_weights = LEVEL_WEIGHTS
        self.content_weights = CONTENT_WEIGHTS
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=CACHE_CONFIG["max_workers"])
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG["level"]),
            format=LOGGING_CONFIG["format"]
        )
        
        if LOGGING_CONFIG["handlers"]["file"]["enabled"]:
            file_handler = logging.FileHandler(LOGGING_CONFIG["handlers"]["file"]["filename"])
            file_handler.setFormatter(logging.Formatter(LOGGING_CONFIG["format"]))
            logging.getLogger().addHandler(file_handler)
            
        self.logger = logging.getLogger(__name__)
        
    def _setup_cache(self):
        """Setup cache directory and configuration."""
        cache_dir = Path(CACHE_CONFIG["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache = {
            "embeddings": {},
            "graph_paths": {},
            "similarity_scores": {}
        }
        
        # Load existing cache if available
        for cache_type in self.cache.keys():
            cache_file = cache_dir / f"{cache_type}_cache.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        self.cache[cache_type] = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load {cache_type} cache: {str(e)}")
                    
    def _save_cache(self):
        """Save cache to disk."""
        cache_dir = Path(CACHE_CONFIG["cache_dir"])
        for cache_type, cache_data in self.cache.items():
            if CACHE_CONFIG["cache_types"][cache_type]:
                cache_file = cache_dir / f"{cache_type}_cache.json"
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f)
                except Exception as e:
                    self.logger.error(f"Failed to save {cache_type} cache: {str(e)}")
                    
    @lru_cache(maxsize=CACHE_CONFIG["path_cache_size"])
    def _get_cached_path(self, start_node: str, end_node: str) -> Optional[List[str]]:
        """Get cached path between nodes."""
        return self.cache["graph_paths"].get(f"{start_node}:{end_node}")
        
    def _cache_path(self, start_node: str, end_node: str, path: List[str]):
        """Cache path between nodes."""
        self.cache["graph_paths"][f"{start_node}:{end_node}"] = path
        
    def retrieve(self, 
                query: str, 
                query_embedding: np.ndarray,
                content_types: Optional[List[str]] = None,
                top_k: int = RETRIEVAL_CONFIG["default_top_k"],
                min_similarity: float = RETRIEVAL_CONFIG["similarity_threshold"],
                **kwargs) -> Tuple[Dict[str, List[Any]], str]:
        """
        Retrieve relevant information from different content types.
        
        Args:
            query: The query string
            query_embedding: The embedding vector of the query
            content_types: List of content types to search in
            top_k: Number of results to return per content type
            min_similarity: Minimum similarity threshold for results
            **kwargs: Additional parameters for retrieval
            
        Returns:
            Tuple containing:
            - Dictionary of retrieved content by type
            - Structured context information
        """
        start_time = time.time()
        self.logger.info(f"Starting retrieval for query: {query}")
        
        if content_types is None:
            content_types = list(self.content.keys())
            
        results = {}
        scores = {}
        
        try:
            # Retrieve text content
            if 'text' in content_types and 'text' in self.embeddings:
                text_results, text_scores = self._retrieve_text(
                    query_embedding,
                    min_similarity,
                    top_k
                )
                if text_results:
                    results['text'] = text_results
                    scores['text'] = text_scores
                    
            # Retrieve image content
            if 'image' in content_types and 'image' in self.embeddings:
                image_results, image_scores = self._retrieve_images(
                    query_embedding,
                    min_similarity,
                    top_k
                )
                if image_results:
                    results['image'] = image_results
                    scores['image'] = image_scores
                    
            # Retrieve table content
            if 'table' in content_types and 'tables' in self.content:
                table_results, table_scores = self._retrieve_tables(
                    query,
                    min_similarity,
                    top_k
                )
                if table_results:
                    results['table'] = table_results
                    scores['table'] = table_scores
                    
            # Get graph context
            graph_context = self._get_graph_context(query)
            
            # Apply diversity penalty and final ranking
            final_results = self._rank_results(results, scores)
            
            self.logger.info(f"Retrieval completed in {time.time() - start_time:.2f} seconds")
            return final_results, graph_context
            
        except Exception as e:
            self.logger.error(f"Error during retrieval: {str(e)}")
            raise RuntimeError(f"Retrieval failed: {str(e)}")
        
    def _retrieve_text(self,
                      query_embedding: np.ndarray,
                      min_similarity: float,
                      top_k: int) -> Tuple[List[str], List[float]]:
        """Retrieve relevant text content."""
        text_chunks = self.content.get('text_chunks', [])
        text_embeddings = self.embeddings['text']
        
        if not text_chunks or text_embeddings is None:
            return [], []
            
        # Calculate similarities
        text_sims = cosine_similarity([query_embedding], text_embeddings)[0]
        
        # Filter and sort
        valid_indices = np.where(text_sims >= min_similarity)[0]
        if len(valid_indices) == 0:
            return [], []
            
        top_indices = valid_indices[np.argsort(text_sims[valid_indices])[-top_k:][::-1]]
        return [text_chunks[i] for i in top_indices], text_sims[top_indices].tolist()
        
    def _retrieve_images(self,
                        query_embedding: np.ndarray,
                        min_similarity: float,
                        top_k: int) -> Tuple[List[Dict], List[float]]:
        """Retrieve relevant image content."""
        image_features = self.embeddings['image']
        if image_features is None:
            return [], []
            
        # Calculate similarities
        image_sims = cosine_similarity([query_embedding], image_features)[0]
        
        # Filter and sort
        valid_indices = np.where(image_sims >= min_similarity)[0]
        if len(valid_indices) == 0:
            return [], []
            
        top_indices = valid_indices[np.argsort(image_sims[valid_indices])[-top_k:][::-1]]
        return [self.content['images'][i] for i in top_indices], image_sims[top_indices].tolist()
        
    def _retrieve_tables(self,
                        query: str,
                        min_similarity: float,
                        top_k: int) -> Tuple[List[Dict], List[float]]:
        """Retrieve relevant table content."""
        tables = self.content['tables']
        if not tables:
            return [], []
            
        query_doc = self.nlp(query)
        table_scores = []
        
        for table in tables:
            score = 0
            # Check table metadata
            if 'metadata' in table:
                for key, value in table['metadata'].items():
                    if isinstance(value, str):
                        score += query_doc.similarity(self.nlp(value)) * 1.5
                        
            # Check column headers
            if 'headers' in table:
                for header in table['headers']:
                    score += query_doc.similarity(self.nlp(header)) * 1.2
                    
            # Check cell values with context
            if 'data' in table:
                for row in table['data']:
                    row_context = " ".join(str(v) for v in row.values())
                    row_doc = self.nlp(row_context)
                    score += query_doc.similarity(row_doc)
                    
                    # Check individual cells for exact matches
                    for cell in row.values():
                        if isinstance(cell, (str, int, float)):
                            cell_doc = self.nlp(str(cell))
                            score += query_doc.similarity(cell_doc) * 0.8
                            
            table_scores.append(score)
            
        # Filter and sort tables
        valid_indices = [i for i, score in enumerate(table_scores) if score >= min_similarity]
        if not valid_indices:
            return [], []
            
        top_indices = sorted(valid_indices, key=lambda i: table_scores[i], reverse=True)[:top_k]
        return [tables[i] for i in top_indices], [table_scores[i] for i in top_indices]
        
    def _rank_results(self,
                     results: Dict[str, List[Any]],
                     scores: Dict[str, List[float]]) -> Dict[str, List[Any]]:
        """Rank and filter results using diversity penalty."""
        ranked_results = {}
        
        for content_type, content_list in results.items():
            if not content_list:
                continue
                
            content_scores = scores[content_type]
            content_weight = self.content_weights.get(content_type, 1.0)
            
            # Apply content type weight
            weighted_scores = [score * content_weight for score in content_scores]
            
            # Apply diversity penalty
            final_scores = []
            used_embeddings = []
            
            for i, (content, score) in enumerate(zip(content_list, weighted_scores)):
                if i == 0:
                    final_scores.append(score)
                    if isinstance(content, str):
                        used_embeddings.append(self.nlp(content).vector)
                    continue
                    
                # Calculate similarity with previous results
                if isinstance(content, str):
                    current_embedding = self.nlp(content).vector
                    max_similarity = max(
                        cosine_similarity([current_embedding], [emb])[0][0]
                        for emb in used_embeddings
                    )
                    
                    # Apply diversity penalty
                    diversity_penalty = RETRIEVAL_CONFIG["ranking"]["diversity_penalty"]
                    final_score = score * (1 - max_similarity * diversity_penalty)
                    final_scores.append(final_score)
                    used_embeddings.append(current_embedding)
                else:
                    final_scores.append(score)
                    
            # Sort by final scores
            sorted_indices = np.argsort(final_scores)[::-1]
            ranked_results[content_type] = [content_list[i] for i in sorted_indices]
            
        return ranked_results
        
    def _get_graph_context(self, query: str, max_hops: int = RETRIEVAL_CONFIG["max_hops"]) -> str:
        """Get context from the knowledge graph."""
        query_doc = self.nlp(query)
        initial_nodes = []
        
        # Find initial nodes through semantic matching
        for node in self.graph.nodes():
            node_doc = self.nlp(node)
            if node_doc.similarity(query_doc) > RETRIEVAL_CONFIG["similarity_threshold"]:
                initial_nodes.append(node)
                
        if not initial_nodes:
            return ""
            
        # Expand context through graph traversal
        all_context_nodes = set()
        context_scores = {}
        
        for node in initial_nodes:
            context_nodes, node_scores = self._traverse_graph(node, max_hops)
            all_context_nodes.update(context_nodes)
            
            # Merge scores
            for ctx_node, score in node_scores.items():
                context_scores[ctx_node] = max(context_scores.get(ctx_node, 0), score)
                
        # Generate structured context
        structured_context = []
        for node, score in sorted(context_scores.items(), key=lambda x: x[1], reverse=True):
            node_data = self.graph.nodes[node]
            description = node_data.get("description", "")
            level = node_data.get("level", "unknown")
            structured_context.append(f"[{level}] {node}: {description}")
            
        return "\n".join(structured_context)
        
    def _traverse_graph(self, 
                       start_node: str, 
                       max_hops: int) -> Tuple[set, Dict[str, float]]:
        """Traverse the graph to find relevant context."""
        context_nodes = set()
        node_scores = {}
        visited = {start_node}
        queue = [(start_node, 0, 1.0)]  # (node, hops, path_weight)
        
        while queue:
            current_node, hops, path_weight = queue.pop(0)
            
            if hops > max_hops:
                continue
                
            if current_node != start_node:
                context_nodes.add(current_node)
                node_scores[current_node] = path_weight
                
            # Process outgoing edges
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    edge_weight = self._calculate_edge_weight(current_node, neighbor)
                    neighbor_level = self.graph.nodes[neighbor].get("level", "unknown")
                    level_weight = self.level_weights.get(neighbor_level, 0.5)
                    
                    # Optimized path weight calculation
                    hop_decay = 0.85 ** hops  # Slower decay for better long-range connections
                    semantic_weight = self._calculate_semantic_weight(current_node, neighbor)
                    
                    new_path_weight = path_weight * edge_weight * level_weight * hop_decay * semantic_weight
                    queue.append((neighbor, hops + 1, new_path_weight))
                    
            # Process incoming edges
            for predecessor in self.graph.predecessors(current_node):
                if predecessor not in visited:
                    visited.add(predecessor)
                    edge_weight = self._calculate_edge_weight(predecessor, current_node)
                    predecessor_level = self.graph.nodes[predecessor].get("level", "unknown")
                    level_weight = self.level_weights.get(predecessor_level, 0.5)
                    
                    # Optimized path weight calculation
                    hop_decay = 0.85 ** hops  # Slower decay for better long-range connections
                    semantic_weight = self._calculate_semantic_weight(predecessor, current_node)
                    
                    new_path_weight = path_weight * edge_weight * level_weight * hop_decay * semantic_weight
                    queue.append((predecessor, hops + 1, new_path_weight))
                    
        return context_nodes, node_scores
        
    def _calculate_edge_weight(self, source: str, target: str) -> float:
        """Calculate edge weight based on relation type."""
        edge_weight = 1.0
        for edge_key in self.graph[source][target]:
            edge_data = self.graph[source][target][edge_key]
            relation_type = edge_data.get("type")
            edge_weight = max(edge_weight, 
                           self.relation_weights.get(relation_type, 0.8))
        return edge_weight
        
    def _calculate_semantic_weight(self, source: str, target: str) -> float:
        """Calculate semantic similarity weight between nodes."""
        source_doc = self.nlp(source)
        target_doc = self.nlp(target)
        return source_doc.similarity(target_doc)
        
    def get_structured_prompt(self, 
                            query: str,
                            relevant_content: Dict[str, List[Any]],
                            graph_context: str,
                            **kwargs) -> str:
        """Generate a structured prompt combining different types of content."""
        prompt_parts = []
        
        # Add text content
        if 'text' in relevant_content:
            prompt_parts.append("Text Content:")
            for chunk in relevant_content['text']:
                prompt_parts.append(f"- {chunk}")
                
        # Add image content
        if 'image' in relevant_content:
            prompt_parts.append("\nImage Content:")
            for img in relevant_content['image']:
                prompt_parts.append(f"- Image on page {img.get('page', 'N/A')}")
                if 'text' in img:
                    prompt_parts.append(f"  Text: {img['text']}")
                    
        # Add table content
        if 'table' in relevant_content:
            prompt_parts.append("\nTable Content:")
            for table in relevant_content['table']:
                if 'sheet' in table:
                    prompt_parts.append(f"- Sheet: {table['sheet']}")
                prompt_parts.append("  Data:")
                for row in table['data'][:3]:  # Show first 3 rows
                    prompt_parts.append(f"    {row}")
                    
        # Add graph context
        if graph_context:
            prompt_parts.append("\nKnowledge Graph Context:")
            prompt_parts.append(graph_context)
            
        # Add query and instructions
        prompt_parts.append(f"\nQuestion: {query}")
        prompt_parts.append("""
Please provide a comprehensive answer that:
1. Directly addresses the question using the most relevant information
2. Includes related context from the knowledge graph when it adds value
3. Maintains the hierarchical structure of the information
4. Cites specific details from all available content types
5. Explains how different content types contribute to the answer

Answer:""")
        
        return "\n".join(prompt_parts)
        
    def get_retriever_info(self) -> Dict[str, Any]:
        """Get information about the retriever configuration."""
        return {
            "type": "multimodal",
            "model": self.model_name,
            "has_graph": self.graph is not None,
            "content_types": list(self.content.keys()),
            "relation_weights": self.relation_weights,
            "level_weights": self.level_weights,
            "content_weights": self.content_weights,
            "cache_enabled": all(CACHE_CONFIG["cache_types"].values()),
            "cache_size": {
                "embeddings": len(self.cache["embeddings"]),
                "graph_paths": len(self.cache["graph_paths"]),
                "similarity_scores": len(self.cache["similarity_scores"])
            }
        } 