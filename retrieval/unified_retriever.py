from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Set, Any
import spacy
from .base import BaseRetriever
from .config import RELATION_WEIGHTS, LEVEL_WEIGHTS, DEFAULT_MODEL_CONFIG, RETRIEVAL_CONFIG

class UnifiedRetriever(BaseRetriever):
    """A unified retrieval system that combines semantic search with knowledge graph traversal.
    
    This retriever is designed for scenarios where you need to combine direct semantic
    matching with structured knowledge from a graph database. It's particularly effective
    when dealing with hierarchical or relational data where context and relationships
    are important.
    
    Use Cases:
    1. Technical Documentation:
       - When retrieving information from structured documentation
       - When relationships between components are important
       - When hierarchical organization of information matters
    
    2. Knowledge Base Systems:
       - For FAQ systems with related questions
       - When answers need to consider multiple related concepts
       - For systems requiring context-aware responses
    
    3. Code Documentation:
       - When retrieving information about code components
       - When understanding dependencies is crucial
       - For systems that need to consider code structure
    
    Key Features:
    - Semantic search using embeddings
    - Graph-based context expansion
    - Weighted relationship traversal
    - Hierarchical importance consideration
    - Structured prompt generation
    
    Performance Considerations:
    - Best for medium-sized knowledge graphs (up to 100k nodes)
    - Efficient for queries requiring both direct matches and context
    - Memory usage scales with graph size and embedding dimensions
    
    Example Usage:
        retriever = UnifiedRetriever(
            graph=knowledge_graph,
            chunks=text_chunks,
            embeddings=chunk_embeddings
        )
        chunks, context = retriever.retrieve(
            query="How does component A interact with component B?",
            query_embedding=query_vector
        )
    """
    
    def __init__(self, graph: nx.MultiDiGraph, chunks: List[str], embeddings: np.ndarray, 
                 model_name: str = DEFAULT_MODEL_CONFIG["text_model"]):
        """
        Initialize the unified retriever.
        
        Args:
            graph: Knowledge graph containing relationships between entities
            chunks: List of text chunks for semantic search
            embeddings: Pre-computed embeddings for the text chunks
            model_name: Name of the embedding model to use
        """
        self.graph = graph
        self.chunks = chunks
        self.embeddings = embeddings
        self.nlp = spacy.load(DEFAULT_MODEL_CONFIG["spacy_model"])
        self.model_name = model_name
        
        # Use weights from configuration
        self.relation_weights = RELATION_WEIGHTS
        self.level_weights = LEVEL_WEIGHTS

    def _get_node_embedding(self, node_text: str) -> np.ndarray:
        """Get embedding for a node text using the same model as chunks."""
        return self.nlp(node_text).vector

    def _calculate_semantic_similarity(self, query_embedding: np.ndarray, node_embedding: np.ndarray) -> float:
        """Calculate semantic similarity between query and node embeddings."""
        return cosine_similarity([query_embedding], [node_embedding])[0][0]

    def _get_graph_context(self, node: str, max_hops: int = 2) -> Tuple[Set[str], Dict[str, float]]:
        """Get context nodes and their relevance scores through graph traversal."""
        context_nodes = set()
        node_scores = {}
        visited = {node}
        queue = [(node, 0, 1.0)]  # (node, hops, path_weight)
        
        while queue:
            current_node, hops, path_weight = queue.pop(0)
            
            if hops > max_hops:
                continue
                
            # Add current node to context if not the original node
            if current_node != node:
                context_nodes.add(current_node)
                node_scores[current_node] = path_weight
            
            # Process outgoing edges
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    # Calculate edge weight based on relation type
                    edge_weight = 1.0
                    for edge_key in self.graph[current_node][neighbor]:
                        edge_data = self.graph[current_node][neighbor][edge_key]
                        relation_type = edge_data.get("type")
                        edge_weight = max(edge_weight, 
                                       self.relation_weights.get(relation_type, 0.8))
                    
                    # Apply level-based weight adjustment
                    neighbor_level = self.graph.nodes[neighbor].get("level", "unknown")
                    level_weight = self.level_weights.get(neighbor_level, 0.5)
                    
                    # Calculate new path weight with decay
                    new_path_weight = path_weight * edge_weight * level_weight * (0.8 ** hops)
                    queue.append((neighbor, hops + 1, new_path_weight))
            
            # Process incoming edges
            for predecessor in self.graph.predecessors(current_node):
                if predecessor not in visited:
                    visited.add(predecessor)
                    # Similar weight calculation for incoming edges
                    edge_weight = 1.0
                    for edge_key in self.graph[predecessor][current_node]:
                        edge_data = self.graph[predecessor][current_node][edge_key]
                        relation_type = edge_data.get("type")
                        edge_weight = max(edge_weight, 
                                       self.relation_weights.get(relation_type, 0.8))
                    
                    predecessor_level = self.graph.nodes[predecessor].get("level", "unknown")
                    level_weight = self.level_weights.get(predecessor_level, 0.5)
                    
                    new_path_weight = path_weight * edge_weight * level_weight * (0.8 ** hops)
                    queue.append((predecessor, hops + 1, new_path_weight))
        
        return context_nodes, node_scores

    def retrieve(self, query: str, query_embedding: np.ndarray, top_k: int = 5, **kwargs) -> Tuple[List[str], str]:
        """
        Unified retrieval combining semantic search and graph traversal.
        Returns both relevant chunks and structured graph context.
        """
        # 1. Initial semantic search to find relevant chunks
        chunk_sims = cosine_similarity([query_embedding], self.embeddings)[0]
        top_chunk_indices = np.argsort(chunk_sims)[-top_k:][::-1]
        relevant_chunks = [self.chunks[i] for i in top_chunk_indices]
        
        # 2. Find initial graph nodes through semantic matching
        query_doc = self.nlp(query)
        initial_nodes = []
        for node in self.graph.nodes():
            node_doc = self.nlp(node)
            if node_doc.similarity(query_doc) > 0.3:  # Threshold for semantic matching
                initial_nodes.append(node)
        
        # 3. Expand context through graph traversal
        all_context_nodes = set()
        context_scores = {}
        
        for node in initial_nodes:
            context_nodes, node_scores = self._get_graph_context(node)
            all_context_nodes.update(context_nodes)
            
            # Merge scores, taking maximum for overlapping nodes
            for ctx_node, score in node_scores.items():
                context_scores[ctx_node] = max(context_scores.get(ctx_node, 0), score)
        
        # 4. Combine and rank context nodes
        ranked_context = sorted(
            [(node, score) for node, score in context_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 5. Generate structured context
        structured_context = []
        for node, score in ranked_context[:top_k]:
            node_data = self.graph.nodes[node]
            description = node_data.get("description", "")
            level = node_data.get("level", "unknown")
            structured_context.append(f"[{level}] {node}: {description}")
        
        # 6. Combine chunks and structured context
        combined_context = "\n".join(structured_context)
        
        return relevant_chunks, combined_context

    def get_structured_prompt(self, query: str, relevant_chunks: List[str], 
                            graph_context: str, **kwargs) -> str:
        """Generate a structured prompt that combines both semantic and graph-based context."""
        prompt = f"""Use the following context to answer the question. The context includes both direct text matches and related information from the knowledge graph.

Direct Text Matches:
{chr(10).join(relevant_chunks)}

Knowledge Graph Context (showing hierarchical relationships):
{graph_context}

Question: {query}

Please provide a comprehensive answer that:
1. Directly addresses the question using the most relevant information
2. Includes related context from the knowledge graph when it adds value
3. Maintains the hierarchical structure of the information
4. Cites specific details from both the direct matches and graph context

Answer:"""
        return prompt

    def get_retriever_info(self) -> Dict[str, Any]:
        """Get information about the retriever configuration."""
        return {
            "type": "unified",
            "model": self.model_name,
            "has_graph": self.graph is not None,
            "num_chunks": len(self.chunks),
            "relation_weights": self.relation_weights,
            "level_weights": self.level_weights
        } 