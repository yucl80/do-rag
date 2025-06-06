from typing import List, Dict, Tuple, Set, Any, Optional
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from .base import BaseRetriever
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from dataclasses import dataclass
from collections import defaultdict
import concurrent.futures
from functools import lru_cache
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import time
from .config import (
    RELATION_WEIGHTS, 
    LEVEL_WEIGHTS, 
    DEFAULT_MODEL_CONFIG, 
    CACHE_CONFIG,
    RETRIEVAL_CONFIG
)

@dataclass
class SubQuery:
    intent: str
    query: str
    embedding: np.ndarray
    confidence: float
    reasoning: str
    required_context: List[str]

class GraphPath:
    def __init__(self, nodes: List[str], edges: List[Dict], score: float):
        self.nodes = nodes
        self.edges = edges
        self.score = score
        self.last_accessed = time.time()

class HybridRetriever(BaseRetriever):
    """An advanced hybrid retrieval system that combines multiple retrieval strategies with LLM-powered query processing.
    
    This retriever is designed for complex scenarios requiring sophisticated query understanding
    and multi-strategy retrieval. It's particularly effective for handling ambiguous queries,
    multi-part questions, and scenarios requiring deep context understanding.
    
    Use Cases:
    1. Complex Question Answering:
       - Multi-part questions requiring different types of information
       - Questions with implicit context or assumptions
       - Queries that need to be decomposed into sub-questions
    
    2. Conversational Systems:
       - Chatbots requiring context awareness
       - Systems needing to maintain conversation history
       - Applications requiring query clarification
    
    3. Research and Analysis:
       - When queries need to be expanded or rephrased
       - When multiple information sources need to be combined
       - When query intent needs to be explicitly understood
    
    Key Features:
    - Query decomposition using LLM
    - Intent analysis and classification
    - Multi-threaded retrieval processing
    - Path caching for performance optimization
    - Conversation history integration
    - Query rewriting with context awareness
    
    Performance Considerations:
    - Higher computational overhead due to LLM integration
    - Benefits from GPU acceleration for embedding models
    - Memory usage scales with cache size and conversation history
    - Best for scenarios where query understanding is crucial
    
    Dependencies:
    - OpenAI API (optional, for advanced query processing)
    - Transformers library for intent classification
    - Spacy for basic NLP tasks
    - NetworkX for graph operations
    
    Example Usage:
        retriever = HybridRetriever(
            graph=knowledge_graph,
            chunks=text_chunks,
            embeddings=chunk_embeddings,
            openai_api_key="your-api-key"  # Optional
        )
        chunks, context = retriever.retrieve(
            query="What are the performance implications of using feature X in scenario Y?",
            query_embedding=query_vector
        )
    """
    
    def __init__(self, graph: nx.MultiDiGraph, chunks: List[str], embeddings: np.ndarray,
                 model_name: str = DEFAULT_MODEL_CONFIG["text_model"],
                 intent_model: str = DEFAULT_MODEL_CONFIG["intent_model"],
                 openai_api_key: Optional[str] = None,
                 cache_size: int = CACHE_CONFIG["path_cache_size"],
                 max_workers: int = CACHE_CONFIG["max_workers"]):
        """
        Initialize the hybrid retriever.
        
        Args:
            graph: Knowledge graph containing relationships between entities
            chunks: List of text chunks for semantic search
            embeddings: Pre-computed embeddings for the text chunks
            model_name: Name of the embedding model to use
            intent_model: Model to use for intent classification
            openai_api_key: Optional OpenAI API key for advanced query processing
            cache_size: Maximum number of paths to cache
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.graph = graph
        self.chunks = chunks
        self.embeddings = embeddings
        self.nlp = spacy.load(DEFAULT_MODEL_CONFIG["spacy_model"])
        self.model_name = model_name
        self.max_workers = max_workers
        
        # Initialize intent analysis model
        self.intent_tokenizer = AutoTokenizer.from_pretrained(intent_model)
        self.intent_model = AutoModel.from_pretrained(intent_model)
        self.intent_classifier = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Initialize OpenAI for advanced query processing
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            self.use_openai = True
        else:
            self.openai_client = None
            self.use_openai = False
        
        # Use weights from configuration
        self.relation_weights = RELATION_WEIGHTS
        self.level_weights = LEVEL_WEIGHTS
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Initialize path cache
        self.path_cache = {}
        self.cache_size = cache_size
        
        # Initialize LLM chains
        self._init_llm_chains()

    def _init_llm_chains(self):
        """Initialize LLM chains for various tasks."""
        if self.use_openai:
            # Query decomposition chain
            self.decomposition_prompt = PromptTemplate(
                input_variables=["query", "conversation_history"],
                template="""Given the following query and conversation history, decompose it into sub-queries with specific intents.

Conversation History:
{conversation_history}

Query: {query}

For each sub-query, provide:
1. Intent (FACTUAL, COMPARATIVE, CAUSAL, ANALYTICAL, or PROCEDURAL)
2. Sub-query text
3. Confidence score (0-1)
4. Reasoning for the decomposition
5. Required context types (e.g., DEFINITION, EXAMPLE, COMPARISON)

Format the response as a JSON array of objects with these fields."""
            )
            
            # Query rewriting chain
            self.rewriting_prompt = PromptTemplate(
                input_variables=["query", "graph_context", "conversation_history"],
                template="""Given the following query, graph context, and conversation history, rewrite the query to be more specific and clear.

Conversation History:
{conversation_history}

Graph Context:
{graph_context}

Original Query: {query}

Rewrite the query to:
1. Resolve any ambiguities
2. Include specific terms from the context
3. Maintain the original intent
4. Be more precise and detailed

Provide the rewritten query."""
            )
            
            # Initialize chains
            def chat_completion_llm(messages, **kwargs):
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    **kwargs
                )
                return response.choices[0].message.content

            self.decomposition_chain = LLMChain(
                llm=chat_completion_llm,
                prompt=self.decomposition_prompt
            )
            
            self.rewriting_chain = LLMChain(
                llm=chat_completion_llm,
                prompt=self.rewriting_prompt
            )

    @lru_cache(maxsize=1000)
    def _get_cached_path(self, start_node: str, end_node: str, max_hops: int) -> Optional[GraphPath]:
        """Get cached path between nodes if available."""
        cache_key = (start_node, end_node, max_hops)
        if cache_key in self.path_cache:
            path = self.path_cache[cache_key]
            path.last_accessed = time.time()
            return path
        return None

    def _cache_path(self, start_node: str, end_node: str, max_hops: int, path: GraphPath):
        """Cache a path between nodes."""
        if len(self.path_cache) >= self.cache_size:
            # Remove least recently accessed path
            oldest_path = min(self.path_cache.items(), key=lambda x: x[1].last_accessed)
            del self.path_cache[oldest_path[0]]
        
        cache_key = (start_node, end_node, max_hops)
        self.path_cache[cache_key] = path

    def _analyze_intent(self, text: str) -> Tuple[str, float]:
        """Analyze the intent of a text using the intent classifier."""
        # Use the intent classifier to get the most likely intent
        result = self.intent_classifier(
            text,
            candidate_labels=[
                "FACTUAL", "COMPARATIVE", "CAUSAL", "ANALYTICAL", "PROCEDURAL"
            ],
            hypothesis_template="This text is asking for {} information."
        )
        
        return result[0]["label"], result[0]["score"]

    def _decompose_query(self, query: str) -> List[SubQuery]:
        """Decompose the query into sub-queries with intents using LLM."""
        if self.use_openai:
            # Get conversation history for context
            history_text = "\n".join([
                f"Q: {item['query']}\nA: [Previous response]"
                for item in self.conversation_history[-3:]
            ])
            
            # Use LLM for decomposition
            response = self.decomposition_chain.run(
                query=query,
                conversation_history=history_text
            )
            
            try:
                sub_queries_data = json.loads(response)
                sub_queries = []
                
                for sq_data in sub_queries_data:
                    # Create embedding for the sub-query
                    embedding = self.nlp(sq_data["query"]).vector
                    
                    sub_queries.append(SubQuery(
                        intent=sq_data["intent"],
                        query=sq_data["query"],
                        embedding=embedding,
                        confidence=sq_data["confidence"],
                        reasoning=sq_data["reasoning"],
                        required_context=sq_data["required_context"]
                    ))
                
                return sub_queries
                
            except json.JSONDecodeError:
                # Fallback to heuristic approach if LLM response is invalid
                pass
        
        # Heuristic approach as fallback
        sub_queries = []
        parts = query.split(" and ")
        
        for part in parts:
            intent, confidence = self._analyze_intent(part)
            embedding = self.nlp(part).vector
            
            sub_queries.append(SubQuery(
                intent=intent,
                query=part.strip(),
                embedding=embedding,
                confidence=confidence,
                reasoning="Heuristic-based decomposition",
                required_context=["GENERAL"]
            ))
        
        return sub_queries

    def _get_graph_context(self, node: str, max_hops: int = 2) -> Tuple[Set[str], Dict[str, float]]:
        """Get context nodes and their relevance scores through advanced graph traversal."""
        context_nodes = set()
        node_scores = {}
        visited = {node}
        queue = [(node, 0, 1.0, [])]  # (node, hops, path_weight, path)
        
        while queue:
            current_node, hops, path_weight, current_path = queue.pop(0)
            
            if hops > max_hops:
                continue
            
            if current_node != node:
                context_nodes.add(current_node)
                node_scores[current_node] = path_weight
                
                # Cache the path if it's not too long
                if len(current_path) <= 3:
                    path = GraphPath(
                        nodes=current_path + [current_node],
                        edges=[],  # You could store edge information here
                        score=path_weight
                    )
                    self._cache_path(node, current_node, max_hops, path)
            
            # Process outgoing edges with parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_neighbor = {}
                
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor not in visited:
                        future = executor.submit(
                            self._process_neighbor,
                            current_node,
                            neighbor,
                            hops,
                            path_weight,
                            current_path
                        )
                        future_to_neighbor[future] = neighbor
                
                for future in concurrent.futures.as_completed(future_to_neighbor):
                    neighbor = future_to_neighbor[future]
                    try:
                        new_path_weight, new_path = future.result()
                        if new_path_weight > 0:
                            visited.add(neighbor)
                            queue.append((neighbor, hops + 1, new_path_weight, new_path))
                    except Exception as e:
                        print(f"Error processing neighbor {neighbor}: {e}")
            
            # Process incoming edges similarly
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_predecessor = {}
                
                for predecessor in self.graph.predecessors(current_node):
                    if predecessor not in visited:
                        future = executor.submit(
                            self._process_neighbor,
                            predecessor,
                            current_node,
                            hops,
                            path_weight,
                            current_path
                        )
                        future_to_predecessor[future] = predecessor
                
                for future in concurrent.futures.as_completed(future_to_predecessor):
                    predecessor = future_to_predecessor[future]
                    try:
                        new_path_weight, new_path = future.result()
                        if new_path_weight > 0:
                            visited.add(predecessor)
                            queue.append((predecessor, hops + 1, new_path_weight, new_path))
                    except Exception as e:
                        print(f"Error processing predecessor {predecessor}: {e}")
        
        return context_nodes, node_scores

    def _process_neighbor(self, source: str, target: str, hops: int, 
                         path_weight: float, current_path: List[str]) -> Tuple[float, List[str]]:
        """Process a neighbor node and calculate its path weight."""
        edge_weight = 1.0
        for edge_key in self.graph[source][target]:
            edge_data = self.graph[source][target][edge_key]
            relation_type = edge_data.get("type")
            edge_weight = max(edge_weight, 
                           self.relation_weights.get(relation_type, 0.8))
        
        target_level = self.graph.nodes[target].get("level", "unknown")
        level_weight = self.level_weights.get(target_level, 0.5)
        
        new_path_weight = path_weight * edge_weight * level_weight * (0.8 ** hops)
        new_path = current_path + [target]
        
        return new_path_weight, new_path

    def _rewrite_query_with_context(self, query: str, graph_context: str) -> str:
        """Rewrite the query using graph context to disambiguate and refine it."""
        if self.use_openai:
            # Get conversation history for context
            history_text = "\n".join([
                f"Q: {item['query']}\nA: [Previous response]"
                for item in self.conversation_history[-3:]
            ])
            
            # Use LLM for rewriting
            rewritten = self.rewriting_chain.run(
                query=query,
                graph_context=graph_context,
                conversation_history=history_text
            )
            
            return rewritten.strip()
        
        # Heuristic approach as fallback
        rewritten = query
        if "it" in query.lower() or "this" in query.lower():
            context_terms = [word for word in graph_context.split() if len(word) > 4]
            if context_terms:
                rewritten = rewritten.replace("it", context_terms[0])
                rewritten = rewritten.replace("this", context_terms[0])
        
        return rewritten

    def retrieve(self, query: str, query_embedding: np.ndarray, top_k: int = 5, **kwargs) -> Tuple[List[str], str]:
        """
        Hybrid retrieval combining query decomposition, graph traversal, and vector search.
        """
        # 1. Decompose query into sub-queries
        sub_queries = self._decompose_query(query)
        
        # 2. Process each sub-query in parallel
        all_relevant_chunks = []
        all_graph_contexts = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_subquery = {
                executor.submit(self._process_subquery, sq, top_k): sq
                for sq in sub_queries
            }
            
            for future in concurrent.futures.as_completed(future_to_subquery):
                sub_query = future_to_subquery[future]
                try:
                    chunks, contexts = future.result()
                    all_relevant_chunks.extend(chunks)
                    all_graph_contexts.extend(contexts)
                except Exception as e:
                    print(f"Error processing sub-query {sub_query.query}: {e}")
        
        # 3. Combine and deduplicate results
        unique_chunks = list(dict.fromkeys(all_relevant_chunks))
        unique_contexts = list(dict.fromkeys(all_graph_contexts))
        
        # 4. Update conversation history
        self.conversation_history.append({
            "query": query,
            "sub_queries": [sq.query for sq in sub_queries],
            "context": unique_contexts
        })
        
        return unique_chunks[:top_k], "\n".join(unique_contexts)

    def _process_subquery(self, sub_query: SubQuery, top_k: int) -> Tuple[List[str], List[str]]:
        """Process a single sub-query to get relevant chunks and context."""
        # 1. Find initial graph nodes through semantic matching
        query_doc = self.nlp(sub_query.query)
        initial_nodes = []
        for node in self.graph.nodes():
            node_doc = self.nlp(node)
            if node_doc.similarity(query_doc) > 0.3:
                initial_nodes.append(node)
        
        # 2. Expand context through graph traversal
        context_nodes = set()
        context_scores = {}
        
        for node in initial_nodes:
            nodes, scores = self._get_graph_context(node)
            context_nodes.update(nodes)
            for ctx_node, score in scores.items():
                context_scores[ctx_node] = max(context_scores.get(ctx_node, 0), score)
        
        # 3. Generate structured context
        structured_context = []
        for node, score in sorted(context_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            node_data = self.graph.nodes[node]
            description = node_data.get("description", "")
            level = node_data.get("level", "unknown")
            structured_context.append(f"[{level}] {node}: {description}")
        
        # 4. Rewrite sub-query using graph context
        rewritten_query = self._rewrite_query_with_context(
            sub_query.query,
            "\n".join(structured_context)
        )
        
        # 5. Vector search with rewritten query
        rewritten_embedding = self.nlp(rewritten_query).vector
        chunk_sims = cosine_similarity([rewritten_embedding], self.embeddings)[0]
        top_chunk_indices = np.argsort(chunk_sims)[-top_k:][::-1]
        relevant_chunks = [self.chunks[i] for i in top_chunk_indices]
        
        return relevant_chunks, structured_context

    def get_structured_prompt(self, query: str, relevant_chunks: List[str], 
                            graph_context: str, **kwargs) -> str:
        """Generate a structured prompt that combines all sources of information."""
        # Get recent conversation history
        recent_history = self.conversation_history[-3:] if self.conversation_history else []
        history_text = "\n".join([
            f"Q: {item['query']}\nA: [Previous response]"
            for item in recent_history
        ])
        
        prompt = f"""Use the following context to answer the question. The context includes:
1. Direct text matches from the vector store
2. Related information from the knowledge graph
3. Recent conversation history for context

Recent Conversation History:
{history_text}

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
5. Takes into account the conversation history for context and continuity

Answer:"""
        return prompt

    def get_retriever_info(self) -> Dict[str, Any]:
        """Get information about the retriever configuration."""
        return {
            "type": "hybrid",
            "model": self.model_name,
            "has_graph": self.graph is not None,
            "num_chunks": len(self.chunks),
            "relation_weights": self.relation_weights,
            "level_weights": self.level_weights,
            "conversation_history_length": len(self.conversation_history),
            "cache_size": len(self.path_cache),
            "max_workers": self.max_workers,
            "use_openai": self.use_openai
        } 