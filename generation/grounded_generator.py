from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
from .base import BaseLLM
from .generator import Generator
from .refiner import AnswerRefiner
import networkx as nx
from typing_extensions import TypedDict

class Evidence(TypedDict):
    source: str
    text: str

@dataclass
class GroundedAnswer:
    """A class representing a grounded answer with citations and metadata.
    
    Attributes:
        answer (str): The final generated answer
        citations (List[Dict[str, str]]): List of citations with source and text
        follow_up_questions (List[str]): List of relevant follow-up questions
        confidence (float): Confidence score between 0 and 1
    """
    answer: str
    citations: List[Dict[str, str]]
    follow_up_questions: List[str]
    confidence: float

class GroundedGenerator:
    """A generator that produces grounded answers based on provided evidence.
    
    This class implements a multi-stage prompting strategy to generate answers
    that are grounded in the provided evidence, with citations and follow-up questions.
    """
    
    def __init__(self, llm: BaseLLM, graph: Optional[nx.MultiDiGraph] = None):
        """Initialize the GroundedGenerator.
        
        Args:
            llm (BaseLLM): The language model to use for generation
            graph (Optional[nx.MultiDiGraph]): Optional knowledge graph for refinement
        """
        if not llm:
            raise ValueError("LLM instance is required")
            
        self.llm = llm
        self.generator = Generator(llm)
        self.refiner = AnswerRefiner(graph) if graph else None

    def set_graph(self, graph: nx.MultiDiGraph) -> None:
        """Update the knowledge graph used for refinement.
        
        Args:
            graph (nx.MultiDiGraph): The new knowledge graph to use
        """
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("Graph must be an instance of nx.MultiDiGraph")
        self.refiner = AnswerRefiner(graph)

    def _create_initial_prompt(self, query: str, evidence: List[Evidence]) -> str:
        """Create the initial prompt that instructs the LLM to answer based only on evidence.
        
        Args:
            query (str): The question to answer
            evidence (List[Evidence]): List of evidence items
            
        Returns:
            str: The formatted prompt
        """
        if not query or not evidence:
            raise ValueError("Query and evidence are required")
            
        evidence_text = "\n\n".join([f"Source {i+1}: {e['text']}" for i, e in enumerate(evidence)])
        
        return f"""Based ONLY on the following evidence, answer the question. If the evidence is insufficient, respond with "I do not know."
Do not include any information not directly supported by the evidence.

Question: {query}

Evidence:
{evidence_text}

Answer:"""

    def _create_refinement_prompt(self, initial_answer: str, query: str) -> str:
        """Create the refinement prompt to restructure and validate the answer."""
        return f"""Refine and validate the following answer to ensure it:
1. Directly addresses the original question
2. Only contains information from the provided evidence
3. Is clear and well-structured
4. Maintains a professional tone

Original Question: {query}

Initial Answer:
{initial_answer}

Refined Answer:"""

    def _create_condensation_prompt(self, refined_answer: str, query: str) -> str:
        """Create the condensation prompt to align tone and style with the query."""
        return f"""Condense and align the following answer to match the tone and style of the original question.
Maintain all factual information while making the response more concise and engaging.

Original Question: {query}

Refined Answer:
{refined_answer}

Condensed Answer:"""

    def _create_followup_prompt(self, final_answer: str, query: str) -> str:
        """Create a prompt to generate relevant follow-up questions."""
        return f"""Based on the following answer and original question, generate 3 relevant follow-up questions that:
1. Explore related aspects not covered in the answer
2. Request clarification on specific points
3. Are natural and engaging

Original Question: {query}

Answer:
{final_answer}

Follow-up Questions:"""

    def _extract_citations(self, answer: str, evidence: List[Evidence]) -> List[Dict[str, str]]:
        """Extract citations from the answer by matching against evidence."""
        citations = []
        for i, e in enumerate(evidence):
            if e['text'] in answer:
                citations.append({
                    'source': e.get('source', f'Source {i+1}'),
                    'text': e['text']
                })
        return citations

    def generate(self, 
                query: str, 
                evidence: List[Evidence],
                graph_context: Optional[Dict[str, Any]] = None) -> GroundedAnswer:
        """Generate a grounded answer using the staged prompting strategy.
        
        Args:
            query (str): The question to answer
            evidence (List[Evidence]): List of evidence items to use
            graph_context (Optional[Dict[str, Any]]): Optional knowledge graph context
            
        Returns:
            GroundedAnswer: The generated answer with citations and metadata
            
        Raises:
            ValueError: If query or evidence is empty
            RuntimeError: If LLM generation fails
        """
        if not query:
            raise ValueError("Query cannot be empty")
            
        if not evidence:
            return GroundedAnswer(
                answer="I do not know.",
                citations=[],
                follow_up_questions=[],
                confidence=0.0
            )

        try:
            # Stage 1: Initial answer generation
            initial_prompt = self._create_initial_prompt(query, evidence)
            initial_answer = self.generator.generate(initial_prompt)

            if "I do not know" in initial_answer.lower():
                return GroundedAnswer(
                    answer="I do not know.",
                    citations=[],
                    follow_up_questions=[],
                    confidence=0.0
                )

            # Stage 2: Answer refinement
            refinement_prompt = self._create_refinement_prompt(initial_answer, query)
            refined_answer = self.generator.generate(refinement_prompt)

            # Apply knowledge graph refinement if available
            if self.refiner and graph_context:
                refined_answer = self.refiner.refine_answer(refined_answer, graph_context)

            # Stage 3: Answer condensation
            condensation_prompt = self._create_condensation_prompt(refined_answer, query)
            final_answer = self.generator.generate(condensation_prompt)

            # Extract citations
            citations = self._extract_citations(final_answer, evidence)

            # Generate follow-up questions
            followup_prompt = self._create_followup_prompt(final_answer, query)
            followup_response = self.generator.generate(followup_prompt)
            follow_up_questions = [q.strip() for q in followup_response.split('\n') if q.strip()]

            # Calculate confidence based on citation coverage
            confidence = len(citations) / len(evidence) if evidence else 0.0

            return GroundedAnswer(
                answer=final_answer,
                citations=citations,
                follow_up_questions=follow_up_questions[:3],  # Limit to 3 questions
                confidence=confidence
            )
        except ValueError as e:
            # Handle validation errors
            print(f"Validation error: {str(e)}")
            return GroundedAnswer(
                answer="Invalid input provided.",
                citations=[],
                follow_up_questions=[],
                confidence=0.0
            )
        except RuntimeError as e:
            # Handle LLM generation errors
            print(f"LLM generation error: {str(e)}")
            return GroundedAnswer(
                answer="Failed to generate answer due to LLM error.",
                citations=[],
                follow_up_questions=[],
                confidence=0.0
            )
        except Exception as e:
            # Handle unexpected errors
            print(f"Unexpected error: {str(e)}")
            return GroundedAnswer(
                answer="An unexpected error occurred while generating the answer.",
                citations=[],
                follow_up_questions=[],
                confidence=0.0
            ) 