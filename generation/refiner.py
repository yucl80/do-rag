import networkx as nx
from typing import List, Dict, Tuple, Set
import spacy
import re
from dataclasses import dataclass
from enum import Enum

class VerificationStatus(Enum):
    VERIFIED = "verified"
    INCONSISTENT = "inconsistent"
    UNVERIFIABLE = "unverifiable"

@dataclass
class FactCheck:
    statement: str
    status: VerificationStatus
    evidence: List[str]
    confidence: float
    suggested_correction: str = ""

class AnswerRefiner:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define patterns for fact extraction
        self.fact_patterns = [
            r"([^.]*?(?:is|are|has|have|can|will|should|must|does|do)[^.]*\.)",  # Statements with common verbs
            r"([^.]*?(?:default|value|parameter|setting|configuration)[^.]*\.)",  # Technical specifications
            r"([^.]*?(?:requires|needs|depends on|relies on)[^.]*\.)",  # Dependencies
            r"([^.]*?(?:performs|executes|handles|manages)[^.]*\.)",  # Actions/behaviors
        ]
        
        # Define relationship verification rules
        self.verification_rules = {
            "CONTAINS": lambda source, target: self._verify_contains(source, target),
            "DESCRIBES": lambda source, target: self._verify_describes(source, target),
            "DEPENDS_ON": lambda source, target: self._verify_depends_on(source, target),
            "HAS_METRIC": lambda source, target: self._verify_has_metric(source, target),
            "RELATED_TO": lambda source, target: self._verify_related_to(source, target)
        }

    def _extract_facts(self, text: str) -> List[str]:
        """Extract potential facts from text using pattern matching."""
        facts = []
        for pattern in self.fact_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            facts.extend(match.group(1).strip() for match in matches)
        return facts

    def _get_relevant_nodes(self, fact: str) -> List[str]:
        """Find relevant nodes in the graph for fact verification."""
        fact_doc = self.nlp(fact)
        relevant_nodes = []
        
        for node in self.graph.nodes():
            node_doc = self.nlp(node)
            if node_doc.similarity(fact_doc) > 0.3:  # Semantic similarity threshold
                relevant_nodes.append(node)
                
            # Also check node descriptions
            description = self.graph.nodes[node].get("description", "")
            if description:
                desc_doc = self.nlp(description)
                if desc_doc.similarity(fact_doc) > 0.3:
                    relevant_nodes.append(node)
        
        return relevant_nodes

    def _verify_contains(self, source: str, target: str) -> Tuple[bool, float, List[str]]:
        """Verify if source contains target based on graph structure."""
        evidence = []
        confidence = 0.0
        
        # Check direct containment
        if self.graph.has_edge(source, target):
            for edge_key in self.graph[source][target]:
                edge_data = self.graph[source][target][edge_key]
                if edge_data.get("type") == "CONTAINS":
                    evidence.append(f"Direct containment relationship found")
                    confidence = 0.9
                    return True, confidence, evidence
        
        # Check transitive containment
        for path in nx.all_simple_paths(self.graph, source, target, cutoff=2):
            if len(path) > 1:
                evidence.append(f"Transitive containment through: {' -> '.join(path)}")
                confidence = 0.7
                return True, confidence, evidence
        
        return False, 0.0, []

    def _verify_describes(self, source: str, target: str) -> Tuple[bool, float, List[str]]:
        """Verify if source describes target based on graph structure."""
        evidence = []
        confidence = 0.0
        
        if self.graph.has_edge(source, target):
            for edge_key in self.graph[source][target]:
                edge_data = self.graph[source][target][edge_key]
                if edge_data.get("type") == "DESCRIBES":
                    evidence.append(f"Direct description relationship found")
                    confidence = 0.9
                    return True, confidence, evidence
        
        # Check if source's description contains target
        source_desc = self.graph.nodes[source].get("description", "")
        if source_desc and target.lower() in source_desc.lower():
            evidence.append(f"Target found in source's description")
            confidence = 0.8
            return True, confidence, evidence
        
        return False, 0.0, []

    def _verify_depends_on(self, source: str, target: str) -> Tuple[bool, float, List[str]]:
        """Verify if source depends on target based on graph structure."""
        evidence = []
        confidence = 0.0
        
        if self.graph.has_edge(source, target):
            for edge_key in self.graph[source][target]:
                edge_data = self.graph[source][target][edge_key]
                if edge_data.get("type") == "DEPENDS_ON":
                    evidence.append(f"Direct dependency relationship found")
                    confidence = 0.9
                    return True, confidence, evidence
        
        return False, 0.0, []

    def _verify_has_metric(self, source: str, target: str) -> Tuple[bool, float, List[str]]:
        """Verify if source has metric target based on graph structure."""
        evidence = []
        confidence = 0.0
        
        if self.graph.has_edge(source, target):
            for edge_key in self.graph[source][target]:
                edge_data = self.graph[source][target][edge_key]
                if edge_data.get("type") == "HAS_METRIC":
                    evidence.append(f"Direct metric relationship found")
                    confidence = 0.9
                    return True, confidence, evidence
        
        return False, 0.0, []

    def _verify_related_to(self, source: str, target: str) -> Tuple[bool, float, List[str]]:
        """Verify if source is related to target based on graph structure."""
        evidence = []
        confidence = 0.0
        
        if self.graph.has_edge(source, target):
            for edge_key in self.graph[source][target]:
                edge_data = self.graph[source][target][edge_key]
                if edge_data.get("type") == "RELATED_TO":
                    evidence.append(f"Direct relationship found")
                    confidence = 0.8
                    return True, confidence, evidence
        
        # Check for indirect relationships
        if nx.has_path(self.graph, source, target):
            path = nx.shortest_path(self.graph, source, target)
            evidence.append(f"Indirect relationship through: {' -> '.join(path)}")
            confidence = 0.6
            return True, confidence, evidence
        
        return False, 0.0, []

    def _verify_fact(self, fact: str) -> FactCheck:
        """Verify a single fact against the knowledge graph."""
        relevant_nodes = self._get_relevant_nodes(fact)
        if not relevant_nodes:
            return FactCheck(fact, VerificationStatus.UNVERIFIABLE, [], 0.0)
        
        fact_doc = self.nlp(fact)
        evidence = []
        max_confidence = 0.0
        suggested_correction = ""
        
        # Check relationships between relevant nodes
        for i, source in enumerate(relevant_nodes):
            for target in relevant_nodes[i+1:]:
                for rel_type, verify_func in self.verification_rules.items():
                    is_valid, confidence, rel_evidence = verify_func(source, target)
                    if is_valid and confidence > max_confidence:
                        max_confidence = confidence
                        evidence.extend(rel_evidence)
        
        # Determine verification status
        if max_confidence >= 0.7:
            status = VerificationStatus.VERIFIED
        elif max_confidence >= 0.4:
            status = VerificationStatus.INCONSISTENT
            # Generate correction suggestion based on graph structure
            suggested_correction = self._generate_correction(fact, relevant_nodes)
        else:
            status = VerificationStatus.UNVERIFIABLE
        
        return FactCheck(fact, status, evidence, max_confidence, suggested_correction)

    def _generate_correction(self, fact: str, relevant_nodes: List[str]) -> str:
        """Generate a correction suggestion based on graph structure."""
        # Extract key entities from the fact
        fact_doc = self.nlp(fact)
        entities = [ent.text for ent in fact_doc.ents]
        
        if not entities:
            return ""
        
        # Find the most relevant node for each entity
        corrections = []
        for entity in entities:
            best_node = None
            best_sim = 0.0
            
            for node in relevant_nodes:
                node_doc = self.nlp(node)
                sim = node_doc.similarity(self.nlp(entity))
                if sim > best_sim:
                    best_sim = sim
                    best_node = node
            
            if best_node and best_sim > 0.5:
                corrections.append(f"Consider using '{best_node}' instead of '{entity}'")
        
        return " | ".join(corrections) if corrections else ""

    def refine_answer(self, answer: str, max_iterations: int = 3) -> str:
        """Refine the answer through iterative verification and correction."""
        current_answer = answer
        iteration = 0
        
        while iteration < max_iterations:
            # Extract facts from current answer
            facts = self._extract_facts(current_answer)
            if not facts:
                break
            
            # Verify each fact
            fact_checks = [self._verify_fact(fact) for fact in facts]
            
            # Check if any facts need correction
            needs_correction = any(
                check.status == VerificationStatus.INCONSISTENT 
                for check in fact_checks
            )
            
            if not needs_correction:
                break
            
            # Generate correction prompt
            correction_prompt = self._generate_correction_prompt(current_answer, fact_checks)
            
            # Get corrected answer from LLM
            # Note: This would need to be integrated with your LLM generator
            corrected_answer = self._get_corrected_answer(correction_prompt)
            
            if corrected_answer == current_answer:
                break
                
            current_answer = corrected_answer
            iteration += 1
        
        return current_answer

    def _generate_correction_prompt(self, answer: str, fact_checks: List[FactCheck]) -> str:
        """Generate a prompt for correcting the answer."""
        inconsistent_facts = [
            check for check in fact_checks 
            if check.status == VerificationStatus.INCONSISTENT
        ]
        
        prompt = f"""Please correct the following answer based on the knowledge graph verification results.

Original Answer:
{answer}

Inconsistent Facts and Suggested Corrections:
"""
        
        for check in inconsistent_facts:
            prompt += f"\n- Fact: {check.statement}"
            prompt += f"\n  Evidence: {', '.join(check.evidence)}"
            prompt += f"\n  Suggested Correction: {check.suggested_correction}"
        
        prompt += "\n\nPlease provide a corrected version of the answer that addresses these inconsistencies while maintaining the overall structure and flow of the original answer."
        
        return prompt

    def _get_corrected_answer(self, correction_prompt: str) -> str:
        """Get corrected answer from LLM.
        This method should be implemented to use your LLM generator.
        """
        # This is a placeholder - you'll need to integrate with your LLM generator
        # For now, we'll return the original answer
        return correction_prompt 