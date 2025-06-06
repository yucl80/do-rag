import spacy
import networkx as nx
import json # Import the json library
from generation.generator import AnswerGenerator
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import re

class KnowledgeGraphBuilder:
    def __init__(self, llm_generator: AnswerGenerator):
        self.graph = nx.MultiDiGraph()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            raise RuntimeError(f"Failed to load spaCy model: {str(e)}")
        self.llm_generator = llm_generator
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize validation rules
        self.validation_rules = {
            'max_nodes': 10000,
            'max_edges_per_node': 100,
            'min_node_description_length': 3,
            'max_node_description_length': 500,
            'allowed_relation_types': {
                'CONTAINS', 'DESCRIBES', 'DEPENDS_ON', 
                'HAS_METRIC', 'RELATED_TO'
            }
        }

    def validate_graph_structure(self) -> Tuple[bool, List[str]]:
        """Validate the graph structure against defined rules."""
        errors = []
        
        # Check node count
        if len(self.graph.nodes) > self.validation_rules['max_nodes']:
            errors.append(f"Graph exceeds maximum node count: {len(self.graph.nodes)}")
        
        # Check node descriptions
        for node, data in self.graph.nodes(data=True):
            description = data.get('description', '')
            if not description:
                errors.append(f"Node {node} has no description")
            elif len(description) < self.validation_rules['min_node_description_length']:
                errors.append(f"Node {node} description too short: {description}")
            elif len(description) > self.validation_rules['max_node_description_length']:
                errors.append(f"Node {node} description too long: {description}")
        
        # Check edge counts and types
        for node in self.graph.nodes():
            edge_count = len(list(self.graph.edges(node)))
            if edge_count > self.validation_rules['max_edges_per_node']:
                errors.append(f"Node {node} has too many edges: {edge_count}")
            
            for _, _, edge_data in self.graph.edges(node, data=True):
                relation_type = edge_data.get('type')
                if relation_type not in self.validation_rules['allowed_relation_types']:
                    errors.append(f"Invalid relation type: {relation_type}")
        
        return len(errors) == 0, errors
    
    def clean_graph(self) -> None:
        """Clean the graph by removing invalid nodes and edges."""
        # Remove nodes with invalid descriptions
        nodes_to_remove = []
        for node, data in self.graph.nodes(data=True):
            description = data.get('description', '')
            if (not description or 
                len(description) < self.validation_rules['min_node_description_length'] or
                len(description) > self.validation_rules['max_node_description_length']):
                nodes_to_remove.append(node)
        
        self.graph.remove_nodes_from(nodes_to_remove)
        
        # Remove edges with invalid relation types
        edges_to_remove = []
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            if data.get('type') not in self.validation_rules['allowed_relation_types']:
                edges_to_remove.append((u, v, k))
        
        self.graph.remove_edges_from(edges_to_remove)
        
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated_nodes)
        
        # Remove duplicate edges
        edges_to_keep = set()
        edges_to_remove = []
        for u, v, k, data in self.graph.edges(data=True, keys=True):
            edge_key = (u, v, data.get('type'))
            if edge_key in edges_to_keep:
                edges_to_remove.append((u, v, k))
            else:
                edges_to_keep.add(edge_key)
        
        self.graph.remove_edges_from(edges_to_remove)
    
    def extract_high_level(self, text):
        # Simulate detection of structural elements (e.g., Section headings)
        lines = text.splitlines()
        high_level_keywords = {
            "Document Structure": ["document", "report", "manual", "guide"],
            "Section": ["section", "heading", "subsection"],
            "Chapter": ["chapter", "part"],
            "Paragraph": ["paragraph", "para"],
            "Sentence": ["sentence"]
        }
        for line in lines:
            if "section" in line.lower():
                # We will rely on LLM for primary extraction and dynamic layering now
                pass # self.graph.add_node(line.strip(), level="high", type="section")

        for entity_type, keywords in high_level_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                     # We will rely on LLM for primary extraction and dynamic layering now
                    pass # self.graph.add_node(keyword, level="high", type=entity_type)

    def extract_mid_level(self, text):
         # We will rely on LLM for primary extraction and dynamic layering now
         pass

    def extract_low_level(self, text):
         # We will rely on LLM for primary extraction and dynamic layering now
         pass

    def extract_covariates(self, text):
         # We will rely on LLM for primary extraction and dynamic layering now
         pass

    def extract_and_add_entities(self, text):
        # This method is now superseded by the LLM-based extraction for dynamic layering
        # self.extract_high_level(text)
        # self.extract_mid_level(text)
        # self.extract_low_level(text)
        # self.extract_covariates(text)
        pass

    def assign_level_by_description(self, entity_text, description, text_block_type):
        # Initial dynamic level assignment based on entity text and description keywords
        level = "unknown"

        # Prioritize level assignment based on text block type
        if text_block_type == "heading":
            level = "High"
        elif text_block_type in ["caption", "table_cell"]:
            level = "Covariate"
        elif text_block_type == "list_item":
             # List items can vary, start as Mid and let relations refine
             level = "Mid"
        else:
            # Fallback to rules based on entity text and description for paragraph/other types
            # Rules based on entity text (for structural elements) - still relevant for initial high-level structural items
            if any(structural_keyword in entity_text.lower() for structural_keyword in ["section", "chapter", "paragraph", "sentence"]):
                level = "High"
            # Rules based on description keywords for other levels
            elif any(low_level_keyword in description.lower() for low_level_keyword in ["thread", "memory", "process", "metric", "network", "execution", "runtime"]):
                 level = "Low"
            elif any(covariate_keyword in description.lower() for covariate_keyword in ["parameter", "config", "setting", "dependency", "status", "behaviour", "specification"]):
                 level = "Covariate"
            else:
                # Default to Mid for other entities if not classified by description
                level = "Mid"

        return level

    def refine_level_by_relations(self) -> None:
        """Refine levels based on entity relationships with optimized traversal."""
        level_hierarchy = {"High": 4, "Mid": 3, "Low": 2, "Covariate": 1, "unknown": 0}
        max_hops = 2
        
        # Use defaultdict for better performance
        level_suggestions_weighted = defaultdict(lambda: {level: 0.0 for level in level_hierarchy.keys()})
        
        # Pre-compute node properties for faster access
        node_properties = {
            node: {
                'type': self.graph.nodes[node].get('text_block_type', 'other'),
                'current_level': level_hierarchy.get(self.graph.nodes[node].get('level', 'unknown'), 0)
            }
            for node in self.graph.nodes()
        }
        
        # Define relationship influence and weights
        relation_level_influence = {
            "CONTAINS": ("High", "Low"),
            "DESCRIBES": ("Mid", "Low"),
            "DEPENDS_ON": ("Low", "Covariate"),
            "HAS_METRIC": ("Low", "Covariate"),
            "RELATED_TO": (None, None)
        }
        
        relation_weights = {
            "CONTAINS": 2.5,
            "DESCRIBES": 1.8,
            "DEPENDS_ON": 2.0,
            "HAS_METRIC": 2.0,
            "RELATED_TO": 0.8
        }
        
        hop_decay_factor = 0.5
        
        # Optimized BFS with early termination
        for start_node in self.graph.nodes():
            visited = {start_node}
            queue = [(start_node, 0, [start_node])]
            
            while queue:
                current_node, hops, path = queue.pop(0)
                
                if hops >= max_hops:
                    continue
                
                # Process both outgoing and incoming edges in one pass
                for neighbor in set(self.graph.neighbors(current_node)) | set(self.graph.predecessors(current_node)):
                    if neighbor in visited:
                        continue
                        
                    visited.add(neighbor)
                    
                    # Get all edges between current_node and neighbor
                    edges = list(self.graph.edges(current_node, neighbor, data=True, keys=True))
                    edges.extend(self.graph.edges(neighbor, current_node, data=True, keys=True))
                    
                    for u, v, k, edge_data in edges:
                        relation_type = edge_data.get('type')
                        if relation_type not in relation_level_influence:
                            continue
                            
                        # Calculate weight with optimizations
                        weight = relation_weights.get(relation_type, 0.8)
                        weight *= (hop_decay_factor ** (hops + 1))
                        weight *= (hop_decay_factor ** len(path))
                        
                        # Adjust weight based on node types
                        source_type = node_properties[u]['type']
                        target_type = node_properties[v]['type']
                        
                        if source_type == 'heading' and target_type != 'heading':
                            weight *= 1.2
                        elif source_type != 'heading' and target_type == 'heading':
                            weight *= 1.2
                        elif source_type in ['caption', 'table_cell'] or target_type in ['caption', 'table_cell']:
                            weight *= 0.9
                        
                        # Apply suggestions
                        source_suggestion, target_suggestion = relation_level_influence[relation_type]
                        if source_suggestion:
                            level_suggestions_weighted[u][source_suggestion] += weight
                        if target_suggestion:
                            level_suggestions_weighted[v][target_suggestion] += weight
                    
                    queue.append((neighbor, hops + 1, path + [neighbor]))
        
        # Update node levels based on weighted suggestions
        for node, suggestions in level_suggestions_weighted.items():
            if not suggestions:
                continue
                
            # Get the level with highest weight
            max_level = max(suggestions.items(), key=lambda x: x[1])[0]
            current_level = self.graph.nodes[node].get('level', 'unknown')
            
            # Only update if the new level is different and has sufficient weight
            if (max_level != current_level and 
                suggestions[max_level] > 1.0):  # Threshold for level change
                self.graph.nodes[node]['level'] = max_level

    def _extract_json_from_response(self, response: str) -> dict:
        """Extract JSON object from LLM response.
        
        Args:
            response: The raw response from LLM
            
        Returns:
            dict: The extracted JSON object
            
        Raises:
            RuntimeError: If no valid JSON object is found
        """
        # Try to find JSON object between ```json and ``` markers
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
                
        # If no JSON markers found, try to find any JSON object
        try:
            # Find the first { and last } in the text
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
            
        raise RuntimeError("No valid JSON object found in LLM response")

    def extract_entities_and_relations_with_llm(self, text: str) -> None:
        """Extract entities and relations using LLM with error handling."""
        try:
            # Define the prompt for the LLM
            prompt = f"""Extract entities and their relationships from the following text.
            For each entity, provide:
            1. Entity name
            2. Description
            3. Level (High/Mid/Low/Covariate)
            4. Text block type (heading/paragraph/list_item/caption/table_cell/other)

            For each relationship, provide:
            1. Source entity
            2. Target entity
            3. Relationship type (CONTAINS/DESCRIBES/DEPENDS_ON/HAS_METRIC/RELATED_TO)

            Text:
            {text}

            Format the response as a JSON object with 'entities' and 'relations' arrays."""

            # Get LLM response
            response = self.llm_generator.generate(prompt, skip_quality_check=True)

            try:
                # Extract and parse JSON response
                data = self._extract_json_from_response(response)

                # Process entities
                for entity in data.get('entities', []):
                    try:
                        self.graph.add_node(
                            entity['name'],
                            description=entity['description'],
                            level=entity['level'],
                            text_block_type=entity['text_block_type']
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to add entity {entity.get('name')}: {str(e)}")

                # Process relations
                for relation in data.get('relations', []):
                    try:
                        if (relation['source'] in self.graph and
                            relation['target'] in self.graph and
                            relation['type'] in self.validation_rules['allowed_relation_types']):
                            self.graph.add_edge(
                                relation['source'],
                                relation['target'],
                                type=relation['type']
                            )
                    except Exception as e:
                        self.logger.warning(f"Failed to add relation {relation}: {str(e)}")

                # Validate and clean graph
                is_valid, errors = self.validate_graph_structure()
                if not is_valid:
                    self.logger.warning(f"Graph validation errors: {errors}")
                    self.clean_graph()

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
                raise RuntimeError("Invalid LLM response format")

        except Exception as e:
            self.logger.error(f"Failed to extract entities and relations: {str(e)}")
            raise RuntimeError(f"Entity extraction failed: {str(e)}")
