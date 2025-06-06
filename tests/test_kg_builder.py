import unittest
from kg.builder import KnowledgeGraphBuilder
from generation.generator import AnswerGenerator
import networkx as nx

class MockLLM:
    def __init__(self):
        pass
    
    def generate(self, prompt):
        return "Mock response"

class TestKnowledgeGraphBuilder(unittest.TestCase):
    def setUp(self):
        # 创建一个模拟的 LLM 和 AnswerGenerator
        self.mock_llm = MockLLM()
        self.mock_generator = AnswerGenerator(self.mock_llm)
        self.builder = KnowledgeGraphBuilder(self.mock_generator)

    def test_initialization(self):
        """测试知识图谱构建器的初始化"""
        self.assertIsInstance(self.builder.graph, nx.MultiDiGraph)
        self.assertEqual(len(self.builder.graph.nodes), 0)
        self.assertEqual(len(self.builder.graph.edges), 0)

    def test_extract_entities_and_relations(self):
        """测试实体和关系的提取"""
        test_text = """
        Section 1: System Overview
        The system consists of multiple components.
        Component A depends on Component B.
        Component B has a performance metric of 100ms.
        """
        
        # 模拟 LLM 生成器的响应
        self.mock_generator.generate = lambda prompt, skip_quality_check: '''
        ```json
        {
            "entities": [
                {
                    "name": "System Overview",
                    "description": "Main system description section",
                    "level": "High",
                    "text_block_type": "heading"
                },
                {
                    "name": "Component A",
                    "description": "A system component",
                    "level": "Mid",
                    "text_block_type": "paragraph"
                },
                {
                    "name": "Component B",
                    "description": "Another system component",
                    "level": "Mid",
                    "text_block_type": "paragraph"
                }
            ],
            "relations": [
                {
                    "source": "System Overview",
                    "target": "Component A",
                    "type": "CONTAINS"
                },
                {
                    "source": "Component A",
                    "target": "Component B",
                    "type": "DEPENDS_ON"
                },
                {
                    "source": "Component B",
                    "target": "Performance Metric",
                    "type": "HAS_METRIC"
                }
            ]
        }
        ```
        '''
        
        # 执行实体和关系提取
        self.builder.extract_entities_and_relations_with_llm(test_text)
        
        # 验证结果
        self.assertGreater(len(self.builder.graph.nodes), 0)
        self.assertGreater(len(self.builder.graph.edges), 0)
        
        # 验证节点属性
        for node in self.builder.graph.nodes(data=True):
            self.assertIn('description', node[1])
            self.assertIn('level', node[1])
            self.assertIn('text_block_type', node[1])
        
        # 验证边属性
        for edge in self.builder.graph.edges(data=True):
            self.assertIn('type', edge[2])
            self.assertIn(edge[2]['type'], self.builder.validation_rules['allowed_relation_types'])

    def test_graph_validation(self):
        """测试图结构验证"""
        # 添加一些测试节点和边
        self.builder.graph.add_node("Test Node 1", description="Valid description", level="High", text_block_type="heading")
        self.builder.graph.add_node("Test Node 2", description="Another valid description", level="Mid", text_block_type="paragraph")
        self.builder.graph.add_edge("Test Node 1", "Test Node 2", type="CONTAINS")
        
        # 验证图结构
        is_valid, errors = self.builder.validate_graph_structure()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

if __name__ == '__main__':
    unittest.main() 