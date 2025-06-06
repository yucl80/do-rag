import unittest
import numpy as np
import networkx as nx
from retrieval.multimodal_retriever import MultimodalRetriever
from retrieval.unified_retriever import UnifiedRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.config import DEFAULT_MODEL_CONFIG
import spacy

class TestRetrieval(unittest.TestCase):
    def setUp(self):
        """Set up test data and models"""
        # 创建测试用的知识图谱
        self.graph = nx.MultiDiGraph()
        
        # 添加一些测试节点
        self.graph.add_node("database", description="A database system", level="High")
        self.graph.add_node("replication", description="Data replication mechanism", level="Mid")
        self.graph.add_node("performance", description="System performance metrics", level="Mid")
        
        # 添加一些边
        self.graph.add_edge("database", "replication", type="CONTAINS", weight=2.5)
        self.graph.add_edge("database", "performance", type="HAS_METRIC", weight=2.0)
        
        # 创建测试用的文本块
        self.chunks = [
            "The database system uses a master-slave replication model.",
            "Replication ensures data consistency across multiple nodes.",
            "Performance metrics include latency and throughput.",
            "The system supports both synchronous and asynchronous replication."
        ]
 
        # 加载spaCy模型用于生成测试用的embeddings
        self.nlp = spacy.load(DEFAULT_MODEL_CONFIG["spacy_model"])
        
        # 生成测试用的embeddings
        self.embeddings = np.array([self.nlp(chunk).vector for chunk in self.chunks])
        
        # 创建测试用的内容字典
        self.content = {
            'text': self.chunks,
            'images': [],  # 测试中暂不使用图像
            'tables': []   # 测试中暂不使用表格
        }

    def test_multimodal_retriever(self):
        """测试多模态检索器"""
        retriever = MultimodalRetriever(self.graph, self.content, {'text': self.embeddings})
        
        # 测试查询
        query = "How does the database handle replication?"
        query_embedding = self.nlp(query).vector
        
        # 执行检索
        relevant_content, graph_context = retriever.retrieve(
            query, query_embedding, content_types=['text']
        )
        
        # 验证结果
        self.assertIsNotNone(relevant_content)
        self.assertIsNotNone(graph_context)
        self.assertIn('text', relevant_content)
        self.assertTrue(len(relevant_content['text']) > 0)
        
        # 验证检索器信息
        info = retriever.get_retriever_info()
        self.assertEqual(info['type'], 'multimodal')
        self.assertTrue(info['has_graph'])

    def test_unified_retriever(self):
        """测试统一检索器"""
        retriever = UnifiedRetriever(self.graph, self.chunks, self.embeddings)
        
        # 测试查询
        query = "What are the performance aspects of replication?"
        query_embedding = self.nlp(query).vector
        
        # 执行检索
        relevant_chunks, graph_context = retriever.retrieve(query, query_embedding)
        
        # 验证结果
        self.assertIsNotNone(relevant_chunks)
        self.assertIsNotNone(graph_context)
        self.assertTrue(len(relevant_chunks) > 0)
        
        # 验证检索器信息
        info = retriever.get_retriever_info()
        self.assertEqual(info['type'], 'unified')
        self.assertTrue(info['has_graph'])

    def test_hybrid_retriever(self):
        """测试混合检索器"""
        retriever = HybridRetriever(
            graph=self.graph,
            chunks=self.chunks,
            embeddings=self.embeddings
        )
        
        # 测试查询
        query = "Explain the replication mechanism and its performance implications"
        query_embedding = self.nlp(query).vector
        
        # 执行检索
        relevant_chunks, graph_context = retriever.retrieve(query, query_embedding)
        
        # 验证结果
        self.assertIsNotNone(relevant_chunks)
        self.assertIsNotNone(graph_context)
        self.assertTrue(len(relevant_chunks) > 0)
        
        # 验证检索器信息
        info = retriever.get_retriever_info()
        self.assertEqual(info['type'], 'hybrid')
        self.assertTrue(info['has_graph'])

if __name__ == '__main__':
    unittest.main() 