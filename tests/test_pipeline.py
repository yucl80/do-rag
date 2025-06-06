import unittest
import os
import numpy as np
from pathlib import Path
from pipeline import ModularRAGPipeline
from generation.openai_llm import OpenAILLM
from generation.generator import AnswerGenerator
from retrieval.multimodal_retriever import MultimodalRetriever
from ingest.multimodal_processor import MultimodalProcessor
from kg.builder import KnowledgeGraphBuilder
from dotenv import load_dotenv

class TestModularRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        # 确保加载环境变量
        load_dotenv()
        print("Current working directory:", os.getcwd())
        print("OPENAI_API_KEY from env:", os.getenv("OPENAI_API_KEY"))
        
        cls.test_doc_path = "./docs/SunDB_manual.pdf"  # 确保这个文件存在
        
        # 初始化组件
        cls.processor = MultimodalProcessor()
        cls.llm = OpenAILLM()  # 不再传递 API key
        cls.answer_generator = AnswerGenerator(cls.llm)  # 使用 AnswerGenerator 包装 llm
        cls.kg_builder = KnowledgeGraphBuilder(cls.answer_generator)  # 使用 answer_generator 初始化
        
        # 创建pipeline
        cls.pipeline = ModularRAGPipeline(
            llm=cls.llm,
            retriever=None,  # 将在处理文档后设置
            processor=cls.processor,
            kg_builder=cls.kg_builder
        )

    def test_document_processing(self):
        """测试文档处理功能"""
        content, embeddings = self.pipeline.process_document(self.test_doc_path)
        
        # 验证处理结果
        self.assertIsNotNone(content)
        self.assertIsNotNone(embeddings)
        self.assertIn('text_chunks', content)
        self.assertIsInstance(embeddings, dict)

    def test_knowledge_graph_building(self):
        """测试知识图谱构建功能"""
        content, _ = self.pipeline.process_document(self.test_doc_path)
        
        # 验证知识图谱构建
        self.assertIsNotNone(self.kg_builder.graph)
        self.assertTrue(len(self.kg_builder.graph.nodes) > 0)

    def test_retrieval(self):
        """测试检索功能"""
        content, embeddings = self.pipeline.process_document(self.test_doc_path)
        
        # 创建retriever
        retriever = MultimodalRetriever(self.kg_builder.graph, content, embeddings)
        self.pipeline.retriever = retriever
        
        # 测试查询
        query = "How does SunDB handle replication?"
        query_embedding = self.processor.text_model.encode([query])[0]
        
        relevant_content, graph_context = retriever.retrieve(
            query, query_embedding, content_types=['text']
        )
        
        self.assertIsNotNone(relevant_content)
        self.assertIsNotNone(graph_context)

    def test_query_answering(self):
        """测试问答功能"""
        content, embeddings = self.pipeline.process_document(self.test_doc_path)
        
        # 创建retriever
        retriever = MultimodalRetriever(self.kg_builder.graph, content, embeddings)
        self.pipeline.retriever = retriever
        
        # 测试查询
        query = "How does SunDB handle replication?"
        query_embedding = self.processor.text_model.encode([query])[0]
        
        answer = self.pipeline.answer_query(query, query_embedding)
        
        self.assertIsNotNone(answer)
        self.assertIsNotNone(answer.answer)
        self.assertIsInstance(answer.citations, list)

    def test_pipeline_info(self):
        """测试pipeline信息获取功能"""
        info = self.pipeline.get_pipeline_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('llm', info)
        self.assertIn('retriever', info)
        self.assertIn('has_processor', info)
        self.assertIn('has_kg_builder', info)
        
        # 验证 retriever 为 None 的情况
        self.assertIsNone(info['retriever'])

if __name__ == '__main__':
    unittest.main() 