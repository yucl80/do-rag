import unittest
import os
from ingest.multimodal_processor import MultimodalProcessor
#from ingest.chunker import DocumentChunker

class TestIngestModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 创建一个简单的 txt 文件用于测试
        cls.test_txt_path = 'test_sample.txt'
        with open(cls.test_txt_path, 'w', encoding='utf-8') as f:
            f.write('This is a test document.\nIt contains multiple lines.\nThis is the third line.')

    @classmethod
    def tearDownClass(cls):
        # 删除测试文件
        if os.path.exists(cls.test_txt_path):
            os.remove(cls.test_txt_path)

    def test_multimodal_processor_txt(self):
        processor = MultimodalProcessor()
        result = processor.process_document(self.test_txt_path)
        self.assertIn('text_chunks', result)
        self.assertGreater(len(result['text_chunks']), 0)
        self.assertIn('metadata', result)

    # def test_document_chunker_txt(self):
    #     chunker = DocumentChunker()
    #     chunks, embeddings = chunker.extract_chunks(self.test_txt_path, chunk_size=20)
    #     self.assertGreater(len(chunks), 0)
    #     self.assertEqual(len(chunks), len(embeddings))

if __name__ == '__main__':
    unittest.main() 