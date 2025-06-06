from ingest.multimodal_processor import MultimodalProcessor
from kg.builder import KnowledgeGraphBuilder
from retrieval.multimodal_retriever import MultimodalRetriever
from generation.openai_llm import OpenAILLM
from pipeline import ModularRAGPipeline
from config import OPENAI_API_KEY

def create_pipeline(file_path: str, api_key: str = OPENAI_API_KEY):
    """
    Create a modular RAG pipeline with the specified components.
    
    Args:
        file_path: Path to the document to process
        api_key: OpenAI API key
        
    Returns:
        Configured ModularRAGPipeline instance
    """
    # Initialize components
    processor = MultimodalProcessor()
    kg_builder = KnowledgeGraphBuilder(None)
    
    # Create pipeline
    pipeline = ModularRAGPipeline(
        llm=OpenAILLM(api_key),
        retriever=None,  # Will be set after processing
        processor=processor,
        kg_builder=kg_builder
    )
    
    # Process document and get content and embeddings
    content, embeddings = pipeline.process_document(file_path)
    
    # Create retriever with processed content
    retriever = MultimodalRetriever(kg_builder.graph, content, embeddings)
    pipeline.retriever = retriever
    
    return pipeline

def do_rag_pipeline(query: str, file_path: str, content_types: list = None, api_key: str = OPENAI_API_KEY):
    """
    Process a query using the RAG pipeline.
    
    Args:
        query: The query string
        file_path: Path to the document
        content_types: List of content types to search in (e.g., ['text', 'image', 'table'])
        api_key: OpenAI API key
        
    Returns:
        Generated answer
    """
    # Create pipeline
    pipeline = create_pipeline(file_path, api_key)
    
    # Get query embedding
    query_embedding = pipeline.processor.text_model.encode([query])[0]
    
    # Process query
    return pipeline.answer_query(query, query_embedding, content_types=content_types)

if __name__ == '__main__':
    file_path = "./docs/SunDB_manual.pdf"
    query = "How does SunDB handle replication and what are default thread values?"
    answer = do_rag_pipeline(query, file_path)
    print("Answer:\n", answer)
