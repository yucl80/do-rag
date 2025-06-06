from sentence_transformers import SentenceTransformer
import pdfplumber

class DocumentChunker:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def extract_chunks(self, file_path, chunk_size=500):
        chunks = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i:i+chunk_size]
                        if len(chunk) > 50:
                            chunks.append(chunk)
        embeddings = self.model.encode(chunks)
        return chunks, embeddings
