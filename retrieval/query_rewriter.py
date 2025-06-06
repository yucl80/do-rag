from transformers import pipeline

class QueryRewriter:
    def __init__(self):
        self.intent_detector = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
        self.rewriter = pipeline("text2text-generation", model="google/flan-t5-base")

    def rewrite(self, query, graph_context):
        # Handle multi-intent queries
        if " and " in query:
            sub_queries = query.split(" and ")
            rewritten_queries = []
            for sub_query in sub_queries:
                prompt = f"Disambiguate and clarify the user query using graph context.\nGraphContext: {graph_context}\nQuery: {sub_query}"
                result = self.rewriter(prompt, max_length=128)[0]['generated_text']
                rewritten_queries.append(result)
            return rewritten_queries
        
        # Single intent query
        prompt = f"Disambiguate and clarify the user query using graph context.\nGraphContext: {graph_context}\nQuery: {query}"
        result = self.rewriter(prompt, max_length=128)[0]['generated_text']
        return [result]
