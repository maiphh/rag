from db import chromaDb
from langsmith.run_helpers import traceable

class Retriever:
    def __init__(self):
        self.db = chromaDb

    @traceable(name="retrieve")
    def retrieve(self, query, top_k=3, threshold=0.3):
        results = self.db.db.similarity_search(
            query,
            k=top_k,
            filter=None,
        )
        return results
    
retriever = Retriever()