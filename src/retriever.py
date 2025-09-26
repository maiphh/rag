from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from router import Router
from db import Database
from enum_manager import *


class DomainRetriever(BaseRetriever):
    db: Database
    router: Router
    k: int = 20
    threshold: float = 0.5

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        domains = self.router.route()  # 👈 user-selected domains

        if DOMAIN.ALL.value in domains:
            retriever = self.db.db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":self.k, "score_threshold": self.threshold})
            return retriever.invoke(query)
        
        results: List[Document] = []
        for domain in domains:
            print(f"Searching in domain: {domain}")
            retriever = self.db.db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":self.k, "score_threshold": self.threshold, "filter":{"domain": domain}})
            results.extend(retriever.invoke(query))

        # dedup
        seen, dedup = set(), []
        for doc in results:
            key = doc.metadata.get("id")
            if key not in seen:
                seen.add(key)
                dedup.append(doc)
        return dedup
    

class BasicRetriever(BaseRetriever):
    db: Database
    k: int = 20
    threshold: float = 0.5

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        retriever = self.db.db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":self.k, "score_threshold": self.threshold})
        return retriever.invoke(query)