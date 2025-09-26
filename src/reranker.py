from typing import List, Sequence, Tuple, Optional, Dict, Any
from langchain_core.documents import Document
from langchain.retrievers.document_compressors import CrossEncoderReranker
import math


class CrossEncoderRerankerWithScores(CrossEncoderReranker):
    """
    Same behavior as CrossEncoderReranker, but also writes `rerank_score` into
    each returned Document's metadata and preserves existing metadata.
    """
    def compress_documents(self, documents: Sequence[Document], query: str, **kwargs) -> Sequence[Document]:
        # score all candidates at once (implementation may batch internally)
        pairs = [(query, d.page_content) for d in documents]
        scores: List[float] = self.model.score(pairs)  # 1 score per (query, doc) pair
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[: self.top_n]

        docs: List[Document] = []
        for doc, s in ranked:
            point = self._sigmoid(s)
            doc.metadata["rerank_score"] = float(s)
            doc.metadata["confidence"] = self._to_confidence(point)
            doc.metadata["confidence_label"] = "high" if point >= 0.80 else "medium" if point >= 0.60 else "low"
            
            
            docs.append(doc)
        return docs
    
    @staticmethod
    def _to_confidence(prob_like: float) -> float:
        # prob_like is sigmoid(logit) or a normalized 0..1 score
        return max(0.0, min(1.0, float(prob_like)))
    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))