from dotenv import load_dotenv
from db import Database
from document_loader import DocumentLoader
from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama,OllamaEmbeddings
from enum import Enum
from chain import *
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from enum_manager import *
from router import ManualDomainRouter
from retriever import BasicRetriever, DomainRetriever
from reranker import CrossEncoderRerankerWithScores
from pathlib import Path
import pandas as pd
from typing import List, Dict


class Rag:
    def __init__(self):
        if not load_dotenv():
            print(".env file not found")

        # LLM + EMBEDDING
        self.llm = ChatOllama(model = LLM.LLAMA3_2.value)
        self.embed = OllamaEmbeddings(model = EMBEDDING.MX_BAI.value)
        self.rag_type = RagType.SIMPLE

        # DB
        self.cache_dir = "data/cache"
        self.document_loader = DocumentLoader(embed = self.embed, cache_dir=self.cache_dir)

        self.db_dir = "db"
        self.db = Database(self.embed, self.db_dir, self.cache_dir)

        # RETRIEVER
        self.retrieve_num = 20
        self.top_n = 5  # for reranker
        self.threshold = 0
        self.is_rerank = True
        self.reranker = RERANKER.MACRO_MINI.value

        # ROUTER
        self.domain_router = ManualDomainRouter(domain = DOMAIN.ALL.value)

        # INIT
        self.load_cached_documents()

    def load_cached_documents(self):
        self.db.load_cached_docs()


    def load_documents(self):
        loaded = self.get_loaded_src()
        cached = self.db.get_cached_src()
        processed = set(loaded + cached)

        self.document_loader.load_documents(loaded_files=[Path(file).stem for file in processed])

        # if docs:
        #     chunks = self.db.split_documents(docs)
        #     self.db.add(chunks)
        # return docs
        return self.load_cached_documents()
    
    # def load_documents(self):
    #     loaded = self.get_loaded_src()
    #     docs = self.document_loader.load_documents(loaded_files=loaded)
    #     if docs:
    #         chunks = self.db.split_documents(docs)
    #         self.db.add(chunks)
    #     return docs

    
    def get_loaded_src(self):
        return self.db.get_loaded_src()
    
    def clear_db(self):
        self.db.clear()
        self.db = Database(self.embed, self.db_dir, self.cache_dir)

    def invoke_simplify(self, query):
        response = self.invoke(query)
        answer = response["answer"]
        docs = response["docs"]

        return answer, docs

    def invoke(self, query):
        builder = self.get_rag_type_builder()
        final_chain = builder(self.get_llm(), self.get_retriever())
        
        response = final_chain.invoke(query)
        return response

    def get_retriever(self):
        basic_retriever = self.get_basic_retriever(self.retrieve_num, self.threshold)
        if self.is_rerank:
            return self.get_rank_retriever(basic_retriever, model_name=self.reranker, top_n=self.top_n)
        return basic_retriever

    def get_basic_retriever(self, k=20, threshold=0.5):
        return DomainRetriever(db=self.db, router=self.domain_router, k=k, threshold=threshold)

    def get_rank_retriever(self, retriever,  model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5): 
        cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
        reranker = CrossEncoderRerankerWithScores(model=cross_encoder, top_n=top_n)

        rank_retriever = ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=reranker
        )
        return rank_retriever
    
    def clear_cache(self):
        self.db.clear_cache()

    def set_domain(self, domain: DOMAIN):
        self.domain_router.set_domain(domain)

    def set_top_n(self,n):
        self.top_n = n
    
    def set_retrieve_num(self,n):
        self.retrieve_num = n

    def set_threshold(self, t):
        self.threshold = t

    def get_top_n(self):
        return self.top_n
    
    def get_retrieve_num(self):
        return self.retrieve_num
    
    def get_threshold(self):
        return self.threshold
    
    
    def get_rag_type(self):
        return self.rag_type
    
    def get_rag_type_builder(self):
        return RAG_TYPE_BUILDERS[self.rag_type]
    
    def set_rag_type(self, rag_type: RagType):
        self.rag_type = rag_type
    
    def get_db(self):
        return self.db
    
    
    def get_llm(self):
        return self.llm
    
    def get_embed(self):
        return self.embed
    
    def list_llms(self):
        return [e.value for e in LLM]
    
    def list_embeddings(self):
        return [e.value for e in EMBEDDING]
    
    def set_llm(self, llm: LLM):
        self.llm = ChatOllama(model = llm.value)
    
    def set_embedding(self, embed: EMBEDDING):
        self.embed = OllamaEmbeddings(model = embed.value)

    def to_string(self):
        return f"RAG Type: {self.rag_type.name}, LLM: {self.llm.model}, Embedding: {self.embed.model}"

    
    def evaluate_retrieval_performance(self) -> Dict:
        """Evaluate retrieval performance using test queries"""
        from evaluator import RetrievalEvaluator
        evaluator = RetrievalEvaluator(self)
        return evaluator.evaluate()

    def benchmark_retrieval_settings(self, test_queries: List[Dict]) -> pd.DataFrame:
        """Benchmark different retrieval settings"""
        from evaluator import RetrievalEvaluator
        evaluator = RetrievalEvaluator(self)
        return evaluator.benchmark_different_settings(test_queries)
