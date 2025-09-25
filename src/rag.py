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

class Rag:
    def __init__(self):
        if not load_dotenv():
            print(".env file not found")

        # LLM + EMBEDDING
        self.llm = ChatOllama(model = LLM.GEMMA3_1B.value)
        self.embed = OllamaEmbeddings(model = EMBEDDING.NOMIX_EMBED_TEXT.value)
        self.rag_type = RagType.SIMPLE

        # DB
        self.document_loader = DocumentLoader(self.embed)
        self.db = Database(self.embed)

        # RETRIEVER
        self.retrieve_num = 20
        self.top_n = 4  # for reranker
        self.threshold = 0.5
        self.is_rerank = True
        self.reranker = RERANKER.MACRO_MINI.value

        # ROUTER
        self.domain_router = ManualDomainRouter(domain = DOMAIN.ALL.value)

        # INIT
        self.load_documents()

    def load_documents(self):
        loaded = self.db.get_loaded_src()
        docs = self.document_loader.load_documents(loaded_files=loaded)

        if docs:
            chunks = self.document_loader.split_documents(docs)
            self.db.add_to_db(chunks)
        return docs
    
    def clear_db(self):
        self.db.clear()

    def invoke(self, query):
        response = self.invoke_full(query)
        answer = response["answer"]
        docs = response["docs"]

        return answer, docs

    def invoke_full(self, query):
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
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

        rank_retriever = ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=reranker
        )
        return rank_retriever

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

    
    
