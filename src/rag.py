from dotenv import load_dotenv
from db import Database
from document_loader import DocumentLoader
from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama,OllamaEmbeddings
from enum import Enum
from chain import *

class LLM(Enum):
    LLAMA3_8B = "llama3:8b"
    GEMMA3_1B = "gemma3:1b"

class EMBEDDING(Enum):
    NOMIX_EMBED_TEXT = "nomic-embed-text:latest"

class RagType(Enum):
    SIMPLE = "simple"
    MULTI_QUERY = "multi_query"
    RAG_FUSION = "rag_fusion"

RAG_TYPE_BUILDERS = {
    RagType.SIMPLE: simple_rag_chain,
    RagType.MULTI_QUERY: multi_query_chain,
    RagType.RAG_FUSION: rag_fusion_chain,
}


class Rag:
    def __init__(self):
        if not load_dotenv():
            print(".env file not found")

    
        self.llm = ChatOllama(model = LLM.LLAMA3_8B.value)
        self.embed = OllamaEmbeddings(model = EMBEDDING.NOMIX_EMBED_TEXT.value)
        self.document_loader = DocumentLoader()
        self.db = Database(self.embed)
        self.retriever = self.db.get_retriever()
        self.rag_type = RagType.SIMPLE

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
        # Placeholder
        builder = self.get_rag_type_builder()
        final_chain = builder(self.get_llm(), self.get_retriever())
        
        response = final_chain.invoke(query)
        answer = response["answer"]
        docs = response["docs"]

        return answer, docs


    def get_rag_type(self):
        return self.rag_type
    
    def get_rag_type_builder(self):
        return RAG_TYPE_BUILDERS[self.rag_type]
    
    def set_rag_type(self, rag_type: RagType):
        self.rag_type = rag_type
    
    def get_db(self):
        return self.db
    
    def get_retriever(self):
        return self.retriever
    
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

    
    
