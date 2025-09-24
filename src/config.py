from langchain_ollama import ChatOllama,OllamaEmbeddings
from enum import Enum

class LLM(Enum):
    LLAMA3_8B = "llama3:8b"
    GEMMA3_1B = "gemma3:1b"

class EMBEDDING(Enum):
    NOMIX_EMBED_TEXT = "nomic-embed-text:latest"

class Settings:
    def __init__(self):
        self.llm = ChatOllama(model = LLM.LLAMA3_8B.value)
        self.embed = OllamaEmbeddings(model = EMBEDDING.NOMIX_EMBED_TEXT.value)

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
    
    def set_embbedding(self, embed: EMBEDDING):
        self.embed = OllamaEmbeddings(model = embed.value)
    

    
    
settings = Settings()
