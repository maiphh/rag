from langchain_ollama import ChatOllama,OllamaEmbeddings



class Settings:
    def __init__(self):
        self.llm = ChatOllama(model = "mistral")
        self.embed = OllamaEmbeddings(model = "dengcao/Qwen3-Embedding-0.6B:Q8_0")
    
    def get_llm(self):
        return self.llm
    
    def get_embed(self):
        return self.embed
    
    
settings = Settings()
