from langchain_ollama import ChatOllama,OllamaEmbeddings



class Settings:
    def __init__(self):
        self.llm = ChatOllama(model = "gemma3n:e4b")
        self.embed = OllamaEmbeddings(model = "nomic-embed-text:latest")
    
    def get_llm(self):
        return self.llm
    
    def get_embed(self):
        return self.embed
    
    
settings = Settings()
