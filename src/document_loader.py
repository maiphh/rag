from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pathlib import Path
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter

default_root = "data"

class DocumentLoader:
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        self.converter = DocumentConverter()
        # Supported file extensions
        self.supported_extensions = {
            '.pdf', '.docx', '.doc', '.pptx', '.ppt', 
            '.xlsx', '.xls', '.html', '.htm', '.md', '.txt'
        }



    def load_documents(self, root = default_root):
        loader = PyPDFDirectoryLoader(root)
        return loader.load()
    
    # def load_documents(self, root=default_root):
    #     """Load documents from directory using Docling"""
    #     documents = []
    #     root_path = Path(root)
        
    #     if not root_path.exists():
    #         raise FileNotFoundError(f"Directory {root} not found")
        
    #     # Find all supported files in the directory
    #     for file_path in root_path.rglob('*'):
    #         if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
    #             try:
    #                 # Convert document using Docling
    #                 result = self.converter.convert(str(file_path))
                    
    #                 documents.append(result)
    #                 print(f"✅ Loaded: {file_path.name}")
                    
    #             except Exception as e:
    #                 print(f"❌ Error loading {file_path.name}: {str(e)}")
    #                 continue
        
    #     print(f"📚 Total documents loaded: {len(documents)}")
    #     return documents

    def get_all_files(self, root=default_root):
        root_path = Path(root)
        all_files = [p for p in root_path.rglob('*') if p.is_file() and p.suffix.lower() in self.supported_extensions]
        return all_files
    
    def split_documents(self, documents):
        return self.splitter.split_documents(documents)
    

    

    
document_loader = DocumentLoader()