from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pathlib import Path
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from langchain_core.document_loaders import BaseLoader
from typing import Iterable, List, Optional

default_root = "data"

class DoclingLoader(BaseLoader):
    def __init__(self, path: str | list[str]):
        self._file_paths = path if isinstance(path,list) else [path]
        self._converter = DocumentConverter()
    
    def lazy_load(self):
        for path in self._file_paths:
            docling_doc = self._converter.convert(path).document
            text = docling_doc.export_to_markdown()
            yield Document(page_content=text, metadata={"source": str(path)})

class DocumentLoader:
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

        self.converter = DocumentConverter()
        # Supported file extensions
        self.supported_extensions = {
            '.pdf', '.docx', '.doc', '.pptx', '.ppt', 
            '.xlsx', '.xls', '.html', '.htm', '.md', '.txt'
        }



    def load_documents(self, root = default_root, loaded_files : list[str] = None):
        files = self.get_all_files(root)
        unloaded_files = [f for f in files if f not in loaded_files]
        
        if not unloaded_files:
            print("No new files to load.")
            return []
        
        # loader = PyPDFDirectoryLoader(root)
        loader = DoclingLoader(unloaded_files)
        return loader.load()

    def get_all_files(self, root=default_root) -> list[str]:
        root_path = Path(root)
        all_files = [str(p) for p in root_path.rglob('*') if p.is_file() and p.suffix.lower() in self.supported_extensions]
        return all_files
    
    def split_documents(self, documents):
        return self.splitter.split_documents(documents)
    

