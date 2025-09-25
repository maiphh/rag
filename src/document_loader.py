from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from chunking_evaluation.chunking import KamradtModifiedChunker

from langchain_chroma import Chroma
from pathlib import Path
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter
from langchain_core.document_loaders import BaseLoader
from typing import Iterable, List, Optional
from enum_manager import *
default_root = "data"

class DoclingLoader(BaseLoader):
    def __init__(self, path: str | list[str]):
        self._file_paths = path if isinstance(path,list) else [path]
        self._converter = DocumentConverter()
    
    def lazy_load(self):
        for path in self._file_paths:
            docling_doc = self._converter.convert(path).document
            text = docling_doc.export_to_markdown()
            domain = self.get_domain_from_path(path)
            yield Document(page_content=text, metadata={"source": str(path), "domain": domain})

    def get_domain_from_path(self,path: str) -> str | None:
        p = Path(path).resolve()
        domain_map = {d.value.lower(): d for d in DOMAIN}

        for parent in p.parents:
            name = parent.name.lower()
            if name in domain_map:
                return domain_map[name].value  # or domain_map[name].value
        return DOMAIN.ALL.value

        

class DocumentLoader:
    
    def __init__(self, embed):
        
        self.embed = embed
        # self.splitter = SemanticChunker(embed)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
        # self.splitter = KamradtModifiedChunker(avg_chunk_size=400, min_chunk_size=50, embedding_function= self.embed)

        self.converter = DocumentConverter()
        # Supported file extensions
        self.supported_extensions = {
            '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.html', '.htm', '.md', '.txt'
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
        if not root_path.exists():
            raise FileNotFoundError(f"❌ The directory '{root}' does not exist.")

        files = []
        for path in root_path.rglob("*"):  # rglob('*') = recursive glob
            if path.is_file():
                files.append(str(path.resolve()))  # absolute path
        return files
    
    def split_documents(self, documents):
        return self.splitter.split_documents(documents)
    

