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

    def split_documents(self, documents):
        return self.splitter.split_documents(documents)
    
    # def add_to_vectorstore(self, chunks):
    #     chunks_with_id = self.calculate_chunk_ids(chunks)
    #     existing_items = db.get(include=[])
    #     existing_ids = set(existing_items['ids'])
    #     print(f"Number of existing documents in DB: {len(existing_ids)}")

    #     # Add new chunks
    #     new_chunks = []
    #     for chunk in chunks_with_id:
    #         if chunk.metadata["id"] not in existing_ids:
    #             new_chunks.append(chunk)
        
    #     if len(new_chunks):
    #         print(f"👉 Adding new documents: {len(new_chunks)}")
    #         new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
    #         db.add_documents(new_chunks, ids=new_chunk_ids)

    #     else:
    #         print("✅ No new documents to add")


    # def calculate_chunk_ids(self, chunks):

    #     # This will create IDs like "data/monopoly.pdf:6:2"
    #     # Page Source : Page Number : Chunk Index

    #     last_page_id = None
    #     current_chunk_index = 0

    #     for chunk in chunks:
    #         source = chunk.metadata.get("source")
    #         page = chunk.metadata.get("page")
    #         current_page_id = f"{source}:{page}"

    #         # If the page ID is the same as the last one, increment the index.
    #         if current_page_id == last_page_id:
    #             current_chunk_index += 1
    #         else:
    #             current_chunk_index = 0

    #         # Calculate the chunk ID.
    #         chunk_id = f"{current_page_id}:{current_chunk_index}"
    #         last_page_id = current_page_id

    #         # Add it to the page meta-data.
    #         chunk.metadata["id"] = chunk_id

    #     return chunks
    
    

    
document_loader = DocumentLoader()

