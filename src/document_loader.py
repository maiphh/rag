from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma



default_root = "data"

class DocumentLoader:
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


    def load_documents(self, root = default_root):
        loader = PyPDFDirectoryLoader(root)
        return loader.load()
    
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

