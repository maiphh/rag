from langchain_chroma import Chroma
from config import settings
from langsmith.run_helpers import traceable


db_path = "db"
SCORE_THRESHOLD = 0.5

class Database:
    def __init__(self, embed):
        self.db = Chroma(
            persist_directory = db_path,
            embedding_function = embed
        )

        self.retriever = self.db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":20, "score_threshold": SCORE_THRESHOLD})

    def add_to_db(self, chunks):
        chunks_with_id = self.calculate_chunk_ids(chunks)
        existing_items = self.db.get(include=[])
        existing_ids = set(existing_items['ids'])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Add new chunks
        new_chunks = []
        for chunk in chunks_with_id:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)
        
        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self.db.add_documents(new_chunks, ids=new_chunk_ids)

        else:
            print("✅ No new documents to add")


    def calculate_chunk_ids(self, chunks):

        # This will create IDs like "data/monopoly.pdf:6:2"
        # Page Source : Page Number : Chunk Index

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

        return chunks
    
    def get_loaded_src(self) -> list[str]:
        existing = self.db.get(include=["metadatas"])
        loaded_sources = {meta["source"] for meta in existing["metadatas"] if "source" in meta}
        print("Already loaded:", loaded_sources)
        return list(loaded_sources)
    
    def clear(self):
        self.db.delete_collection()
        print("🗑️  Database cleared")

    @traceable(name="retrieve")
    def retrieve(self, query, threshold = SCORE_THRESHOLD):
        return self.retriever.invoke(query, threshold = threshold)
    
    def get_retriever(self):
        return self.retriever

