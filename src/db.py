from langchain_chroma import Chroma
from langsmith.run_helpers import traceable
from enum_manager import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json  # add
from pathlib import Path
import shutil
    
class Database:
    def __init__(self, embed, dir, cache_dir):
        self.db = Chroma(
            persist_directory = dir,
            embedding_function = embed
        )

        # self.splitter = SemanticChunker(embed)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
        # self.splitter = KamradtModifiedChunker(avg_chunk_size=400, min_chunk_size=50, embedding_function= self.embed)
        self.source_id_map = {}  # source file path -> list of chunk IDs
        self.cache_dir = cache_dir

    def add(self, big_chunks):
        for chunks in self.batch(big_chunks, 2000):
            self.add_to_db(chunks)

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
    

        for chunk in chunks:
            source = chunk.metadata.get("source")
            current_chunk_index = self.get_current_source_index(source)
            self.set_current_source_index(source, current_chunk_index + 1)
            # Calculate the chunk ID.
            chunk_id = f"{source}:{current_chunk_index}"
            
            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

        return chunks

    def get_current_source_index(self,src) -> int:
        return self.source_id_map.get(src, 0)
    
    def set_current_source_index(self,src, index):
        self.source_id_map[src] = index
    
    def get_loaded_src(self) -> list[str]:
        existing = self.db.get(include=["metadatas"])
        loaded_sources = {meta["source"] for meta in existing["metadatas"] if "source" in meta}
        return list(loaded_sources)
    
    def clear(self):
        self.db.delete_collection()
        print("🗑️  Database cleared")


    def batch(self,docs, size):
        for i in range(0, len(docs), size):
            yield docs[i:i+size]

    def split_documents(self, documents):
        return self.splitter.split_documents(documents)


    def parse_json_to_document(self, json_path: str) -> Document:
        """
        Parse a single cached .json file into a LangChain Document.
        Expects format:
        {
            "content": "...",
            "metdata": { ... }   # note: key intentionally spelled 'metdata'
        }
        """
        p = Path(json_path)
        raw = self.get_json_from_path(str(p))

        if "content" not in raw:
            raise ValueError(f"Missing 'content' in {json_path}")

        # Handle typo key 'metdata' while remaining tolerant.
        metadata = raw.get("metdata") or raw.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Metadata not a dict in {json_path}")

        return Document(page_content=raw["content"], metadata=metadata)
    
    def get_json_from_path(self, path: str) -> dict:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
            return raw
        print(raw)
        
        
    def get_cached_docs(self, paths: list[str]) -> list[Document]:
        docs = []
        for fp in self.get_cached_src():
            try:
                raw = self.get_json_from_path(str(fp))
                metadata = raw.get("metadata")
                source = metadata.get("source")
                if source and source not in paths:
                    docs.append(self.parse_json_to_document(str(fp)))
            except Exception as e:
                print(f"⚠️  Skipping {fp}: {e}")
        
        if not len(docs):
            print("No cached documents found.")
        return docs

    def get_cached_src(self) -> list[str]:
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            return []

        srcs = []
        for fp in cache_path.glob("*.json"):
            raw = self.get_json_from_path(str(fp))
            metadata = raw.get("metadata")
            source = metadata.get("source")
            if source and source not in srcs:
                srcs.append(source)
        return srcs

    
    def clear_cache(self):
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            print("Cache directory does not exist.")
            return

        shutil.rmtree(cache_path)
        print("Cache cleared.")