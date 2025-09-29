from langchain_chroma import Chroma
from langsmith.run_helpers import traceable
from enum_manager import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json  # add
from pathlib import Path
import shutil
from langchain_experimental.text_splitter import SemanticChunker
from util import *
    
class Database:
    def __init__(self, embed, dir, cache_dir):
        self.db = Chroma(
            persist_directory = dir,
            embedding_function = embed
        )

        # self.splitter = SemanticChunker(embed)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
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
        
        
    def get_cached_docs(self) -> list[Document]:
        loaded_tokens = build_match_tokens(self.get_loaded_src())

        docs = []

        for cached_file in self._list_cache_files():
            cache_tokens = build_match_tokens([
                cached_file.name,
                cached_file.stem,
                cached_file.with_suffix("").name,
            ])
            if not cache_tokens.isdisjoint(loaded_tokens):
                continue
            try:
                docs.append(self.parse_json_to_document(str(cached_file)))
            except Exception as e:
                print(f"⚠️  Skipping {cached_file.stem}: {e}")

        if not docs:
            print("No cached documents found.")
        return docs


    
    
    def load_cached_docs(self):
        print("Loading cached documents...")
        docs = self.get_cached_docs()
        print(f"Found {len(docs)} cached documents.")
        if docs:
            chunks = self.split_documents(docs)
            self.add(chunks)

    def get_cached_src(self) -> list[str]:
        return [fp.stem for fp in self._list_cache_files()]

    def _list_cache_files(self) -> list[Path]:
        cache_path = Path(self.cache_dir)
        if not cache_path.exists() or not cache_path.is_dir():
            return []
        return sorted(fp for fp in cache_path.glob("*.json") if fp.is_file())

    
    def clear_cache(self):
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            print("Cache directory does not exist.")
            return

        shutil.rmtree(cache_path)
        print("Cache cleared.")