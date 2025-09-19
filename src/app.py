from dotenv import load_dotenv
from config import settings
from db import chromaDb
from document_loader import document_loader
from dotenv import load_dotenv
import os
from test import test_tracing
from rag_chain import chain

def main():
    load_dotenv()

    llm = settings.get_llm()
    embed = settings.get_embed()

    docs = document_loader.load_documents()
    chunks = document_loader.split_documents(docs)
    chromaDb.add_to_db(chunks)

    rag = chain.rag_chain()
    rag.invoke("Machine learning in broiler")

    




    



main()