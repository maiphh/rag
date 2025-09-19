from dotenv import load_dotenv
from config import settings
from db import chromaDb
from document_loader import document_loader
from dotenv import load_dotenv
import os
from test import test_tracing
from chain import chain

def main():
    
    if not load_dotenv():
        print(".env file not found")
        return

    llm = settings.get_llm()
    embed = settings.get_embed()

    docs = document_loader.load_documents()
    chunks = document_loader.split_documents(docs)
    chromaDb.add_to_db(chunks)

    rag = chain.simple_rag_chain(llm)
    multi_rag = chain.multi_query_chain(llm)
    
    multi_rag.invoke("Broiler health and machine learning")


    




    
main()