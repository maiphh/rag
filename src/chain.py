from db import chromaDb
from document_loader import document_loader
from config import settings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from util import *

llm = settings.get_llm()
retriever = chromaDb.retriever

class Chain:
    def __init__(self):
        pass

    def simple_rag_chain(self):
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
      
        return chain
    
    def multi_query_chain(self):
        template = """You are an AI language model assistant. Your task is to generate 3 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""
        
        prompt_perspectives = ChatPromptTemplate.from_template(template)

        generate_queries_chain = (
            prompt_perspectives
            |llm
            | StrOutputParser()
            | split_queries
        )

        retrieval_chain = (
            generate_queries_chain
            | retriever.map()
            | get_unique_union
        )

        template = """Answer the following question based on this context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        final_chain = (
            {"context": retrieval_chain, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return final_chain

        

chain = Chain()