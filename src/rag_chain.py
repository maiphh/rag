from db import chromaDb
from document_loader import document_loader
from retriever import retriever
from config import settings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = settings.get_llm()
embed = settings.get_embed()
retriever = chromaDb.retriever

class Chain:
    def __init__(self):
        pass

    def rag_chain(self):
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


chain = Chain()