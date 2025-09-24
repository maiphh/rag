from tempfile import template
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from util import *
from operator import itemgetter


def simple_rag_chain(llm, retriever):
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    cached = (
        {"question": RunnablePassthrough()}
        | RunnablePassthrough.assign(docs = itemgetter("question") | retriever)
    )

    chain = (
        {"context": itemgetter("docs"), "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return (
        cached 
        | RunnableParallel({
            "answer": chain,
            "docs": itemgetter("docs")
        })
    )

def multi_query_chain(llm, retriever):

    template = """You are an AI language model assistant. Your task is to generate 3 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines, provides the answer only. Original question: {question}"""
    
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

    cached = (
        {"question": RunnablePassthrough()}
        | RunnablePassthrough.assign(docs = itemgetter("question") | retrieval_chain)
    )

    template = """Answer the following question based on this context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    final_chain = (
        {"context": itemgetter("docs"), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return (
        cached
        | RunnableParallel({
            "answer": final_chain,
            "docs": itemgetter("docs")
        })
    )

def rag_fusion_chain(llm, retriever):

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

    cached = (
        {"question": RunnablePassthrough()}
        | RunnablePassthrough.assign(docs = itemgetter("question") | retrieval_chain)
        | RunnablePassthrough.assign(ranked_docs = itemgetter("docs") | RunnableLambda(reciprocal_rank_fusion))
    )

    template = """Answer the following question based on this context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    final_chain = (
        {"context": itemgetter("ranked_docs"), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return (
        cached
        | RunnableParallel({
            "answer": final_chain,
            "docs": itemgetter("docs")
        })
    )
    

    
