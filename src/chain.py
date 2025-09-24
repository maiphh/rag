from tempfile import template
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from util import *
from operator import itemgetter
import prompt_template
from langchain.retrievers import ContextualCompressionRetriever



def simple_rag_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_template(prompt_template.QA_TEMPLATE)
    
    cached = (
        {"question": RunnablePassthrough()}
        | RunnablePassthrough.assign(docs = itemgetter("question") | retriever)
    )

    chain = (
        {"context": itemgetter("docs") | RunnableLambda(format_docs), "question": itemgetter("question")}
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
    prompt_perspectives = ChatPromptTemplate.from_template(prompt_template.MULTI_QUERY_TEMPLATE)


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

    prompt = ChatPromptTemplate.from_template(prompt_template.QA_TEMPLATE)

    final_chain = (
        {"context": itemgetter("docs") | RunnableLambda(format_docs), "question": RunnablePassthrough()}
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

    prompt_perspectives = ChatPromptTemplate.from_template(prompt_template.MULTI_QUERY_TEMPLATE)

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

    prompt = ChatPromptTemplate.from_template(prompt_template.QA_TEMPLATE)

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
    

    
