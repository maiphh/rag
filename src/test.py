

from rag import Rag
from chain import simple_rag_chain, rag_fusion_chain, multi_query_chain

rag = Rag()
response = simple_rag_chain(rag.get_llm(), rag.get_retriever()).invoke("What is RAG?")
print("answer:", response["answer"])
print("docs:", response["docs"])
