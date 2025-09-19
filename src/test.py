
from langchain.prompts import PromptTemplate
from config import settings

def test_tracing():
    # Simple chain to test tracing
    prompt = PromptTemplate(
        input_variables=["question"],
        template="Answer this question: {question}"
    )
    llm = settings.get_llm()
    chain = llm | prompt
    
    # This should appear in LangSmith
    result = chain.invoke("What is 2+2?")
    print("Test result:", result)
