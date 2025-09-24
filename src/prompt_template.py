QA_TEMPLATE = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

MULTI_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate 3
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.

    RULES:
    - Output only the 3 alternative queries, one per line.
    - Do not repeat the original question.
    - No numbering, bullets, or quotes.
    - Keep meaning, vary wording; include synonyms, broader/narrower forms, and common phrasing.
    - Do not put trailing newlines after the last query.

    OUTPUT: 
    - Return answer only, no explanations, no introduction.
    - Each query on its own line.
    - Exactly 3 queries.

    Examples:
    Original question: Who is the CEO of Apple?
    Alternative queries:
    Who currently leads Apple as CEO?
    Who holds the position of Apple's chief executive?
    Who is Apple's current chief executive officer?

    Original question: {question}
    Alternative queries:"""