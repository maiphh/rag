from langchain.load import dumps, loads
from sentence_transformers import CrossEncoder
from langchain_core.runnables import RunnableLambda
from pathlib import Path
def split_queries(queries : str):
    return [q.strip() for q in queries.splitlines() if q.strip()]

def get_unique_union(documents: list[list], by_content_only=True):
    """ Unique union of retrieved docs """
    if by_content_only:
        # Deduplicate based on page_content only, ignoring metadata
        flattened_docs = [doc for sublist in documents for doc in sublist]
        seen_content = set()
        unique_docs = []
        
        for doc in flattened_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
        
        return unique_docs
    else:
        # Original behavior: deduplicate exact matches including metadata
        # Flatten list of lists, and convert each Document to string
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Get unique documents
        unique_docs = list(set(flattened_docs))
        # Return
        return [loads(doc) for doc in unique_docs]
    
def limit_docs(documents: list, limit=4):
    """ Limit number of documents """
    return documents[:limit]
    
def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rerank_docs(top_n=5, model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    ce = CrossEncoder(model)
    def _fn(response):
        q = response["question"]; docs = response["docs"] or []
        if not docs: return []
        scores = ce.predict([(q, d.page_content) for d in docs])
        reranked = [d for d, _ in sorted(zip(docs, scores), key=lambda z: z[1], reverse=True)]
        return reranked[:top_n]
    return RunnableLambda(_fn)

def build_match_tokens(values):
    tokens = set()
    for value in values:
        if not value:
            continue
        text = str(value)
        variants = {
            text,
            text.replace(" ", "_"),
        }

        p = Path(text)
        variants.update({
            p.stem,
            p.name,
            p.stem.replace(" ", "_"),
            p.name.replace(" ", "_"),
        })

        tokens.update(v.lower() for v in variants if v)
    return tokens
