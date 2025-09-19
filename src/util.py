from langchain.load import dumps, loads

def split_queries(queries):
    return queries.split("\n")

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