from enum import Enum
from chain import *

class LLM(Enum):
    LLAMA3_8B = "llama3:8b"
    GEMMA3_1B = "gemma3:1b"
    LLAMA3_2 = "llama3.2:latest"
    QWEN3 = "qwen3:8b"
    LLMA3_QA = "llama3-chatqa:latest"

class EMBEDDING(Enum):
    NOMIX_EMBED_TEXT = "nomic-embed-text:latest"
    QWEN3_EMBED = "qwen3-embedding:latest"
    MX_BAI = "mxbai-embed-large:latest"


class RagType(Enum):
    SIMPLE = "simple"
    MULTI_QUERY = "multi_query"

RAG_TYPE_BUILDERS = {
    RagType.SIMPLE: simple_rag_chain,
    RagType.MULTI_QUERY: multi_query_chain,
    # RagType.RAG_FUSION: rag_fusion_chain,
}

class DOMAIN(Enum):
    ALL = "all"
    PAYROLL = "pyr"
    ECON = "econ"
    TEST = "test"

class RERANKER(Enum):
    MACRO_MINI = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # MACRO_MINI = "models/cross-encoder/ms-marco-MiniLM-L-6-v2"