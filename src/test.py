from huggingface_hub import snapshot_download
from dotenv import load_dotenv
from pathlib import Path
import json
from rag import Rag


rag = Rag()
print(rag.evaluate_retrieval_performance())