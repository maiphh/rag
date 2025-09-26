from huggingface_hub import snapshot_download
from dotenv import load_dotenv
from pathlib import Path
import json

def get_json_from_path(path: str) -> dict:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

print(get_json_from_path("/Users/phu.mai/Projects/rag/data/cache/The_Japan_s_AI_White_Paper_English_Translaiton__1684318555.pdf.json"))