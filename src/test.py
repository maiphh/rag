from enum_manager import *
from pathlib import Path

def get_domain_from_path(path: str) -> str | None:
        p = Path(path).resolve()
        domain_map = {d.value.lower(): d for d in DOMAIN}

        for parent in p.parents:
            name = parent.name.lower()
            if name in domain_map:
                return domain_map[name].value  # or domain_map[name].value
        return DOMAIN.ALL.value

print(get_domain_from_path("/Users/phu.mai/Projects/rag/data/Hướng dẫn dùng chức năng Scan trên máy Fuji.docx"))