from pathlib import Path
from langchain_core.documents import Document
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from langchain_core.document_loaders import BaseLoader
import json  # added
import hashlib  # added
import shutil

from enum_manager import *
default_root = "data"

class DoclingLoader(BaseLoader):
    def __init__(self, path: str | list[str], cache_dir="data/cache"):
        self._file_paths = path if isinstance(path,list) else [path]

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False # pick what you need  
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=DoclingParseV2DocumentBackend)
            }
        )
        self.cache_dir = cache_dir

          
    
    def lazy_load(self):
        for path in self._file_paths:
            docling_doc = self._converter.convert(path).document
            text = docling_doc.export_to_markdown()
            domain = self.get_domain_from_path(path)
            metadata = {"source": str(Path(path).stem), "domain": domain}
            self.cache(text, metadata)
            yield Document(page_content=text, metadata=metadata)

    def get_domain_from_path(self,path: str) -> str | None:
        p = Path(path).resolve()
        domain_map = {d.value.lower(): d for d in DOMAIN}

        for parent in p.parents:
            name = parent.name.lower()
            if name in domain_map:
                return domain_map[name].value  # or domain_map[name].value
        return DOMAIN.ALL.value
    
    def cache(self, content, metadata):
        src = metadata.get("source", "unknown")
        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Build deterministic, safe filename
        base_name = "unknown" if src == "unknown" else Path(src).name
        safe_name = base_name.replace(" ", "_")
        file_name = f"{safe_name}.json"
        file_path = cache_path / file_name

        # Prepare JSON payload (note: key 'metdata' as specified)
        payload = {
            "content": content,
            "metadata": metadata,
        }

        # If file exists with identical payload, skip rewrite
        if file_path.exists():
            try:
                existing = json.loads(file_path.read_text(encoding="utf-8"))
                if existing == payload:
                    return file_path
            except Exception:
                pass  # proceed to overwrite if unreadable / malformed

        tmp_path = file_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path.replace(file_path)

        return file_path

class DocumentLoader:
    
    def __init__(self, embed, cache_dir):
        
        self.embed = embed
    
        # Supported file extensions
        self.supported_extensions = {
            '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.html', '.htm', '.md', '.txt'
        }

        self.cache_dir = cache_dir



    def load_documents(self, root=default_root, loaded_files: list[str] | None = None):
        files = self.get_all_files(root)

        loaded_files = loaded_files or []
        loaded_tokens = build_match_tokens(loaded_files)

        unloaded_files = [
            f for f in files
            if build_match_tokens([f]).isdisjoint(loaded_tokens)
        ]
        # print("[document_loader] files:", files)
        print("[document_loader] loaded_files:", sorted(loaded_tokens))
        # print("[document_loader] unloaded_files:", unloaded_files)
        
        if not unloaded_files:
            print("No new files to load.")
            return []
        
        # loader = PyPDFDirectoryLoader(root)
        loader = DoclingLoader(unloaded_files,self.cache_dir)
        return loader.load()

    def get_all_files(self, root=default_root) -> list[str]:
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"❌ The directory '{root}' does not exist.")

        files = []
        for path in root_path.rglob("*"):  # rglob('*') = recursive glob
            if path.is_file() and not path.name.startswith('.') and not path.name.startswith('~'):
                if path.suffix.lower() in self.supported_extensions:
                    files.append(str(path.resolve()))  # absolute path
        return files



