import streamlit as st
from dotenv import load_dotenv
from rag import Rag, LLM, RagType
from enum_manager import DOMAIN   # <-- added import
from pathlib import Path          # <-- new
import re                         # <-- new
import zipfile                    # <-- new
import io                         # <-- new
import html                       # <-- new

load_dotenv()

st.set_page_config(
    page_title="📚 RAG Playground",
    page_icon="📚",
    layout="wide"
)

# ---------------- Session State ----------------
if "rag" not in st.session_state:
    st.session_state.rag = Rag()
if "messages" not in st.session_state:
    st.session_state.messages = []

rag: Rag = st.session_state.rag

# ---------------- Helpers ----------------
def format_rag_type(rt: RagType) -> str:
    return rt.name.title().replace("_", " ")

def parse_rag_type(label: str) -> RagType:
    lookup = {format_rag_type(r): r for r in RagType}
    return lookup[label]

# ---- New helpers for Domain ----
def format_domain(d: DOMAIN) -> str:
    return d.name.title().replace("_", " ")

def parse_domain(label: str) -> DOMAIN:
    lookup = {format_domain(d): d for d in DOMAIN}
    return lookup[label]

def _safe_filename(name: str) -> str:
    # Keep extension, sanitize rest
    name = name.strip()
    if not name:
        name = "uploaded"
    # Split extension
    if "." in name:
        base, ext = name.rsplit(".", 1)
        ext = "." + ext
    else:
        base, ext = name, ""
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)[:120] or "file"
    return base + ext

def _unique_path(dir_path: Path, filename: str) -> Path:
    candidate = dir_path / filename
    if not candidate.exists():
        return candidate
    base, ext = (filename.rsplit(".", 1) + [""])[:2]
    if ext:
        ext = "." + ext
    base_clean = base
    i = 1
    while candidate.exists():
        candidate = dir_path / f"{base_clean}_{i}{ext}"
        i += 1
    return candidate

def _write_unique_bytes(save_dir: Path, filename: str, data: bytes) -> str | None:
    target = save_dir / filename
    if target.exists():
        if target.read_bytes() == data:
            return None
        target = _unique_path(save_dir, filename)
    with open(target, "wb") as out:
        out.write(data)
    return str(target)

def _save_uploaded_files(domain: DOMAIN, uploaded_files):
    save_dir = Path("data") / domain.value
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    extracted_paths = []
    for uf in uploaded_files:
        fname = _safe_filename(uf.name)
        # If it's a zip, extract
        if fname.lower().endswith(".zip"):
            try:
                z = zipfile.ZipFile(io.BytesIO(uf.getbuffer()))
                for member in z.infolist():
                    if member.is_dir():
                        continue
                    inner_name = _safe_filename(Path(member.filename).name)
                    inner_path = _unique_path(save_dir, inner_name)
                    with z.open(member) as src, open(inner_path, "wb") as dst:
                        dst.write(src.read())
                    extracted_paths.append(str(inner_path))
            except zipfile.BadZipFile:
                # Fallback: save raw
                path = _unique_path(save_dir, fname)
                with open(path, "wb") as out:
                    out.write(uf.getbuffer())
                saved_paths.append(str(path))
        else:
            written = _write_unique_bytes(save_dir, fname, uf.getbuffer())
            if written:
                saved_paths.append(written)
    return saved_paths, extracted_paths

def _index_domain(domain: DOMAIN):
    # Try domain-aware loading; fallback otherwise
    try:
        added = rag.load_documents(domain=domain.value)
    except TypeError:
        added = rag.load_documents()
    return added

def _render_rag_content_block(content: str):
    if not content:
        st.info("No preview available.")
        return
    st.markdown(
        f"""
        <div style="
            background-color:rgba(240,240,240,0.4);
            padding:0.75rem;
            border-radius:0.5rem;
            white-space:pre-wrap;
            word-break:break-word;
            font-family:var(--font-mono);
            font-size:0.85rem;
        ">
            {html.escape(content)}
        </div>
        """,
        unsafe_allow_html=True,
    )

def _render_retrieved_documents(query, sources, base_key: str, expanded: bool = False):
    if not sources:
        return
    with st.expander(f"📄 Retrieved Documents ({len(sources)})", expanded=expanded):
        if query:
            st.markdown(f"**Query:** {query}")
        for doc_number, src in enumerate(sources, start=1 if query is None else 2):
            source_name = src.get("source") or "Unknown source"
            raw_conf = src.get("confidence")
            conf_label = src.get("confidence_label") or "unknown"
            try:
                conf_str = f"{float(raw_conf):.2f}" if raw_conf is not None else "N/A"
            except Exception:
                conf_str = str(raw_conf)
            toggle_label = f"{doc_number}. {source_name} • Confidence: {conf_str} ({conf_label})"
            show_doc = st.toggle(toggle_label, key=f"{base_key}_doc_{doc_number}")
            if show_doc:
                st.caption(f"Source: {source_name}")
                st.caption(f"Confidence: {conf_str} ({conf_label})")
                _render_rag_content_block(src.get("content") or "")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    # LLM selection
    llm_values = [e.value for e in LLM]
    current_llm_model = rag.get_llm().model
    llm_index = llm_values.index(current_llm_model) if current_llm_model in llm_values else 0
    selected_llm_value = st.selectbox("LLM", llm_values, index=llm_index)
    if selected_llm_value != current_llm_model:
        for enum_item in LLM:
            if enum_item.value == selected_llm_value:
                rag.set_llm(enum_item)
                break

    # RAG strategy
    rag_labels = [format_rag_type(r) for r in RagType]
    current_rag_label = format_rag_type(rag.get_rag_type())
    rag_index = rag_labels.index(current_rag_label)
    selected_rag_label = st.selectbox("RAG Type", rag_labels, index=rag_index)
    if selected_rag_label != current_rag_label:
        rag.set_rag_type(parse_rag_type(selected_rag_label))

    # Domain selection (NEW)
    domain_labels = [format_domain(d) for d in DOMAIN]
    try:
        current_domain_value = rag.get_domain()
    except AttributeError:
        current_domain_value = getattr(getattr(rag, "domain_router", {}), "domain", None)
    current_domain_enum = None
    for d in DOMAIN:
        if d.value == current_domain_value or d.name == str(current_domain_value):
            current_domain_enum = d
            break
    current_domain_label = format_domain(current_domain_enum) if current_domain_enum else domain_labels[0]
    domain_index = domain_labels.index(current_domain_label)
    selected_domain_label = st.selectbox("Domain", domain_labels, index=domain_index)
    if selected_domain_label != current_domain_label:
        rag.set_domain(parse_domain(selected_domain_label))

    # Retriever threshold slider (NEW)
    try:
        current_threshold = float(rag.get_threshold())
    except Exception:
        current_threshold = 0.0
    new_threshold = st.slider("Retriever Threshold", 0.0, 1.0, value=current_threshold, step=0.01)
    if abs(new_threshold - current_threshold) > 1e-9:
        rag.set_threshold(new_threshold)

    st.markdown("---")
    if st.button("🗑️ Clear Database"):
        rag.clear_db()
        st.success("Database cleared.")
        
    if st.button("🔄 Reload Documents"):
        with st.spinner("Loading documents..."):
            added = rag.load_documents()
            st.success(f"Loaded {len(added)} new documents." if added else "No new documents found.")

    # NEW: Load only new docs
    if st.button("🆕 Load New Docs"):
        with st.spinner("Loading new documents..."):
            try:
                if hasattr(rag, "load_new_documents"):
                    added = rag.load_new_documents()
                else:
                    # Try a parameter; fallback to normal
                    try:
                        added = rag.load_documents(new_only=True)
                    except TypeError:
                        added = rag.load_documents()
                st.success(f"Loaded {len(added)} new documents." if added else "No new documents found.")
            except Exception as e:
                st.error(f"Failed: {e}")

    # NEW: Clear cache
    if st.button("🧹 Clear Cache"):
        try:
            cleared = False
            if hasattr(rag, "clear_cache"):
                rag.clear_cache()
                cleared = True
            elif hasattr(rag, "retriever") and hasattr(rag.retriever, "clear_cache"):
                rag.retriever.clear_cache()
                cleared = True
            if cleared:
                st.success("Cache cleared.")
            else:
                st.info("No cache interface available.")
        except Exception as e:
            st.error(f"Cache clear failed: {e}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(f"Active LLM: {rag.get_llm().model}")
    st.caption(f"Active RAG: {format_rag_type(rag.get_rag_type())}")
    try:
        st.caption(f"Active Domain: {parse_domain(selected_domain_label).value}")
    except Exception:
        pass
    st.caption(f"Threshold: {new_threshold:.2f}")

# ---------------- Tabs ----------------
chat_tab, upload_tab = st.tabs(["💬 Chat", "📤 Upload Documents"])

# ---------------- Chat Tab ----------------
with chat_tab:
    st.title("📚 RAG Document Q&A")
    st.caption("Ask questions over your documents with selectable LLM + Retrieval strategy.")

    response_container = st.container()
    input_container = st.container()

    with response_container:
        # Display history
        for idx, m in enumerate(st.session_state.messages):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m["role"] == "assistant" and m.get("sources"):
                    # Find preceding user query for numbering as item 1
                    user_query = None
                    for back in range(idx - 1, -1, -1):
                        if st.session_state.messages[back]["role"] == "user":
                            user_query = st.session_state.messages[back]["content"]
                            break
                    _render_retrieved_documents(user_query, m["sources"], base_key=f"history_{idx}")

    with input_container:
        prompt = st.chat_input("Ask something...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with response_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = "Error: Unknown"
                    sources_payload = []
                    try:
                        response = rag.invoke(prompt)
                        answer = response.get("answer", "No answer.")
                        docs = response.get("docs", [])
                        for d in docs:
                            meta = getattr(d, "metadata", {}) or {}
                            source = meta.get("source") or meta.get("file_path") or "Unknown source"
                            confidence_score = meta.get("confidence", 0)
                            confidence_label = meta.get("confidence_label", "unknown")

                            content = getattr(d, "page_content", "")
                            sources_payload.append({
                                "source": source,
                                "content": content,
                                "confidence": confidence_score,
                                "confidence_label": confidence_label
                            })
                    except Exception as e:
                        answer = f"Error: {e}"

                    st.markdown(answer)
                    _render_retrieved_documents(prompt, sources_payload, base_key=f"live_{len(st.session_state.messages)}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources_payload
        })

    # --- Auto scroll to bottom when new messages appear (NEW) ---
    current_len = len(st.session_state.messages)
    prev_len = st.session_state.get("last_rendered_message_count", 0)
    with response_container:
        # Anchor at the end of the chat
        st.markdown("<div id='chat-end'></div>", unsafe_allow_html=True)
        if current_len != prev_len:
            st.session_state.last_rendered_message_count = current_len
            st.markdown(
                """
                <script>
                const el = document.getElementById('chat-end');
                if (el) {
                    el.scrollIntoView({behavior: 'auto', block: 'start'});
                }
                </script>
                """,
                unsafe_allow_html=True
            )
    # --- End auto scroll block (NEW) ---

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.8em;'>Powered by custom RAG (Ollama + LangChain)</div>",
        unsafe_allow_html=True
    )

# ---------------- Upload Tab ----------------
with upload_tab:
    st.header("📤 Domain Document Upload")
    st.caption("Upload or drag multiple files (.pdf, .txt, .md, .docx, or .zip) for each domain. Zips are extracted.")
    st.markdown(
        """
        - Each domain stores files under data/<domain_code> (e.g. data/pyr).
        - Uploading a .zip extracts its contents (files only).
        - Duplicate names get an auto-incremented suffix.
        - After uploading, click Index to parse and embed new documents.
        """
    )
    st.markdown("---")

    for d in DOMAIN:
        with st.expander(f"📁 {format_domain(d)}  (folder: data/{d.value})", expanded=False):
            uploaded = st.file_uploader(
                f"Drop files for {format_domain(d)}",
                accept_multiple_files=True,
                type=["pdf", "txt", "md", "docx", "zip"],
                key=f"uploader_{d.name}"
            )
            if uploaded:
                saved, extracted = _save_uploaded_files(d, uploaded)
                total = len(saved) + len(extracted)
                if total:
                    st.success(f"Saved {len(saved)} files, extracted {len(extracted)} from zips (total {total}).")
                    if len(saved):
                        st.caption("Saved: " + ", ".join(Path(p).name for p in saved[:6]) + (" ..." if len(saved) > 6 else ""))
                    if len(extracted):
                        st.caption("Extracted: " + ", ".join(Path(p).name for p in extracted[:6]) + (" ..." if len(extracted) > 6 else ""))
                    if st.button(f"Index {format_domain(d)} Documents", key=f"index_btn_{d.name}"):
                        with st.spinner("Indexing..."):
                            try:
                                added = _index_domain(d)
                                st.success(f"Indexed {len(added)} new documents." if added else "No new documents to index.")
                            except Exception as e:
                                st.error(f"Indexing failed: {e}")
                else:
                    st.warning("No files processed.")

    st.markdown("---")
    st.caption("Tip: For folder upload, compress the folder into a .zip and drop it here.")
