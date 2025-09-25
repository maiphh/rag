import streamlit as st
from dotenv import load_dotenv
from rag import Rag, LLM, RagType
from enum_manager import DOMAIN   # <-- added import

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
    # map current value to label
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

# ---------------- Main UI ----------------
st.title("📚 RAG Document Q&A")
st.caption("Ask questions over your documents with selectable LLM + Retrieval strategy.")

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            with st.expander(f"📄 Retrieved Documents ({len(m['sources'])})"):
                for i, src in enumerate(m["sources"], start=1):
                    st.markdown(f"**{i}. {src['source']}**")
                    if src.get("content"):
                        st.code(src["content"], language="markdown")

# Input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = "Error: Unknown"
            sources_payload = []
            try:
                answer, retrieved_docs = rag.invoke(prompt)
                for d in retrieved_docs:
                    meta = getattr(d, "metadata", {}) or {}
                    source = meta.get("source") or meta.get("file_path") or "Unknown source"
                    content = getattr(d, "page_content", "")[:650]
                    sources_payload.append({
                        "source": source,
                        "content": content
                    })
            except Exception as e:
                answer = f"Error: {e}"

            st.markdown(answer)
            if sources_payload:
                with st.expander(f"📄 Retrieved Documents ({len(sources_payload)})", expanded=False):
                    for i, src in enumerate(sources_payload, start=1):
                        st.markdown(f"**{i}. {src['source']}**")
                        if src.get("content"):
                            st.code(src["content"], language="markdown")
                            # st.markdown(src["content"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources_payload
    })

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; font-size:0.8em;'>Powered by custom RAG (Ollama + LangChain)</div>",
    unsafe_allow_html=True
)
