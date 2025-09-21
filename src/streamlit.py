import streamlit as st
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import your existing modules
from config import settings
from db import chromaDb
from chain import Chain

# Optional OpenAI import for better streaming support
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Load environment variables
load_dotenv()

# Initialize your existing chain system
chain_system = Chain()


def get_context_from_chroma(query: str, num_results: int = 5) -> str:
    """Search ChromaDB for relevant context using your existing retriever.

    Args:
        query: User's question
        num_results: Number of results to return

    Returns:
        str: Concatenated context from relevant chunks with source information
    """
    # Use your existing retriever
    documents = chromaDb.retrieve(query)
    
    if not documents:
        return "No relevant context found."
    
    # Limit to num_results if needed
    if len(documents) > num_results:
        documents = documents[:num_results]
    
    contexts = []
    
    for doc in documents:
        # Extract your existing metadata structure
        metadata = doc.metadata
        text = doc.page_content
        
        # Build source citation from your metadata structure
        source_parts = []
        
        # Your metadata has 'source', 'page', and 'id'
        source_file = metadata.get("source", "Unknown source")
        page_num = metadata.get("page", "Unknown page")
        
        # Extract filename from source path
        if source_file and source_file != "Unknown source":
            filename = os.path.basename(source_file)
            source_parts.append(filename)
        
        if page_num and page_num != "Unknown page":
            source_parts.append(f"p. {page_num}")
        
        source = f"\nSource: {' - '.join(source_parts) if source_parts else 'Unknown'}"
        
        contexts.append(f"{text}{source}")
    
    return "\n\n".join(contexts)


def get_chat_response_ollama(messages: List[Dict], context: str) -> str:
    """Get response using your existing Ollama setup.

    Args:
        messages: Chat history
        context: Retrieved context from database

    Returns:
        str: Model's response
    """
    # Use your existing simple RAG chain
    chain = chain_system.simple_rag_chain()
    
    # Get the latest user message
    latest_question = messages[-1]["content"] if messages else ""
    
    # Since your chain expects just a question and uses the retriever internally,
    # we'll use it directly
    try:
        response = chain.invoke(latest_question)
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"


def get_chat_response_openai(messages: List[Dict], context: str) -> str:
    """Get streaming response from OpenAI API (if available).

    Args:
        messages: Chat history
        context: Retrieved context from database

    Returns:
        str: Model's response
    """
    if not OPENAI_AVAILABLE:
        return get_chat_response_ollama(messages, context)
    
    try:
        client = OpenAI()
        
        system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
        Use only the information from the context to answer questions. If you're unsure or the context
        doesn't contain the relevant information, say so.
        
        Context:
        {context}
        """

        messages_with_context = [{"role": "system", "content": system_prompt}, *messages]

        # Create the streaming response
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_with_context,
            temperature=0.7,
            stream=True,
        )

        # Use Streamlit's built-in streaming capability
        response = st.write_stream(stream)
        return response
    except Exception as e:
        st.warning(f"OpenAI not available: {str(e)}. Falling back to Ollama.")
        return get_chat_response_ollama(messages, context)


# Initialize Streamlit app
st.set_page_config(
    page_title="📚 Document Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Document Q&A")
st.caption("Ask questions about your documents using RAG")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # LLM Selection
    llm_option = st.selectbox(
        "Choose LLM:",
        ["Ollama (Mistral)", "OpenAI (GPT-4o-mini)"] if OPENAI_AVAILABLE else ["Ollama (Mistral)"],
        help="Select which language model to use for responses"
    )
    
    # Number of context chunks
    num_results = st.slider(
        "Context chunks:",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of relevant document chunks to retrieve"
    )
    
    # Database info
    st.subheader("📊 Database Info")
    try:
        # Get some basic stats about your database
        all_docs = chromaDb.db.get()
        total_docs = len(all_docs['ids']) if all_docs['ids'] else 0
        st.metric("Total chunks", total_docs)
    except Exception as e:
        st.error(f"Could not load database stats: {str(e)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get relevant context
    with st.status("Searching documents...", expanded=False) as status:
        context = get_context_from_chroma(prompt, num_results)
        
        # Custom CSS for better styling
        st.markdown(
            """
            <style>
            .search-result {
                margin: 10px 0;
                padding: 10px;
                border-radius: 4px;
                background-color: #f0f2f6;
                border: 1px solid #e0e0e0;
            }
            .search-result summary {
                cursor: pointer;
                color: #0f52ba;
                font-weight: 500;
                margin-bottom: 5px;
            }
            .search-result summary:hover {
                color: #1e90ff;
            }
            .metadata {
                font-size: 0.9em;
                color: #666;
                font-style: italic;
                margin-bottom: 8px;
            }
            .chunk-text {
                line-height: 1.5;
                color: #333;
            }
            </style>
        """,
            unsafe_allow_html=True,
        )

        st.write("Found relevant sections:")
        
        # Display context chunks
        for i, chunk in enumerate(context.split("\n\n")):
            if not chunk.strip():
                continue
                
            # Split into text and source parts
            parts = chunk.split("\nSource: ")
            text = parts[0].strip()
            source = parts[1] if len(parts) > 1 else "Unknown source"

            st.markdown(
                f"""
                <div class="search-result">
                    <details>
                        <summary>📄 {source}</summary>
                        <div class="chunk-text">{text}</div>
                    </details>
                </div>
            """,
                unsafe_allow_html=True,
            )
        
        status.update(label="Documents found!", state="complete")

    # Display assistant response
    with st.chat_message("assistant"):
        # Choose response method based on selection
        if llm_option == "OpenAI (GPT-4o-mini)" and OPENAI_AVAILABLE:
            response = get_chat_response_openai(st.session_state.messages, context)
        else:
            with st.spinner("Generating response with Ollama..."):
                response = get_chat_response_ollama(st.session_state.messages, context)
                st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    🤖 Powered by your RAG system with ChromaDB and Ollama/OpenAI
    </div>
    """,
    unsafe_allow_html=True
)
