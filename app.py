"""
Streamlit RAG Chatbot Application
"""
import streamlit as st
import os
from pathlib import Path
from document_loader import DocumentLoader
from vector_store import VectorStore
from rag_engine import RAGEngine

# Page configuration
st.set_page_config(
    page_title="RAG Document Chatbot",
    layout="wide"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'loaded_documents' not in st.session_state:
    st.session_state.loaded_documents = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def initialize_vector_store():
    """Initialize vector store"""
    if st.session_state.vector_store is None:
        with st.spinner("Initializing vector store and embedding model..."):
            st.session_state.vector_store = VectorStore()
            st.success("Vector store initialized!")


def load_documents_from_directory(directory: str):
    """Load documents from directory"""
    loader = DocumentLoader()
    documents = loader.load_directory(directory)
    return documents


def process_and_index_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """Process documents and build index"""
    if not documents:
        return
    
    initialize_vector_store()
    loader = DocumentLoader()
    
    total_chunks = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, doc in enumerate(documents):
        status_text.text(f"Processing: {doc['filename']}")
        
        # Chunk text
        chunks = loader.chunk_text(doc['content'], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Add to vector store
        source_info = {
            'filename': doc['filename'],
            'extension': doc['extension'],
            'file_path': doc.get('file_path', '')
        }
        st.session_state.vector_store.add_documents(chunks, source_info)
        total_chunks += len(chunks)
        
        progress_bar.progress((idx + 1) / len(documents))
    
    status_text.text(f"Complete! Processed {len(documents)} documents, {total_chunks} chunks")
    progress_bar.empty()
    
    # Initialize RAG engine
    try:
        st.session_state.rag_engine = RAGEngine(st.session_state.vector_store)
        st.success("Document index built! You can start asking questions.")
    except Exception as e:
        st.error(f"Failed to initialize RAG engine: {str(e)}")
        st.info("Please make sure OPENAI_API_KEY is set in .env file")


def main():
    st.title("RAG Document Chatbot")
    
    # Sidebar
    with st.sidebar:

        st.subheader("Step 1: Load Documents")
        
        # Document directory selection
        doc_directory = "documents"
        
        st.text(f"Please put your documents in the `/{doc_directory}` directory and hit the \"Load Documents\" button")
        if st.button("📂 Load Documents", type="secondary"):
            with st.spinner("Loading documents..."):
                documents = load_documents_from_directory(doc_directory)
                if documents:
                    st.session_state.loaded_documents = documents
                    st.success(f"Successfully loaded {len(documents)} documents")
                else:
                    st.warning("No supported document formats found (.txt, .md, .docx)")
        
        
        # Chunking parameters
        st.subheader("Step 2: Set Chunking Parameters")
        chunk_size = st.slider("Chunk Size (characters)", 200, 1000, 500, 50)
        chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 200, 50, 10)
        
        st.subheader("Step 3: Build Vector Index")
        if st.button("Build Vector Index", type="secondary"):
            if st.session_state.loaded_documents:
                process_and_index_documents(
                    st.session_state.loaded_documents,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
                st.warning("Please load documents first")
        
        # Display loaded documents
        if st.session_state.loaded_documents:
            st.subheader("Loaded Documents")
            for doc in st.session_state.loaded_documents:
                st.text(f"• {doc['filename']}")
        
        # Display vector store status
        if st.session_state.vector_store:
            stats = st.session_state.vector_store.get_stats()
            st.subheader("📊 Index Statistics")
            st.metric("Total Chunks", stats['total_chunks'])
            st.metric("Vector Dimension", stats['dimension'])
        
        # Clear button
        if st.button("🗑️ Clear All Data"):
            st.session_state.vector_store = None
            st.session_state.rag_engine = None
            st.session_state.loaded_documents = []
            st.session_state.chat_history = []
            st.rerun()
    
    
    # Check if RAG engine is ready
    if st.session_state.rag_engine is None:
        st.info("👈 Please load documents and build vector index in the sidebar to start asking questions")
    else:
        # Display chat history
        for i, chat in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(chat['question'])
            
            with st.chat_message("assistant"):
                st.write(chat['answer'])
                
                # Display retrieved document chunks (expandable)
                with st.expander(f"📄 View Retrieved Document Chunks (Similarity Scores)"):
                    for j, doc in enumerate(chat['retrieved_docs'], 1):
                        st.markdown(f"**Chunk {j}** (Similarity: {doc['similarity']:.3f}, Source: {doc['metadata'].get('filename', 'Unknown')})")
                        st.text(doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'])
        
        # Input box
        user_question = st.chat_input("Enter your question...")
        
        if user_question:
            # Add to chat history (question)
            st.session_state.chat_history.append({
                'question': user_question,
                'answer': '',
                'retrieved_docs': []
            })
            
            # Display user question
            with st.chat_message("user"):
                st.write(user_question)
            
            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.rag_engine.query(user_question, top_k=3)
                    
                    # Update chat history
                    st.session_state.chat_history[-1]['answer'] = result['answer']
                    st.session_state.chat_history[-1]['retrieved_docs'] = result['retrieved_docs']
                    
                    # Display answer
                    st.write(result['answer'])
                    
                    # Display retrieval results
                    with st.expander(f"📄 View Retrieved Document Chunks ({len(result['retrieved_docs'])} chunks)"):
                        if result['retrieved_docs']:
                            for j, doc in enumerate(result['retrieved_docs'], 1):
                                st.markdown(f"**Chunk {j}** (Similarity: {doc['similarity']:.3f}, L2 Distance: {doc['score']:.3f})")
                                st.caption(f"Source: {doc['metadata'].get('filename', 'Unknown')}")
                                st.text(doc['text'])
                        else:
                            st.info("No relevant documents retrieved")
            
            st.rerun()


if __name__ == "__main__":
    main()
