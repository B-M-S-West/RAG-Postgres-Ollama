import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

BACKEND_URL = "http://localhost:8001"

st.set_page_config(
    page_title="RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🔍 RAG Document System")
    st.markdown("Upload, process, and query documents with advanced AI capabilities")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📤 Upload Documents", "🔍 Search & Query", "📊 Analytics", "📚 Document Library", "⚙️ System Status"]
    )

    if page == "📤 Upload Documents":
        upload_page()
    elif page == "🔍 Search & Query":
        search_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "📚 Document Library":
        library_page()
    elif page == "⚙️ System Status":
        status_page()

def upload_page():
    st.header("📤 Upload Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload New Document")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt', 'md', 'html'],
            help="Supported formats: PDF, DOCX, TXT, MD, HTML"
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Processing Options"):
            chunk_size = st.slider("Chunk Size (words)", 100, 1000, 500, 50)
            chunk_overlap = st.slider("Chunk Overlap (words)", 0, 200, 50, 10)
            
            st.info(f"""
            **Chunk Size**: {chunk_size} words per chunk
            **Overlap**: {chunk_overlap} words overlap between chunks
            
            Larger chunks preserve more context but may be less precise for retrieval.
            Smaller chunks are more precise but may lose context.
            """)
        
        if uploaded_file is not None:
            st.write(f"**File**: {uploaded_file.name}")
            st.write(f"**Size**: {uploaded_file.size:,} bytes")
            
            if st.button("🚀 Process Document", type="primary"):
                process_document(uploaded_file, chunk_size, chunk_overlap)
    
    with col2:
        st.subheader("📋 Processing Tips")
        st.info("""
        **Best Practices:**
        
        📄 **PDFs**: Work best with text-based content
        
        📝 **Text Files**: Fastest processing
        
        📊 **Reports**: Use larger chunks (800-1000 words)
        
        📚 **Manuals**: Use smaller chunks (300-500 words)
        
        🎓 **Academic Papers**: Medium chunks (500-700 words)
        """)
        
        # Recent uploads
        show_recent_uploads()

def process_document(uploaded_file, chunk_size: int, chunk_overlap: int):
    """Process uploaded document"""
    with st.spinner("Processing document... This may take a few minutes."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            params = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            
            response = requests.post(
                f"{BACKEND_URL}/process",
                files=files,
                params=params,
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("✅ Document processed successfully!")
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Document Type", result.get('doc_type', 'Unknown'))
                with col2:
                    st.metric("Total Chunks", result.get('total_chunks', 0))
                with col3:
                    st.metric("Stored Chunks", result.get('stored_chunks', 0))
                with col4:
                    st.metric("Processing Time", f"{result.get('processing_time', 0):.1f}s")
                
                # Show processing details
                with st.expander("📊 Processing Details"):
                    st.json(result)
                
                st.info("💡 Your document is now ready for searching! Go to the Search & Query page.")
                
            else:
                st.error(f"❌ Error processing document: {response.text}")
                
        except requests.exceptions.Timeout:
            st.error("⏰ Processing timed out. Please try with a smaller document or contact support.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def search_page():
    st.header("🔍 Search & Query Documents")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your question or search query:",
            placeholder="e.g., What are the key findings about machine learning?",
            help="Ask questions in natural language or use keywords"
        )
    
    with col2:
        search_type = st.selectbox("Search Type", ["🤖 RAG Answer", "🔍 Document Search"])
    
    # Advanced filters
    with st.expander("🎛️ Advanced Filters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            doc_type_filter = st.selectbox(
                "Document Type",
                ["All", "resume", "academic_paper", "business_report", "manual", "general_document"]
            )
        
        with col2:
            chunk_type_filter = st.selectbox(
                "Chunk Type",
                ["All", "semantic", "fixed_size", "sentence", "paragraph"]
            )
        
        with col3:
            top_k = st.slider("Number of Results", 1, 20, 5)
    
    if query and st.button("🔍 Search", type="primary"):
        if search_type == "🤖 RAG Answer":
            perform_rag_search(query, doc_type_filter, top_k)
        else:
            perform_document_search(query, doc_type_filter, chunk_type_filter, top_k)

def perform_rag_search(query: str, doc_type_filter: str, top_k: int):
    """Perform RAG search with answer generation"""
    with st.spinner("🤖 Generating answer..."):
        try:
            payload = {
                "query": query,
                "top_k": top_k
            }
            
            if doc_type_filter != "All":
                payload["doc_types"] = [doc_type_filter]
            
            response = requests.post(f"{BACKEND_URL}/rag", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display answer
                st.subheader("🤖 AI Generated Answer")
                st.write(result['answer'])
                
                # Display metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sources Used", len(result['sources']))
                with col2:
                    st.metric("Context Chunks", result['context_chunks'])
                with col3:
                    st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                
                # Show sources
                if result['sources']:
                    st.subheader("📚 Sources")
                    for i, source in enumerate(result['sources'], 1):
                        st.write(f"{i}. {source}")
                
            else:
                st.error(f"❌ Error: {response.text}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def perform_document_search(query: str, doc_type_filter: str, chunk_type_filter: str, top_k: int):
    """Perform document search"""
    with st.spinner("🔍 Searching documents..."):
        try:
            params = {"q": query, "top_k": top_k}
            
            if doc_type_filter != "All":
                params["doc_type"] = doc_type_filter
            if chunk_type_filter != "All":
                params["chunk_type"] = chunk_type_filter
            
            response = requests.get(f"{BACKEND_URL}/query", params=params)
            
            if response.status_code == 200:
                result = response.json()
                
                st.subheader(f"🔍 Search Results ({result['total_results']} found)")
                
                if result['results']:
                    for i, res in enumerate(result['results'], 1):
                        with st.expander(f"Result {i}: {res['filename']} (Score: {res['similarity_score']:.3f})"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write("**Content:**")
                                st.write(res['content_preview'])
                                
                                if res.get('section_title'):
                                    st.write(f"**Section:** {res['section_title']}")
                            
                            with col2:
                                st.write(f"**Document Type:** {res['doc_type']}")
                                st.write(f"**Chunk Type:** {res['chunk_type']}")
                                st.write(f"**Word Count:** {res['word_count']}")
                                st.write(f"**Similarity:** {res['similarity_score']:.3f}")
                else:
                    st.info("No results found. Try different keywords or adjust filters.")
                
            else:
                st.error(f"❌ Error: {response.text}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def analytics_page():
    st.header("📊 Analytics Dashboard")
    
    # Get database stats
    try:
        response = requests.get(f"{BACKEND_URL}/stats")
        if response.status_code == 200:
            stats = response.json()['stats']
            
            # Overview metrics
            st.subheader("📈 Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", stats.get('total_documents', 0))
            with col2:
                st.metric("Total Chunks", stats.get('total_chunks', 0))
            with col3:
                st.metric("Total Embeddings", stats.get('total_embeddings', 0))
            with col4:
                avg_time = stats.get('avg_processing_time', 0)
                st.metric("Avg Processing Time", f"{avg_time:.1f}s")
            
            # Document type distribution
            if 'documents_by_type' in stats and stats['documents_by_type']:
                st.subheader("📊 Document Types")
                
                doc_types = stats['documents_by_type']
                df = pd.DataFrame(list(doc_types.items()), columns=['Document Type', 'Count'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(df, values='Count', names='Document Type', 
                                title="Document Distribution by Type")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(df, x='Document Type', y='Count',
                                title="Document Count by Type")
                    st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Failed to load analytics data")
            
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")

def library_page():
    st.header("📚 Document Library")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        doc_type_filter = st.selectbox(
            "Filter by Type",
            ["All", "resume", "academic_paper", "business_report", "manual", "general_document"],
            key="lib_doc_type"
        )
    with col2:
        limit = st.selectbox("Documents per page", [10, 25, 50, 100], index=1)
    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Load documents
    try:
        params = {"limit": limit}
        if doc_type_filter != "All":
            params["doc_type"] = doc_type_filter
        
        response = requests.get(f"{BACKEND_URL}/documents", params=params)
        
        if response.status_code == 200:
            result = response.json()
            documents = result['documents']
            
            if documents:
                st.subheader(f"📄 Documents ({len(documents)} shown)")
                
                for doc in documents:
                    with st.expander(f"📄 {doc['filename']} ({doc.get('doc_type', 'unknown')})"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Uploaded:** {doc['created_at']}")
                            if doc.get('content_preview'):
                                st.write("**Preview:**")
                                st.write(doc['content_preview'])
                            
                            if doc.get('chunk_count'):
                                st.write(f"**Chunks:** {doc['chunk_count']}")
                        
                        with col2:
                            if st.button("🗑️ Delete", key=f"del_btn_{doc['document_id']}"):
                                try:
                                    response = requests.delete(f"{BACKEND_URL}/documents/{doc['document_id']}")
                                    if response.status_code == 200:
                                        st.success(f"✅ Deleted '{doc['filename']}' successfully!")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed to delete document: {response.text}")
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                            if st.button(f"🔍 Find Similar", key=f"sim_{doc['document_id']}"):
                                find_similar_documents(doc['document_id'])
            else:
                st.info("No documents found. Upload some documents to get started!")
                
        else:
            st.error("Failed to load documents")
            
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")

def find_similar_documents(document_id: str):
    """Find similar documents"""
    try:
        response = requests.get(f"{BACKEND_URL}/search/similar/{document_id}")
        if response.status_code == 200:
            result = response.json()
            
            st.subheader(f"📄 Documents similar to '{result['source_filename']}'")
            
            if result['similar_documents']:
                for doc in result['similar_documents']:
                    st.write(f"• **{doc['filename']}** (Score: {doc['similarity_score']:.3f})")
                    st.write(f"  {doc['content_preview'][:100]}...")
            else:
                st.info("No similar documents found.")
        else:
            st.error("Failed to find similar documents")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def status_page():
    st.header("⚙️ System Status")
    
    # API Health Check
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API Server: Online")
        else:
            st.error("❌ API Server: Error")
    except:
        st.error("❌ API Server: Offline")
    
    # Database Stats
    try:
        response = requests.get(f"{BACKEND_URL}/stats")
        if response.status_code == 200:
            st.success("✅ Database: Connected")
            stats = response.json()['stats']
            
            with st.expander("📊 Database Statistics"):
                st.json(stats)
        else:
            st.error("❌ Database: Error")
    except:
        st.error("❌ Database: Connection Failed")
    
    # System Information
    st.subheader("🖥️ System Information")
    st.info(f"""
    **Backend URL:** {BACKEND_URL}
    **Frontend:** Streamlit
    **Backend:** FastAPI
    **Database:** PostgreSQL with pgvector
    **Embeddings:** Ollama (nomic-embed-text)
    **Document Processing:** Docling
    """)

def show_recent_uploads():
    """Show recent uploads in sidebar"""
    try:
        response = requests.get(f"{BACKEND_URL}/documents", params={"limit": 5})
        if response.status_code == 200:
            documents = response.json()['documents']
            if documents:
                st.subheader("📋 Recent Uploads")
                for doc in documents[:3]:  # Show only 3 most recent
                    st.write(f"• {doc['filename']}")
                    st.caption(f"  {doc.get('doc_type', 'unknown')} • {doc['created_at'][:10]}")
    except:
        pass  # Silently fail for sidebar

if __name__ == "__main__":
    main()

