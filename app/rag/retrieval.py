import uuid
import os
from typing import List, Dict, Any
from app.rag.storage import upload_file_to_s3
from app.rag.docling_client import process_document_with_docling_from_url
from app.rag.embedding import get_embedding
from app.db.vector_store import VectorStore

class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.vector_store.connect()
        self.vector_store.create_tables()

    def process_document(self, file_obj, filename: str) -> dict:
        """
        Complete pipeline: upload -> process -> embed -> store
        """
        try:
            # Step 1: Upload file to MinIO
            file_url = upload_file_to_s3(file_obj, filename)
            print(f"File uploaded to: {file_url}")
            
            # Step 2: Process document with Docling
            processed_text = process_document_with_docling_from_url(file_url)
            if not processed_text:
                raise Exception("Failed to process document with Docling")
            
            # Step 3: Generate embedding
            embedding = get_embedding(processed_text)
            if not embedding:
                raise Exception("Failed to generate embedding")

            # Step 4: Store in database
            document_id = str(uuid.uuid4())
            embedding_id = str(uuid.uuid4())
            
            # Store document metadata
            metadata = {
                "original_filename": filename,
                "file_size": len(processed_text),
                "processing_method": "docling"
            }
            
            self.vector_store.insert_document(
                document_id=document_id,
                filename=filename,
                file_url=file_url,
                content=processed_text,
                metadata=metadata
            )
            
            # Store embedding
            self.vector_store.insert_embedding(
                embedding_id=embedding_id,
                document_id=document_id,
                embedding=embedding
            )
            
            return {
                "document_id": document_id,
                "filename": filename,
                "file_url": file_url,
                "status": "success",
                "content_length": len(processed_text)
            }
            
        except Exception as e:
            print(f"Error in RAG pipeline: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def query_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Query the document store with a natural language question
        """
        try:
            # Step 1: Generate embedding for the query
            query_embedding = get_embedding(query)
            if not query_embedding:
                raise Exception("Failed to generate query embedding")
            
            # Step 2: Search for similar documents
            results = self.vector_store.search(query_embedding, top_k)
            
            # Step 3: Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "document_id": result["id"],
                    "filename": result["filename"],
                    "content": result["content"][:500] + "...",
                    "similarity_score": result["similarity"],  # Truncate for display
                    "full_content": result["content"],  # Full content for detailed view
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error querying documents: {e}")
            return []

    def get_document_by_id(self, document_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific document by ID
        """
        try:
            document = self.vector_store.get_document_by_id(document_id)
            if not document:
                return {"error": "Document not found"}
            return {
                "document_id": document["id"],
                "filename": document["filename"],
                "file_url": document["file_url"],
                "content": document["content"],
                "metadata": document["metadata"]
            }
        except Exception as e:
            return {"error": str(e)}

    def close(self):
        """
        Clean up database connections
        """
        if self.vector_store:
            self.vector_store.disconnect()

# Convenience functions for use in API endpoints
def process_uploaded_file(file_obj, filename: str) -> Dict[str, Any]:
    """
    Process a single uploaded file through the RAG pipeline
    """
    pipeline = RAGPipeline()
    try:
        result = pipeline.process_document(file_obj, filename)
        return result
    finally:
        pipeline.close()

def search_documents(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Search documents with a query
    """
    pipeline = RAGPipeline()
    try:
        results = pipeline.query_documents(query, top_k)
        return results
    finally:
        pipeline.close()