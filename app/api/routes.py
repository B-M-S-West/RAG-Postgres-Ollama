from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from typing import Optional, List, Dict, Any
from app.rag.retrieval import process_uploaded_file, search_documents, generate_rag_answer, RAGPipeline
from app.rag.storage import get_s3_client
from app.models import DocumentList, DocumentResponse, QueryRequest, QueryResponse, RAGResponse, DocumentType, QueryResult
import os

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "ok", "service": "RAG API"}

@router.get("/stats")
async def get_database_stats():
    """
    Get statistics about the database, such as number of documents, total size, etc.
    """
    pipeline = RAGPipeline()
    try:
        stats = pipeline.get_database_stats()
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pipeline.close()
    
@router.post("/process")
async def process_file(
    file: UploadFile = File(...),
    chunk_size: int = Query(500, description="Size of text chunks"),
    chunk_overlap: int = Query(50, description="Overlap between text chunks")
    ):
    """
    Process a document through the RAG pipeline.
    """
    try:
        #Validate file type
        allowed_extensions = ['.pdf', '.docx', '.txt', '.html', '.pptx']
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file.filename is None:
            raise HTTPException(status_code=400, detail="Uploaded file has no filename")
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Allowed: {allowed_extensions}"
            )
        
        result = process_uploaded_file(
            file.file,
            file.filename,
            chunk_size,
            chunk_overlap
            )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        return result {
            "message": "File processed successfully",
            "document_id": result.get("document_id"),
            "filename": result["filename"],
            "doc_type": result["doc_type"],
            "chunking_strategy": result["chunking_strategy"],
            "total_chunks": result["total_chunks"],
            "stored_chunks": result["stored_chunks"],
            "processing_time": result["processing_time"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/query",response_model=QueryResponse)
async def query_documents_endpoints(request: QueryRequest):
    """
    Query documents with advanced filtering
    """
    try:
        # Convert filters
        filters = {}
        if request.filters:
            filters.update(request.filters)

        if request.doc_types:
            filters["doc_type"] = request.doc_types[0] # Assuming single doc_type for simplicity

        results = search_documents(
            request.query,
            request.top_k,
            filters
        )

        if results.get('error'):
            raise HTTPException(status_code=500,detail=results['error'])
        
        return QueryResponse(
            query=results['query'],
            results=[
                QueryResult(
                    document_id=r["document_id"],
                    filename=r["filename"],
                    chunk_id=r["chunk_id"],
                    content=r["content_preview"],
                    section_title=r.get("section_title"),
                    similarity_score=r["similarity_score"],
                    doc_type=DocumentType(r["doc_type"]),
                    metadata=r.get("chunk_metadata", {})
                )
                for r in results['results']
            ],
            total_results=results['total_results'],
            processing_time=results['processing_time']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/query")
async def query_documents_simple(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results to return"),
    doc_type: Optional[str] = Query(None, description="Filter by document type"),
    chunk_type: Optional[str] = Query(None, description="Filter by chunk type"),
    priority: Optional[str] = Query(None, description="Filter by priority")
):
    """
    Simple query endpoint for backward compatibility
    """
    try:
        filters = {}
        if doc_type:
            filters['doc_type'] = doc_type
        if chunk_type:
            filters['chunk_type'] = chunk_type
        if priority:
            filters['priority'] = priority
        
        results = search_documents(q, top_k, filters if filters else None)
        
        if results.get('error'):
            raise HTTPException(status_code=500, detail=results['error'])
        
        return {
            "query": q,
            "results": results['results'],
            "total_results": results['total_results'],
            "processing_time": results['processing_time'],
            "filters_applied": filters
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/rag", response_model=RAGResponse)
async def generate_answer(
    query: str = Body(..., embed=True),
    top_k: int = Body(5, embed=True),
    doc_types: Optional[List[DocumentType]] = Body(None, embed=True),
    filters: Optional[Dict[str, Any]] = Body(None, embed=True)
):
    """
    Complete RAG endpoint: retrieve and generate answer
    """
    try:
        # Prepare filters
        search_filters = filters or {}
        if doc_types:
            search_filters['doc_type'] = doc_types[0].value
        
        result = generate_rag_answer(query, top_k, search_filters if search_filters else None)
        
        if result.get('error'):
            raise HTTPException(status_code=500, detail=result['error'])
        
        return RAGResponse(
            answer=result['answer'],
            sources=result['sources'],
            context_chunks=result['context_chunks'],
            confidence_score=result.get('confidence_score'),
            processing_time=result['processing_time']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """
    Get a specific document with its chunks
    """
    pipeline = RAGPipeline()
    try:
        result = pipeline.get_document_by_id(document_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    finally:
        pipeline.close()

@router.get("/documents", response_model=DocumentList)
async def list_documents(
    limit: int = Query(100, description="Number of documents to return"),
    offset: int = Query(0, description="Offset for pagination"),
    doc_type: Optional[str] = Query(None, description="Filter by document type")
):
    """
    List documents with pagination and filtering
    """
    pipeline = RAGPipeline()
    try:
        docs = pipeline.vector_store.list_documents(limit, offset, doc_type)
        
        return DocumentList(documents=[
            DocumentResponse(
                document_id=doc["id"],
                filename=doc["filename"],
                file_url=doc["file_url"],
                created_at=doc["created_at"],
                metadata=doc["metadata"],
                content_preview=doc.get("content_preview"),
                doc_type=DocumentType(doc["doc_type"]) if doc.get("doc_type") else None,
                chunk_count=doc.get("chunk_count")
            ) for doc in docs
        ])
    finally:
        pipeline.close()

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks/embeddings
    """
    pipeline = RAGPipeline()
    try:
        # Get document info first
        doc = pipeline.vector_store.get_document_by_id(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        # Delete from S3/MinIO first
        try:
            s3 = get_s3_client()
            bucket = os.getenv("MINIO_BUCKET")
            filename = doc["filename"]
            s3.delete_object(Bucket=bucket, Key=filename)
        except Exception as e:
            print(f"Warning: Could not delete file from storage: {e}")
            
            # Then delete from database
            success = pipeline.vector_store.delete_document(document_id)
            if success:
                return {
                    "status": "success",
                    "message": f"Document {doc['filename']} deleted successfully"
                    }
            else:
                raise HTTPException(status_code=404, detail="Failed to delete document from database")
    finally:
        pipeline.close()

@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str):
    """
    Get all chunks for a specific document
    """
    pipeline = RAGPipeline()
    try:
        chunks = pipeline.vector_store.get_document_chunks(document_id)
        return {
            "document_id": document_id,
            "chunks": chunks,
            "total_chunks": len(chunks)
        }
    finally:
        pipeline.close()

@router.get("/search/similar/{document_id}")
async def find_similar_documents(
    document_id: str,
    top_k: int = Query(5, description="Number of similar documents to return")
):
    """
    Find documents similar to a given document
    """
    pipeline = RAGPipeline()
    try:
        # Get the document
        doc = pipeline.vector_store.get_document_by_id(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Use document content as query
        content_preview = doc['content'][:500]  # Use first 500 chars as query
        
        results = pipeline.query_documents(content_preview, top_k + 1)  # +1 to exclude self
        
        # Filter out the original document
        filtered_results = [
            r for r in results['results'] 
            if r['document_id'] != document_id
        ][:top_k]
        
        return {
            "source_document_id": document_id,
            "source_filename": doc['filename'],
            "similar_documents": filtered_results,
            "total_found": len(filtered_results)
        }
        
    finally:
        pipeline.close()

