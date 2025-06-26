from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from typing import Optional, List, Dict, Any
from app.rag.retrieval import process_uploaded_file, search_documents, generate_rag_answer, RAGPipeline
from app.rag.storage import get_s3_client
from app.models import DocumentList, DocumentResponse, QueryRequest, QueryResponse, RAGResponse, DocumentType, QueryResult
from app.utils.logger import setup_logging
import os
import logging

setup_logging()
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    logger.info("Health check endpoint accessed")
    return {"status": "ok", "service": "RAG API"}

@router.get("/stats")
async def get_database_stats():
    """
    Get statistics about the database, such as number of documents, total size, etc.
    """
    logger.info("Database stats requested")
    pipeline = RAGPipeline()
    try:
        stats = pipeline.get_database_stats()
        logger.info(f"Database stats retrieved: {stats}")
        return {"stats": stats}
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
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
    logger.info(f"File upload received: {file.filename}")
    try:
        #Validate file type
        allowed_extensions = ['.pdf', '.docx', '.txt', '.html', '.pptx']
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file.filename is None:
            logger.warning("Uploaded file has no filename")
            raise HTTPException(status_code=400, detail="Uploaded file has no filename")
        
        if file_extension not in allowed_extensions:
            logger.warning(f"Unsupported file type: {file_extension}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Allowed: {allowed_extensions}"
            )
        logger.info(f"Processing {file.filename} with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

        result = process_uploaded_file(
            file.file,
            file.filename,
            chunk_size,
            chunk_overlap
            )
        
        if result.get("status") == "error":
            logger.error(f"Processing failed for {file.filename}: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
        logger.info(f"Successfully processed {file.filename}")
        return {
            **result,
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
        logger.error(f"Unexpected error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/query/advanced",response_model=QueryResponse)
async def query_documents_endpoints(request: QueryRequest):
    """
    Query documents with advanced filtering
    """
    logger.info(f"Advanced query received: '{request.query}'")
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
            logger.error(f"Query failed: {results['error']}")
            raise HTTPException(status_code=500,detail=results['error'])
        
        logger.info(f"Query completed successfully, found {results['total_results']} results")
        
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
        logger.error(f"Unexpected error in advanced query: {str(e)}")
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
    logger.info(f"Simple query received: '{q}'")
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
            logger.error(f"Simple query failed: {results['error']}")
            raise HTTPException(status_code=500, detail=results['error'])
        
        logger.info(f"Simple query completed, found {results['total_results']} results")

        return {
            "query": q,
            "results": results['results'],
            "total_results": results['total_results'],
            "processing_time": results['processing_time'],
            "filters_applied": filters
        }
        
    except Exception as e:
        logger.error(f"Error in simple query: {str(e)}")
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
    logger.info(f"RAG request received: '{query}'")
    try:
        # Prepare filters
        search_filters = filters or {}
        if doc_types:
            search_filters['doc_type'] = doc_types[0].value
        
        result = generate_rag_answer(query, top_k, search_filters if search_filters else None)
        
        if result.get('error'):
            logger.error(f"RAG generation failed: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])
        
        logger.info("RAG answer generated successfully")
        
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
        logger.error(f"Unexpected error in RAG generation: {str(e)}")
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
    logger.info(f"Delete request for document: {document_id}")
    pipeline = RAGPipeline()
    try:
        # Get document info first
        doc = pipeline.vector_store.get_document_by_id(document_id)
        if not doc:
            logger.warning(f"Document not found for deletion: {document_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        # Delete from S3/MinIO first
        try:
            s3 = get_s3_client()
            bucket = os.getenv("MINIO_BUCKET")
            filename = doc["filename"]
            s3.delete_object(Bucket=bucket, Key=filename)
            logger.info(f"File deleted from storage: {filename}")
        except Exception as e:
            logger.warning(f"Could not delete file from storage: {e}")
            
            # Then delete from database
        success = pipeline.vector_store.delete_document(document_id)
        if success:
            logger.info(f"Document deleted successfully: {doc['filename']}")
            return {
                "status": "success",
                "message": f"Document {doc['filename']} deleted successfully"
                }
        else:
            logger.error(f"Failed to delete document from database: {document_id}")
            raise HTTPException(status_code=404, detail="Failed to delete document from database")
    finally:
        pipeline.close()

@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str):
    """
    Get all chunks for a specific document
    """
    logger.info(f"Request for document chunks: {document_id}")
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
    logger.info(f"Find similar documents to: {document_id}")
    pipeline = RAGPipeline()
    try:
        # Get the document
        doc = pipeline.vector_store.get_document_by_id(document_id)
        if not doc:
            logger.error(f"Document not found: {document_id}")
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

