from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.rag.retrieval import process_uploaded_file, search_documents, RAGPipeline
from app.models import DocumentList, DocumentResponse

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}
    
@router.post("/process")
async def process_file(file: UploadFile = File(...)):
    try:
        result = process_uploaded_file(file.file, file.filename)
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/query")
async def query_documents(q: str = Query(...), top_k: int = 3):
    try:
        results = search_documents(q, top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    pipeline = RAGPipeline()
    try:
        result = pipeline.get_document_by_id(document_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    finally:
        pipeline.close()

@router.get("/documents", response_model=DocumentList)
async def list_documents():
    pipeline = RAGPipeline()
    try:
        docs = pipeline.vector_store.list_documents()
        return DocumentList(documents=[
            DocumentResponse(
                document_id=doc["id"],
                filename=doc["filename"],
                file_url=doc["file_url"],
                created_at=doc["created_at"],
                metadata=doc["metadata"],
                content_preview=doc["content"][:200] if doc["content"] else None
            ) for doc in docs
        ])
    finally:
        pipeline.close()

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    pipeline = RAGPipeline()
    try:
        # Delete from S3/MinIO first
        doc = pipeline.vector_store.get_document_by_id(document_id)
        if doc:
            # Get S3 client and delete file
            from app.rag.storage import get_s3_client
            s3 = get_s3_client()
            bucket = os.getenv("MINIO_BUCKET")
            filename = doc["filename"]
            s3.delete_object(Bucket=bucket, Key=filename)
            
            # Then delete from database
            success = pipeline.vector_store.delete_document(document_id)
            if success:
                return {"status": "success"}
        raise HTTPException(status_code=404, detail="Document not found")
    finally:
        pipeline.close()



