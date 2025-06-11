from fastapi import APIRouter, UploadFile, File, HTTPException
from app.rag.storage import upload_file_to_s3
from app.rag.retrieval import process_uploaded_file, search_documents, RAGPipeline

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
