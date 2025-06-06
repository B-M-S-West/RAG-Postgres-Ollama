from fastapi import APIRouter, UploadFile, File, HTTPException
from app.rag.storage import upload_file_to_s3

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Upload the file to MinIO/S3
        file_url = upload_file_to_s3(file.file, file.filename)
        return {"filename": file.filename, "url": file_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))