from fastapi import FastAPI, APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "ok"}

