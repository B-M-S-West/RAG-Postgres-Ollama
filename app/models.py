from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    file_url: str
    created_at: datetime
    metadata: Dict[str, Any]
    content_preview: Optional[str] = None

class DocumentList(BaseModel):
    documents: list[DocumentResponse]