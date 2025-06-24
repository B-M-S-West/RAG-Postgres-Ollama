from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

class ChunkingStrategy(str, Enum):
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    RECURSIVE = "recursive"

class DocumentChunk(BaseModel):
    id: str
    document_id: str
    text: str
    chunk_index: int
    chunk_type: str
    section_title: Optional[str] = None
    word_count: int
    char_count: int
    metadata: Dict[str, Any] = {}

class ProcessedDocument(BaseModel):
    document_id: str
    filename: str
    file_url: str
    content: str
    chunking_strategy: ChunkingStrategy
    chunks: List[DocumentChunk]
    processing_time: float
    created_at: datetime
    metadata: Dict[str, Any] = {}

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    file_url: str
    created_at: datetime
    metadata: Dict[str, Any]
    content_preview: Optional[str] = None
    chunk_count: Optional[int] = None

class DocumentList(BaseModel):
    documents: List[DocumentResponse]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None

class QueryResult(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    content: str
    section_title: Optional[str] = None
    similarity_score: float
    metadata: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    query: str
    results: List[QueryResult]
    total_results: int
    processing_time: float

class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    context_chunks: int
    confidence_score: Optional[float] = None
    processing_time: float
