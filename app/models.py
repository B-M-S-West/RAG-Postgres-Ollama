from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

class DocumentType(str, Enum):
    RESUME = "resume"
    ACADEMIC_PAPER = "academic_paper"
    BUSINESS_REPORT = "business_report"
    MANUAL = "manual"
    GENERAL_DOCUMENT = "general_document"

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
    doc_type: DocumentType
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
    doc_type: Optional[DocumentType] = None
    chunk_count: Optional[int] = None

class DocumentList(BaseModel):
    documents: List[DocumentResponse]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    doc_types: Optional[List[DocumentType]] = None

class QueryResult(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    content: str
    section_title: Optional[str] = None
    similarity_score: float
    doc_type: DocumentType
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

