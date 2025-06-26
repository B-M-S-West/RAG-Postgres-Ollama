import psycopg2
import psycopg2.extras
import json
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    def __init__(self):
        self.host = os.getenv("POSTGRES_HOST", "localhost")
        self.port = os.getenv("POSTGRES_PORT", "5432")
        self.db = os.getenv("POSTGRES_DB", "vector_db")
        self.user = os.getenv("POSTGRES_USER", "postgres")
        self.password = os.getenv("POSTGRES_PASSWORD", "password")
        self.conn = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.db,
                user=self.user,
                password=self.password
            )
            self.conn.autocommit = True
            return self.conn
        except psycopg2.Error as e:
            print(f"Error connecting to PostgreSQL: {e}")
            return None
        
    def disconnect(self):
        if self.conn:
            self.conn.close()

    def create_tables(self):
        if not self.conn:
            print("Connection not established.")
            return
        
        cur = self.conn.cursor()
        try:
            # Enhanced documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_url TEXT NOT NULL,
                    content TEXT,
                    doc_type TEXT DEFAULT 'general_document',
                    chunking_strategy TEXT DEFAULT 'fixed_size',
                    chunk_count INTEGER DEFAULT 0,
                    processing_time FLOAT DEFAULT 0.0,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT  CURRENT_TIMESTAMP,
                    update_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                        
                -- Enhanced chunks table for better chunk management
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id UUID PRIMARY KEY,
                    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    chunk_type TEXT DEFAULT 'content',
                    section_title TEXT,
                    word_count INTEGER DEFAULT 0,
                    char_count INTEGER DEFAULT 0,
                    priority TEXT DEFAULT 'normal',
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Embeddings table linked to chunks
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    id UUID PRIMARY KEY,
                    chunk_id UUID REFERENCES document_chunks(id) ON DELETE CASCADE,
                    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                    embedding vector(768),
                    embedding_model TEXT DEFAULT 'nomic-embed-text',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                        
                -- Create indexes for better performance
                CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);
                CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON document_chunks(chunk_index);
                CREATE INDEX IF NOT EXISTS idx_chunks_priority ON document_chunks(priority);
                CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON chunk_embeddings(document_id);
                
                -- Create a trigger to update updated_at timestamp
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.update_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                        
                DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
                CREATE TRIGGER update_documents_updated_at
                    BEFORE UPDATE ON documents
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column();
            """)
            self.conn.commit()
            print("Enhanced tables created successfully.")
        except psycopg2.Error as e:
            print(f"Error creating tables: {e}")

    def insert_document(self, document_id: str, filename: str, file_url: str, content: str, doc_type: str, chunking_strategy: str, chunk_count: int, processing_time: float, metadata: Dict):
        # Insert document with enhanced metadata
        if not self.conn:
            print("Connection not established.")
            return False
        
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO documents (id, filename, file_url, content, doc_type, chunking_strategy, chunk_count, processing_time, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (document_id, filename, file_url, content, doc_type, chunking_strategy, chunk_count, processing_time, json.dumps(metadata)))
            self.conn.commit()
            print(f"Document {filename} inserted successfully.")
        except psycopg2.Error as e:
            print(f"Error inserting document: {e}")
            return False
        
    def insert_chunk(self, chunk_id: str, document_id: str, chunk_index: int, text: str, chunk_type: str, section_title: Optional[str], word_count: int, char_count: int, priority: str, metadata: Dict):
        # Insert chunk with enhanced metadata
        if not self.conn:
            print("Connection not established.")
            return False
        
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO document_chunks (id, document_id, chunk_index, text, chunk_type, section_title, word_count, char_count, priority, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (chunk_id, document_id, chunk_index, text, chunk_type, section_title, word_count, char_count, priority, json.dumps(metadata)))
            self.conn.commit()
            print(f"Chunk {chunk_index} for document {document_id} inserted successfully.")
            return True
        except psycopg2.Error as e:
            print(f"Error inserting chunk: {e}")
            return False
        
    def insert_chunk_embedding(self, embedding_id: str, chunk_id: str, document_id: str, embedding: List[float], embedding_model: str = 'nomic-embed-text'):
        # Insert chunk embedding with enhanced metadata
        if not self.conn:
            print("Connection not established.")
            return False
        
        cur = self.conn.cursor()
        try:
            # Ensure embedding is a flat list of floats
            if hasattr(embedding, 'flatten'):
                embedding = embedding.flatten().tolist()
            elif isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                embedding = embedding[0]  # Handle nested lists
            
            cur.execute("""
                INSERT INTO chunk_embeddings (id, chunk_id, document_id, embedding, embedding_model)
                VALUES (%s, %s, %s, %s::vector, %s)
            """, (embedding_id, chunk_id, document_id, embedding, embedding_model))
            self.conn.commit()
            print(f"Embedding for chunk {chunk_id} inserted successfully.")
            return True
        except psycopg2.Error as e:
            print(f"Error inserting chunk embedding: {e}")
            return False
        
    def search(self, query_embedding: List[float], top_k: int = 3, filters: Optional[Dict] = None) -> List[Dict]:
        # Enhanced search with optional filters
        if not self.conn:
            print("Connection not established.")
            return []
        
        # Ensure embedding is a flat 1-D list
        if hasattr(query_embedding, 'flatten'):
            query_embedding = query_embedding.flatten().tolist()
        elif isinstance(query_embedding, list) and len(query_embedding) > 0 and isinstance(query_embedding[0], list):
            # Handle nested lists
            query_embedding = query_embedding[0]
        
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        # Build query with optional filtering
        base_query = """
            SELECT
                d.id as document_id,
                d.filename,
                d.doc_type,
                c.id as chunk_id,
                c.text,
                c.section_title,
                c.chunk_index,
                c.priority,
                c.word_count,
                c.metadata as chunk_metadata,
                d.metadata as document_metadata,
                (e.embedding <-> %s::vector) AS similarity_score
            FROM documents d
            JOIN document_chunks c ON d.id = c.document_id
            JOIN chunk_embeddings e ON c.id = e.chunk_id
        """

        params =[query_embedding]
        where_conditions = []

        # Apply filers
        if filters:
            if 'doc_type' in filters:
                where_conditions.append("d.doc_type = %s")
                params.append(filters['doc_type'])
            if 'chunk_type' in filters:
                where_conditions.append("c.chunk_type = %s")
                params.append(filters['chunk_type'])
            if 'priority' in filters:
                where_conditions.append("c.priority = %s")
                params.append(filters['priority'])
            if 'min_word_count' in filters:
                where_conditions.append("c.word_count >= %s")
                params.append(filters['min_word_count'])

        if where_conditions:
            base_query += " WHERE " + " AND ".join(where_conditions)

        base_query += """
            ORDER BY e.embedding <-> %s::vector
            LIMIT %s
        """
        params.extend([query_embedding, top_k])

        try:
            cur.execute(base_query, params)
            results = cur.fetchall()
            return [dict(result) for result in results]
        except psycopg2.Error as e:
            print(f"Error searching embeddings: {e}")
            return []
        
    def get_document_by_id(self, document_id: str) -> Optional[Dict]:
        """
        Retrieve a document by ID from the database.
        """
        if not self.conn:
            return None
        
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute("""
                SELECT id, filename, file_url, content, doc_type, chunking_strategy, chunk_count, processing_time, metadata, created_at, updated_at
                FROM documents
                WHERE id = %s;
            """, (document_id,))
            result = cur.fetchone()
            return dict(result) if result else None
        except psycopg2.Error as e:
            print(f"Error retrieving document: {e}")
            return None

    def get_document_chunks(self, document_id: str) -> List[Dict]:
        """
        Retrieve all chunks for a specific document by its ID.
        """
        if not self.conn:
            return []
        
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute("""
                SELECT id, chunk_index, text, chunk_type, section_title, word_count, char_count, priority, metadata
                FROM document_chunks
                WHERE document_id = %s
                ORDER BY chunk_index;
            """, (document_id,))
            results = cur.fetchall()
            return [dict(result) for result in results]
        except psycopg2.Error as e:
            print(f"Error retrieving document chunks: {e}")
            return []
        
    def list_documents(self, limit: int = 100, offset: int = 0, doc_type: Optional[str] = None) -> List[Dict]:
        """
        Get all documents with pagination and filtering
        """
        if not self.conn:
            return []
        
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        base_query = """
            SELECT id, filename, file_url, content, doc_type, chunking_strategy, chunk_count, processing_time, metadata, created_at, updated_at
                SUBSTRING(content, 1, 100) AS content_preview
            FROM documents
        """

        params = []
        if doc_type:
            base_query += " WHERE doc_type = %s"
            params.append(doc_type)

        base_query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        try:
            cur.execute(base_query, params)
            results = cur.fetchall()
            return [dict(result) for result in results]
        except psycopg2.Error as e:
            print(f"Error listing documents: {e}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and its associated chunks and embeddings by ID.
        """
        if not self.conn:
            return False
        
        cur = self.conn.cursor()
        try:
            # CASCADE will cause the deletion of associated chunks and embeddings
            cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            self.conn.commit()
            return cur.rowcount > 0
        except psycopg2.Error as e:
            print(f"Error deleting document: {e}")
            self.conn.rollback()
            return False
        
    def get_document_stats(self) -> Dict:
        """
        Get statistics about the documents in the database.
        """
        if not self.conn:
            return {}
        
        cur = self.conn.cursor()
        try:
            stats = {}

            # Document counts by type
            cur.execute("""
                SELECT doc_type, COUNT(*) as count
                FROM documents
                GROUP BY doc_type;
            """)
            stats['documents_by_type'] = dict(cur.fetchall())

            # Total document count
            cur.execute("SELECT COUNT(*) FROM documents;")
            stats['total_documents'] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            stats['total_chunks'] = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM chunk_embeddings;")
            stats['total_embeddings'] = cur.fetchone()[0]

            # Average processing time
            cur.execute("SELECT AVG(processing_time) FROM documents WHERE processing_time > 0;")
            result = cur.fetchone()
            stats['avg_processing_time'] = float(result[0]) if result[0] else 0.0

            return stats
        except psycopg2.Error as e:
            print(f"Error retrieving document stats: {e}")
            return {}
        
    def execute_query(self, query: str, params=None):
        """
        Execute a custom SQL query.
        """
        if not self.conn:
            print("Connection not established.")
            return None
        
        cur = self.conn.cursor()
        try:
            cur.execute(query, params)
            if cur.description:
                result = cur.fetchall()
                return result
            else:
                return None
        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            return None
        

if __name__ == "__main__":
    vector_store = VectorStore()
    if vector_store.connect():
        vector_store.create_tables()
        stats = vector_store.get_document_stats()
        print("Document Stats:", stats)
        vector_store.disconnect()