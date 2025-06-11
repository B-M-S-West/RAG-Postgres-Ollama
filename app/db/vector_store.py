import psycopg2
import psycopg2.extras
import json
import numpy as np
import os
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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    filename TEXT,
                    file_url TEXT,
                    content TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS embeddings (
                    id UUID PRIMARY KEY,
                    document_id UUID REFERENCES documents(id),
                    embedding vector(768),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.conn.commit()
            print("Tables created successfully.")
        except psycopg2.Error as e:
            print(f"Error creating tables: {e}")

    def insert_document(self, document_id, filename, file_url, content, metadata):
        if not self.conn:
            print("Connection not established.")
            return
        
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO documents (id, filename, file_url, content, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (document_id, filename, file_url, content, json.dumps(metadata)))
            self.conn.commit()
            print("Document inserted successfully.")
        except psycopg2.Error as e:
            print(f"Error inserting document: {e}")

    def insert_embedding(self, embedding_id, document_id, embedding):
        if not self.conn:
            print("Not connected to the database.")
            return
        
        cur = self.conn.cursor()
        try:
            # Ensure embedding is a flat list of floats
            if hasattr(embedding, 'flatten'):
                embedding = embedding.flatten().tolist()
            elif isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                embedding = embedding[0]  # Handle nested lists
            cur.execute("""
                INSERT INTO embeddings (id, document_id, embedding)
                VALUES (%s, %s, %s)
            """, (embedding_id, document_id, embedding))
            self.conn.commit()
            print("Embedding inserted successfully.")
        except psycopg2.Error as e:
            print(f"Error inserting embedding: {e}")

    def search(self, query_embedding, top_k=3):
        if not self.conn:
            print("Connection not established.")
            return []
        
        cur = self.conn.cursor()
        try:
            cur.execute("""
                SELECT d.id, d.filename, d.content, (e.embedding <-> %s) AS similarity
                FROM documents d
                JOIN embeddings e ON d.id = e.document_id
                ORDER BY e.embedding <-> %s
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))
            results = cur.fetchall()
            return results
        except psycopg2.Error as e:
            print(f"Error searching embeddings: {e}")
            return []
        
    def execute_query(self, query, params=None):
        """
        Execture a SQL query and fetch the results.
        """
        if not self.conn:
            print("Connection not established.")
            return None
        
        cur = self.conn.cursor()
        try:
            cur.execute(query, params)
            if cur.description:
                result = cur.fetchone()
                return result
            else:
                return None
        except psycopg2.Error as e:
            print(f"Error executing query: {e}")
            return None
        
    def get_document_by_id(self, document_id: str) -> dict:
        """
        Retrieve a document by ID from database.
        """
        query = """
        SELECT id, filename, file_url, content, metadata, created_at
        FROM documents
        WHERE id = %s;
        """

        result = self.execute_query(query, (document_id,))
        if result:
            return {
                "id": result[0],
                "filename": result[1],
                "file_url": result[2],
                "content": result[3],
                "metadata": result[4],
                "created_at": result[5]
            }
        return None
    
    def get_embedding_by_document_id(self, document_id: str):
        """
        Retrieve embeddings for a specific document by its ID.
        """
        query = """
        SELECT embedding
        FROM embeddings
        WHERE document_id = %s;
        """

        result = self.execute_query(query, (document_id,))
        return result[0] if result else None


if __name__ == "__main__":
    vector_store = VectorStore()
    if vector_store.connect():
        vector_store.create_tables()
        vector_store.disconnect()