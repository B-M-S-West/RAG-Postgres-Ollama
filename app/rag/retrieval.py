import os
import uuid
import time
from typing import List, Dict, Any, Optional
from app.rag.storage import upload_file_to_s3
from app.rag.docling_client import process_document_with_docling_from_url
from app.rag.embedding import get_embedding
from app.db.vector_store import VectorStore
from app.rag.chunking import chunk_document
from app.utils.logger import get_logger

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.vector_store.connect()
        self.vector_store.create_tables()
        logger.info("RAG Pipeline initialised")

    def process_document(self, file_obj, filename: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Dict:
        """
        Complete enhanced pipeline: upload -> process -> chunk -> embed -> store
        """
        start_time = time.time()
        logger.info(f"Starting document processing for: {filename}")

        try:
            # Step 1: Upload file to MinIO
            file_url = upload_file_to_s3(file_obj, filename)
            logger.info(f"File uploaded to: {file_url}")
            
            # Step 2: Process document with Docling
            processed_content = process_document_with_docling_from_url(file_url)
            if not processed_content:
                raise Exception("Failed to process document with Docling")
            logger.info(f"Document processed successfully, content length: {len(processed_content)}")

            # Step 3: Chunk the processed content
            chunking_result = chunk_document(
                processed_content,
                filename,
                chunk_size,
                chunk_overlap
            )

            chunks = chunking_result['chunks']
            doc_type = chunking_result['doc_type'].value
            chunking_strategy = chunking_result['chunking_strategy'].value
            logger.info(f"Created {len(chunks)} chunks using {chunking_strategy} strategy")

            # Step 4: Generate embeddings for each chunk
            embedded_chunks = []
            
            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i+1}/{len(chunks)}")

                # Generate embedding for the chunk
                embedding = get_embedding(chunk['text'])
                if not embedding:
                    logger.warning(f"Failed to generate embedding for chunk {i+1}")
                    continue

                chunk['embedding'] = embedding
                embedded_chunks.append(chunk)

                if not embedded_chunks:
                    raise Exception("No valid chunks with embeddings generated")
                logger.info(f"Successfully generated embeddings for {len(embedded_chunks)}/{len(chunks)} chunks")

            # Step 5: Store in database
            logger.info(f"Storing document and {len(embedded_chunks)} chunks in vector store")
            document_id = str(uuid.uuid4())
            
            # Store document metadata
            processing_time = time.time() - start_time
            metadata = {
                "original_filename": filename,
                "content_length": len(processed_content),
                "processing_method": "docling",
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "successful_chunks": len(embedded_chunks),
                "failed_chunks": len(chunks) - len(embedded_chunks),
            }

            # Insert document
            success = self.vector_store.insert_document(
                document_id=document_id,
                filename=filename,
                file_url=file_url,
                content=processed_content,
                doc_type=doc_type,
                chunking_strategy=chunking_strategy,
                chunk_count=len(embedded_chunks),
                processing_time=processing_time,
                metadata=metadata
            )

            if not success:
                raise Exception("Failed to insert document into vector store")
            
            # Insert embeddings for each chunk
            stored_chunks = 0
            for chunk in embedded_chunks:
                chunk_id = str(uuid.uuid4())
                embedding_id = str(uuid.uuid4())

                # Store chunk
                chunk_success = self.vector_store.insert_chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=chunk['text'],
                    chunk_type=chunk['chunk_type'],
                    section_title=chunk.get('section_title'),
                    word_count=chunk['word_count'],
                    char_count=chunk['char_count'],
                    chunk_index=chunk['chunk_index'],
                    priority=chunk.get('priority', 'normal'),
                    metadata=chunk.get('metadata', {})
                )

                if chunk_success:
                    # Store embedding
                    embedding_success = self.vector_store.insert_chunk_embedding(
                        embedding_id=embedding_id,
                        chunk_id=chunk_id,
                        document_id=document_id,
                        embedding=chunk['embedding']
                    )
                    
                    if embedding_success:
                        stored_chunks += 1
                    else:
                        logger.error(f"Failed to store embedding for chunk {chunk['chunk_index']}")
                else:
                    logger.error(f"Failed to store chunk {chunk['chunk_index']}")

            logger.info(f"Successfully stored {stored_chunks}/{len(embedded_chunks)} chunks with embeddings")
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")

            return {
                "document_id": document_id,
                "filename": filename,
                "file_url": file_url,
                "doc_type": doc_type,
                "chunking_strategy": chunking_strategy,
                "total_chunks": len(chunks),
                "stored_chunks": stored_chunks,
                "processing_time": processing_time,
                "status": "success",
                "content_length": len(processed_content)
            }
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": processing_time,
                "filename": filename,
                "file_url": file_url
            }


    def query_documents(self, query: str, top_k: int = 3, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Enhanced query with filtering and better result formatting
        """
        try:
            start_time = time.time()
            logger.info(f"Searching for documents with query: '{query}' (top_k={top_k})")
            # Step 1: Generate embedding for the query
            query_embedding = get_embedding(query)
            if not query_embedding:
                raise Exception("Failed to generate query embedding")
            
            # Step 2: Search for similar chunks
            logger.debug(f"Searching for top {top_k} similar chunks")
            results = self.vector_store.search(query_embedding, top_k=top_k, filters=filters)
            
            # Step 3: Format and enhance results
            formatted_results = []
            for result in results:
                formatted_result = {
                    "document_id": result["document_id"],
                    "chunk_id": result["chunk_id"],
                    "filename": result["filename"],
                    "doc_type": result["doc_type"],
                    "section_title": result.get("section_title"),
                    "chunk_type": result.get("chunk_type"),
                    "priority": result["priority"],
                    "content": result["text"],
                    "content_preview": result["text"][:300] + "..." if len(result["text"]) > 300 else result["text"],  # Preview first 300 chars
                    "word_count": result["word_count"],
                    "similarity_score": float(result["similarity_score"]),
                    "chunk_metadata": result.get("metadata", {}),
                    "document_metadata": result.get("document_metadata", {})
                }
                formatted_results.append(formatted_result)

            processing_time = time.time() - start_time
            logger.info(f"Query completed in {processing_time:.2f} seconds, found {len(formatted_results)} results")

            return {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "processing_time": processing_time,
                "filters_applied": filters or {}
            }
            
        except Exception as e:
            logger.error(f"Error querying documents: {e}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "processing_time": 0,
                "error": str(e),
            }

    def get_document_by_id(self, document_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific document by ID
        """
        try:
            logger.debug(f"Retrieving document by ID: {document_id}")
            document = self.vector_store.get_document_by_id(document_id)
            if not document:
                logger.warning(f"Document not found: {document_id}")
                return {"error": "Document not found"}
            logger.info(f"Successfully retrieved document: {document['filename']}")
            return {
                "document_id": document["id"],
                "filename": document["filename"],
                "file_url": document["file_url"],
                "content": document["content"],
                "metadata": document["metadata"]
            }
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {e}")
            return {"error": str(e)}

    def generate_answer(self, query: str, context_chunks: List[Dict], max_context_length: int = 4000) -> Dict:
        """
        Generate an answer using retrieved context
        """
        try:
            logger.info(f"Generating answer for query with {len(context_chunks)} context chunks")
            # Build context from chunks
            context_parts = []
            total_length = 0

            for i, chunk in enumerate(context_chunks):
                chunk_text = chunk['content']
                source_info = f"[Source: {i+1}: {chunk['filename']}]"

                if total_length + len(chunk_text) > max_context_length: break

                context_parts.append(chunk_text)
                total_length += len(chunk_text)

            context = "\n\n".join(context_parts)
            logger.debug(f"Built context with {len(context_parts)} chunks, total length: {total_length}")

            # Add Ollama interation here
            # ollama_response = ollama.chat(

            answer = f"""based on the provided context, here's what I found regarding your query: "{query}"

            Context Summary:
            - Found {len(context_chunks)} relevant chunks.
            - Sources: {', '.join([chunk['filename'] for chunk in context_chunks])}
            - Document types: {', '.join(set(chunk['doc_type'] for chunk in context_chunks))}
            [Placeholder response before Ollama integration]

            Context used:
            {context[:1000]}{'...' if len(context) > 1000 else ''}
            """

            logger.info("Answer generated successfully")
            return {
                "answer": answer,
                "sources": list(set(chunk['filename'] for chunk in context_chunks)),
                "context_chunks": len(context_parts),
                "total_context_length": total_length,
                "confidence": 0.8,  # Placeholder
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "context_chunks": 0,
                "total_context_length": 0,
                "confidence": 0.0,
            }
                
    def get_database_stats(self) -> Dict:
        """
        Retrieve statistics about the vector store
        """
        try:
            logger.debug("Retrieving database statistics")
            stats = self.vector_store.get_document_stats()
            logger.info(f"Database stats retrieved: {stats.get('total_documents', 0)} documents, {stats.get('total_chunks', 0)} chunks")
            return stats
        except Exception as e:
            logger.error(f"Error retrieving database stats: {e}")
            return {"error": str(e)}
        
    def delete_document(self, document_id: str) -> Dict:
        """
        Delete a document and its associated chunks from the vector store
        """
        try:
            logger.info(f"Deleting document: {document_id}")
            success = self.vector_store.delete_document(document_id)
            if not success:
                logger.warning(f"Failed to delete document: {document_id}")
                return {"error": "Failed to delete document"}
            logger.info(f"Successfully deleted document: {document_id}")
            return {"status": "success", "message": "Document deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return {"error": str(e)}
        
    def close(self):
        """
        Close the database connection
        """
        if self.vector_store:
            logger.info("Closing RAG Pipeline connections")
            self.vector_store.disconnect()


# Convenience functions for use in API endpoints
def process_uploaded_file(file_obj, filename: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Dict[str, Any]:
    """
    Process a single uploaded file through the RAG pipeline
    """
    pipeline = RAGPipeline()
    try:
        result = pipeline.process_document(file_obj, filename, chunk_size, chunk_overlap)
        return result
    finally:
        pipeline.close()

def search_documents(query: str, top_k: int = 3, filters: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Search for documents using enhanced filtering
    """
    pipeline = RAGPipeline()
    try:
        results = pipeline.query_documents(query, top_k, filters)
        return results
    finally:
        pipeline.close()

def generate_rag_answer(query: str, top_k: int = 3, filters: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Generate an answer using RAG with context retrieval
    """
    pipeline = RAGPipeline()
    try:
        # Get relevant chunks
        search_results = pipeline.query_documents(query, top_k, filters)

        if search_results.get('error'):
            return search_results
        
        # Generate answer using the retrieved context
        answer_result = pipeline.generate_answer(
            query,
            search_results['results']
        )

        # Combine results
        return {
            "query": query,
            "answer": answer_result['answer'],
            "sources": answer_result['sources'],
            "context_chunks": answer_result['context_chunks'],
            "confidence_score": answer_result['confidence'],
            "search_results": search_results['results'],
            "processing_time": search_results['processing_time']
        }
    finally:
        pipeline.close()

