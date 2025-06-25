# Create new chunking.py file
import re
import json
from typing import List, Dict, Tuple
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

class UniversalChunker:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_document(self, content: str, filename: str) -> Dict:
        """Process any document type and return chunks with metadata"""
        
        # Detect document type
        doc_type = self._detect_doc_type(content)
        
        # Choose chunking strategy
        chunks = []
        
        # Try semantic chunking first
        semantic_chunks = self._semantic_chunk(content, filename)
        
        # Try fixed-size chunking
        fixed_chunks = self._fixed_size_chunk(content, filename)
        
        # Choose best strategy based on document structure
        if len(semantic_chunks) > 1 and self._has_good_structure(semantic_chunks):
            chunks = semantic_chunks
            chunk_strategy = ChunkingStrategy.SEMANTIC
        else:
            chunks = fixed_chunks
            chunk_strategy = ChunkingStrategy.FIXED_SIZE
            
        # Apply document-specific optimizations
        chunks = self._optimize_for_doc_type(doc_type, chunks)
        
        # Add document-level metadata
        for chunk in chunks:
            chunk['chunking_strategy'] = chunk_strategy.value
            chunk['doc_type'] = doc_type.value
            
        return {
            'chunks': chunks,
            'doc_type': doc_type,
            'chunking_strategy': chunk_strategy,
            'total_chunks': len(chunks)
        }
    
    def _semantic_chunk(self, content: str, filename: str) -> List[Dict]:
        """Chunk by headers and natural breaks"""
        chunks = []
        
        # Split by various header levels (markdown style)
        sections = re.split(r'\\n(#{1,6}\\s+.*)', content)
        
        current_section = ""
        current_header = ""
        
        for i, section in enumerate(sections):
            if re.match(r'^#{1,6}\\s+', section):  # It's a header
                # Save previous section if it exists
                if current_section.strip():
                    chunks.append(self._create_chunk(
                        current_header + "\\n" + current_section,
                        filename, len(chunks), "semantic", 
                        current_header.strip('#').strip()
                    ))
                current_header = section
                current_section = ""
            else:
                current_section += section
        
        # Add final section
        if current_section.strip():
            chunks.append(self._create_chunk(
                current_header + "\\n" + current_section,
                filename, len(chunks), "semantic", 
                current_header.strip('#').strip()
            ))
            
        return chunks
    
    def _fixed_size_chunk(self, content: str, filename: str) -> List[Dict]:
        """Fixed-size chunking with overlap"""
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(self._create_chunk(
                chunk_text, filename, len(chunks), "fixed_size"
            ))
            
        return chunks
    
    def _sentence_chunk(self, content: str, filename: str, max_sentences=5) -> List[Dict]:
        """Chunk by sentences - good for Q&A"""
        sentences = re.split(r'[.!?]+', content)
        chunks = []
        
        for i in range(0, len(sentences), max_sentences):
            chunk_sentences = sentences[i:i + max_sentences]
            chunk_text = '. '.join(chunk_sentences).strip()
            
            if chunk_text:
                chunks.append(self._create_chunk(
                    chunk_text, filename, len(chunks), "sentence"
                ))
                
        return chunks
    
    def _paragraph_chunk(self, content: str, filename: str) -> List[Dict]:
        """Chunk by paragraphs - preserves context"""
        paragraphs = [p.strip() for p in content.split('\\n\\n') if p.strip()]
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            chunks.append(self._create_chunk(
                paragraph, filename, i, "paragraph"
            ))
            
        return chunks
    
    def _recursive_chunk(self, content: str, filename: str, max_size=1000) -> List[Dict]:
        """Recursively split on different separators"""
        separators = ['\\n\\n', '\\n', '. ', ' ']
        chunks = self._recursive_split(content, separators, max_size)
        
        return [
            self._create_chunk(chunk, filename, i, "recursive") 
            for i, chunk in enumerate(chunks)
        ]
    
    def _recursive_split(self, text: str, separators: List[str], max_size: int) -> List[str]:
        """Recursively split text using different separators"""
        if len(text) <= max_size:
            return [text]
        
        if not separators:
            # If no separators left, split by character count
            return [text[i:i+max_size] for i in range(0, len(text), max_size)]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        parts = text.split(separator)
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if len(current_chunk + separator + part) <= max_size:
                current_chunk += (separator if current_chunk else "") + part
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(part) > max_size:
                    # Recursively split this part
                    sub_chunks = self._recursive_split(part, remaining_separators, max_size)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def _create_chunk(self, text: str, filename: str, chunk_id: int, 
                     chunk_type: str, section_title: str = None) -> Dict:
        """Create standardized chunk object"""
        return {
            'id': f"{filename}_{chunk_id}",
            'text': text.strip(),
            'source': filename,
            'chunk_index': chunk_id,
            'chunk_type': chunk_type,
            'section_title': section_title,
            'word_count': len(text.split()),
            'char_count': len(text),
            'preview': text[:200] + '...' if len(text) > 200 else text
        }
    
    def _has_good_structure(self, chunks: List[Dict]) -> bool:
        """Determine if semantic chunking produced good results"""
        if len(chunks) < 2:
            return False
        
        # Check if chunks have reasonable sizes
        avg_words = sum(chunk['word_count'] for chunk in chunks) / len(chunks)
        return 50 < avg_words < 2000  # Reasonable chunk sizes
    
    def _detect_doc_type(self, content: str) -> DocumentType:
        """Simple document type detection"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['experience', 'education', 'skills', 'cv', 'resume']):
            return DocumentType.RESUME
        elif any(word in content_lower for word in ['abstract', 'introduction', 'methodology', 'conclusion']):
            return DocumentType.ACADEMIC_PAPER
        elif any(word in content_lower for word in ['executive summary', 'financial', 'quarterly', 'revenue']):
            return DocumentType.BUSINESS_REPORT
        elif any(word in content_lower for word in ['manual', 'instructions', 'step', 'procedure']):
            return DocumentType.MANUAL
        else:
            return DocumentType.GENERAL_DOCUMENT
    
    def _optimize_for_doc_type(self, doc_type: DocumentType, chunks: List[Dict]) -> List[Dict]:
        """Apply document-specific optimizations"""
        
        if doc_type == DocumentType.ACADEMIC_PAPER:
            # Prioritize abstract, conclusion sections
            for chunk in chunks:
                section_title = chunk.get('section_title', '').lower()
                if any(word in section_title for word in ['abstract', 'conclusion', 'summary']):
                    chunk['priority'] = 'high'
        
        elif doc_type == DocumentType.MANUAL:
            # Enhance step-by-step instructions
            for chunk in chunks:
                if re.search(r'step \\d+|procedure|instruction', chunk['text'].lower()):
                    chunk['chunk_type'] = 'instruction'
                    chunk['priority'] = 'high'
        
        elif doc_type == DocumentType.BUSINESS_REPORT:
            # Highlight key metrics and summaries
            for chunk in chunks:
                if re.search(r'\\d+%|\\$\\d+|revenue|profit|growth', chunk['text'].lower()):
                    chunk['contains_metrics'] = True
                    chunk['priority'] = 'high'
        
        return chunks

# Convenience function for easy use
def chunk_document(content: str, filename: str, chunk_size: int = 500, chunk_overlap: int = 50) -> Dict:
    """
    Main function to chunk any document
    """
    chunker = UniversalChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.process_document(content, filename)
