"""
Document Source for handling document-based data sources.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import structlog
import io

# Document processing imports
import PyPDF2
import pdfplumber
from docx import Document
import docx2txt

from .base import BaseDataSource
from ..models import ContextChunk, SourceType


class DocumentSource(BaseDataSource):
    """
    Document source for handling various document formats.
    
    Supports PDF, Word documents, text files, and other document formats.
    """
    
    def __init__(
        self,
        name: str,
        source_type: SourceType = SourceType.DOCUMENT,
        url: str = "",
        chunk_size: int = 1000,
        overlap: int = 200,
        **kwargs
    ):
        """
        Initialize the document source.
        
        Args:
            name: Name of the document source
            source_type: Type of data source
            url: Path to document directory or file
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            **kwargs: Additional configuration
        """
        super().__init__(name, source_type, **kwargs)
        self.url = url
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = structlog.get_logger(f"{__name__}.{name}")
        
        # Document processing configuration
        self.supported_formats = {
            '.txt': self._process_text_file,
            '.md': self._process_text_file,
            '.pdf': self._process_pdf_file,
            '.docx': self._process_docx_file,
            '.doc': self._process_doc_file,
        }
    
    async def get_chunks(
        self,
        query: str,
        max_chunks: Optional[int] = None,
        min_relevance: float = 0.0,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[ContextChunk]:
        """
        Get context chunks from documents.
        
        Args:
            query: Search query
            max_chunks: Maximum number of chunks to return
            min_relevance: Minimum relevance threshold
            user_id: Optional user ID for personalization
            session_id: Optional session ID for continuity
            
        Returns:
            List of context chunks
        """
        try:
            # Validate query
            query = await self._validate_query(query)
            
            self.logger.info(f"Getting chunks for query: {query}")
            
            # Get document content
            documents = await self._load_documents()
            
            # Process documents into chunks
            all_chunks = []
            for doc_path, content in documents:
                chunks = await self._chunk_content(content, doc_path)
                all_chunks.extend(chunks)
            
            # Filter chunks based on query relevance
            relevant_chunks = await self._filter_chunks_by_query(
                all_chunks, query, min_relevance
            )
            
            # Apply max_chunks limit
            if max_chunks:
                relevant_chunks = relevant_chunks[:max_chunks]
            
            # Update statistics
            await self._update_stats(len(relevant_chunks))
            
            self.logger.info(f"Retrieved {len(relevant_chunks)} chunks from {len(documents)} documents")
            return relevant_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to get chunks: {e}")
            return []
    
    async def refresh(self) -> None:
        """Refresh the document source."""
        try:
            self.logger.info("Refreshing document source")
            
            # Reload documents from the source directory
            if self.directory:
                # Clear existing document cache
                self._document_cache.clear()
                
                # Reload documents
                await self._load_documents()
                
                self.logger.info(f"Reloaded {len(self._document_cache)} documents from {self.directory}")
            
            self.logger.info("Document source refreshed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to refresh document source: {e}")
    
    async def close(self) -> None:
        """Close the document source."""
        try:
            self.logger.info("Closing document source")
            # Clean up any resources
        except Exception as e:
            self.logger.error(f"Error closing document source: {e}")
    
    async def _load_documents(self) -> List[tuple]:
        """
        Load documents from the source path.
        
        Returns:
            List of (path, content) tuples
        """
        documents = []
        source_path = Path(self.url)
        
        if not source_path.exists():
            self.logger.warning(f"Source path does not exist: {self.url}")
            return documents
        
        try:
            if source_path.is_file():
                # Single file
                content = await asyncio.wait_for(
                    self._load_single_document(source_path), 
                    timeout=15.0
                )
                if content:
                    documents.append((str(source_path), content))
            else:
                # Directory - scan for supported files with limit
                file_count = 0
                max_files = 50  # Limit to prevent processing too many files
                
                for file_path in source_path.rglob("*"):
                    if file_count >= max_files:
                        self.logger.warning(f"Reached maximum file limit ({max_files}), stopping")
                        break
                        
                    if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                        try:
                            content = await asyncio.wait_for(
                                self._load_single_document(file_path), 
                                timeout=15.0
                            )
                            if content:
                                documents.append((str(file_path), content))
                                file_count += 1
                        except asyncio.TimeoutError:
                            self.logger.warning(f"Loading timed out for file: {file_path}")
                            continue
                        except Exception as e:
                            self.logger.warning(f"Failed to load file {file_path}: {e}")
                            continue
        except Exception as e:
            self.logger.error(f"Error loading documents: {e}")
        
        return documents
    
    async def _load_single_document(self, file_path: Path) -> Optional[str]:
        """
        Load content from a single document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document content or None if failed
        """
        try:
            suffix = file_path.suffix.lower()
            processor = self.supported_formats.get(suffix)
            
            if processor:
                return await processor(file_path)
            else:
                self.logger.warning(f"Unsupported file format: {suffix}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load document {file_path}: {e}")
            return None
    
    async def _process_text_file(self, file_path: Path) -> str:
        """Process text files (txt, md)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Failed to read text file {file_path}: {e}")
            return ""
    
    async def _process_pdf_file(self, file_path: Path) -> str:
        """Process PDF files using PyPDF2 and pdfplumber for better text extraction."""
        try:
            content = ""
            
            # Try pdfplumber first (better for complex layouts)
            try:
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            content += f"\n--- Page {page_num} ---\n{page_text}\n"
                        else:
                            # Fallback to PyPDF2 if pdfplumber fails
                            self.logger.warning(f"pdfplumber failed to extract text from page {page_num}, trying PyPDF2")
                            break
                    else:
                        # If we successfully processed all pages with pdfplumber
                        return content.strip()
            except Exception as e:
                self.logger.warning(f"pdfplumber failed for {file_path}: {e}, trying PyPDF2")
            
            # Fallback to PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            content += f"\n--- Page {page_num} ---\n{page_text}\n"
                    except Exception as e:
                        self.logger.warning(f"Failed to extract text from page {page_num}: {e}")
                        continue
            
            return content.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to read PDF file {file_path}: {e}")
            return ""
    
    async def _process_docx_file(self, file_path: Path) -> str:
        """Process DOCX files using python-docx library."""
        try:
            content = ""
            
            # Load the document
            doc = Document(file_path)
            
            # Extract text from paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content += paragraph.text + "\n"
            
            # Extract text from tables
            for table in doc.tables:
                content += "\n--- Table ---\n"
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        content += " | ".join(row_text) + "\n"
                content += "--- End Table ---\n"
            
            # Extract headers and footers if available
            for section in doc.sections:
                try:
                    if section.header:
                        header_paragraphs = section.header.paragraphs
                        if header_paragraphs:
                            content += f"\n--- Header ---\n"
                            for para in header_paragraphs:
                                if para.text.strip():
                                    content += para.text + "\n"
                            content += "--- End Header ---\n"
                except Exception as e:
                    self.logger.debug(f"Could not extract header: {e}")
                
                try:
                    if section.footer:
                        footer_paragraphs = section.footer.paragraphs
                        if footer_paragraphs:
                            content += f"\n--- Footer ---\n"
                            for para in footer_paragraphs:
                                if para.text.strip():
                                    content += para.text + "\n"
                            content += "--- End Footer ---\n"
                except Exception as e:
                    self.logger.debug(f"Could not extract footer: {e}")
            
            return content.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to read DOCX file {file_path}: {e}")
            return ""
    
    async def _process_doc_file(self, file_path: Path) -> str:
        """Process DOC files using docx2txt library."""
        try:
            # docx2txt can handle both .doc and .docx files
            content = docx2txt.process(str(file_path))
            
            if not content.strip():
                self.logger.warning(f"No content extracted from DOC file {file_path}")
                return ""
            
            return content.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to read DOC file {file_path}: {e}")
            return ""
    
    async def _chunk_content(self, content: str, doc_path: str) -> List[ContextChunk]:
        """
        Split content into chunks.
        
        Args:
            content: Document content
            doc_path: Path to the document
            
        Returns:
            List of context chunks
        """
        chunks = []
        
        if not content.strip():
            return chunks
        
        # Simple chunking by character count
        start = 0
        max_iterations = len(content) // max(1, self.chunk_size - self.overlap) + 10  # Safety limit
        iteration_count = 0
        
        while start < len(content) and iteration_count < max_iterations:
            iteration_count += 1
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence ending within reasonable bounds
                search_start = max(start, end - 100)
                search_end = min(end, len(content))
                
                # Find sentence boundary
                for i in range(search_end - 1, search_start - 1, -1):
                    if i < len(content) and content[i] in '.!?':
                        end = i + 1
                        break
            
            # Ensure we don't exceed content length
            end = min(end, len(content))
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                chunk = await self._create_chunk(
                    content=chunk_content,
                    metadata={
                        'source_file': doc_path,
                        'chunk_start': start,
                        'chunk_end': end,
                        'chunk_size': len(chunk_content),
                    },
                    token_count=len(chunk_content.split())
                )
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            new_start = end - self.overlap
            
            # Prevent infinite loop - ensure we always advance
            if new_start <= start:
                self.logger.warning(f"Chunking would not advance, forcing advance. start={start}, new_start={new_start}")
                new_start = start + 1
            
            start = new_start
            
            # Additional safety check
            if start >= len(content):
                break
        
        if iteration_count >= max_iterations:
            self.logger.warning(f"Chunking reached maximum iterations ({max_iterations}) for document {doc_path}")
        
        return chunks
    
    async def _filter_chunks_by_query(
        self,
        chunks: List[ContextChunk],
        query: str,
        min_relevance: float
    ) -> List[ContextChunk]:
        """
        Filter chunks based on query relevance.
        
        Args:
            chunks: List of chunks to filter
            query: Search query
            min_relevance: Minimum relevance threshold
            
        Returns:
            Filtered list of chunks
        """
        if min_relevance <= 0.0:
            return chunks
        
        # Simple keyword-based filtering
        query_words = set(query.lower().split())
        relevant_chunks = []
        
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            
            if query_words:
                relevance = overlap / len(query_words)
                if relevance >= min_relevance:
                    # Add relevance score to chunk
                    chunk.relevance_score = type('obj', (object,), {
                        'score': relevance,
                        'confidence_lower': max(0, relevance - 0.1),
                        'confidence_upper': min(1, relevance + 0.1),
                        'confidence_level': 0.95,
                        'factors': {'keyword_overlap': relevance}
                    })()
                    relevant_chunks.append(chunk)
        
        # Sort by relevance
        relevant_chunks.sort(
            key=lambda c: c.relevance_score.score if c.relevance_score else 0.0,
            reverse=True
        )
        
        return relevant_chunks
