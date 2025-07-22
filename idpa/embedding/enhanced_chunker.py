"""
Enhanced text chunking for academic papers with equation, figure, and table awareness.
"""

import re
import nltk
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from utils.logger import get_logger
from extraction.enhanced_pdf_processor import EnhancedContent, EnhancedEquation, EnhancedFigure, EnhancedTable


@dataclass
class DocumentMetadata:
    """Comprehensive metadata for a document."""
    document_id: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    content_hash: Optional[str] = None
    processing_timestamp: Optional[str] = None
    total_pages: Optional[int] = None
    document_type: str = "academic_paper"
    
    def __post_init__(self):
        """Generate processing timestamp if not provided."""
        if self.processing_timestamp is None:
            self.processing_timestamp = datetime.now().isoformat()


@dataclass
class TextChunk:
    """Represents a chunk of text with comprehensive metadata."""
    text: str
    chunk_id: str
    document_id: str  # Changed from paper_id for clarity
    section: str
    page: Optional[int] = None
    chunk_index: int = 0
    start_char: int = 0
    end_char: int = 0
    document_metadata: Optional[DocumentMetadata] = None
    chunk_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Legacy support - keeping paper_id as property for backward compatibility
    @property
    def paper_id(self) -> str:
        """Legacy property for backward compatibility."""
        return self.document_id
    
    @paper_id.setter
    def paper_id(self, value: str):
        """Legacy property setter for backward compatibility."""
        self.document_id = value


@dataclass
class EnhancedTextChunk(TextChunk):
    """Enhanced text chunk with equation, figure, and table awareness."""
    equations: List[EnhancedEquation] = field(default_factory=list)
    figures: List[EnhancedFigure] = field(default_factory=list)
    tables: List[EnhancedTable] = field(default_factory=list)
    content_type: str = "text"  # text, equation, figure, table, mixed
    mathematical_content: bool = False
    visual_content: bool = False
    tabular_content: bool = False
    
    def __post_init__(self):
        """Set content flags based on associated elements."""
        self.mathematical_content = len(self.equations) > 0
        self.visual_content = len(self.figures) > 0
        self.tabular_content = len(self.tables) > 0
        
        # Determine content type
        if self.mathematical_content and self.visual_content and self.tabular_content:
            self.content_type = "mixed"
        elif self.mathematical_content:
            self.content_type = "equation"
        elif self.visual_content:
            self.content_type = "figure"
        elif self.tabular_content:
            self.content_type = "table"
        else:
            self.content_type = "text"


class TextChunker:
    """Intelligent text chunking for academic papers with enhanced document tracking."""
    
    def __init__(self, config=None):
        """Initialize text chunker with configuration."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Default parameters
        self.chunk_size = getattr(config, 'chunk_size', 256) if config else 256
        self.chunk_overlap = getattr(config, 'chunk_overlap', 128) if config else 128
        self.min_chunk_size = getattr(config, 'min_chunk_size', 50) if config else 50
        
        # Semantic chunking parameters
        self.use_semantic_chunking = getattr(config, 'use_semantic_chunking', True) if config else True
        self.semantic_break_patterns = getattr(config, 'semantic_break_patterns', []) if config else []
        self.preserve_section_boundaries = getattr(config, 'preserve_section_boundaries', True) if config else True
        self.max_section_chunk_size = getattr(config, 'max_section_chunk_size', 512) if config else 512
        
        # Download required NLTK data
        self._ensure_nltk_data()
        
        # Sentence patterns for better splitting
        self.sentence_endings = r'[.!?]+(?:\s|$)'
        self.paragraph_break = r'\n\s*\n'
        
        # Academic paper specific patterns
        self.citation_pattern = r'\[[0-9,\s-]+\]|\([A-Za-z]+\s+et\s+al\.,?\s+\d{4}\)'
        self.equation_pattern = r'\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\)'
        self.figure_table_ref = r'(Figure|Table|Fig\.|Tab\.)\s+\d+'
    
    def generate_document_id(self, file_path: str, content_text: str = "") -> str:
        """
        Generate a robust document ID based on file and content.
        
        Args:
            file_path: Path to the source file
            content_text: Optional content text for additional uniqueness
            
        Returns:
            Unique document identifier
        """
        file_path = Path(file_path)
        
        # Create a hash from file name and optional content
        hash_input = f"{file_path.stem}_{file_path.suffix}"
        if content_text:
            # Use first 1000 chars of content for uniqueness
            hash_input += content_text[:1000]
        
        content_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        # Create readable document ID
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', file_path.stem)
        document_id = f"doc_{clean_name}_{content_hash}"
        
        return document_id
    
    def generate_content_hash(self, content_text: str) -> str:
        """Generate content hash for duplicate detection."""
        return hashlib.sha256(content_text.encode()).hexdigest()
    
    def create_document_metadata(
        self, 
        extracted_content, 
        file_path: str = None,
        document_id: str = None
    ) -> DocumentMetadata:
        """
        Create comprehensive document metadata.
        
        Args:
            extracted_content: ExtractedContent object
            file_path: Optional source file path
            document_id: Optional document ID (will generate if not provided)
            
        Returns:
            DocumentMetadata object
        """
        # Generate document ID if not provided
        if not document_id:
            full_text = ""
            if extracted_content.title:
                full_text += extracted_content.title
            if extracted_content.abstract:
                full_text += extracted_content.abstract
            document_id = self.generate_document_id(file_path or "unknown", full_text)
        
        # Get file information
        file_info = {}
        if file_path:
            path_obj = Path(file_path)
            file_info = {
                'file_path': str(path_obj),
                'file_name': path_obj.name,
                'file_size': path_obj.stat().st_size if path_obj.exists() else None
            }
        
        # Generate content hash
        full_content = ""
        if hasattr(extracted_content, 'title') and extracted_content.title:
            full_content += extracted_content.title + "\n"
        if hasattr(extracted_content, 'abstract') and extracted_content.abstract:
            full_content += extracted_content.abstract + "\n"
        for section_content in getattr(extracted_content, 'sections', {}).values():
            if section_content:
                full_content += section_content + "\n"
        
        content_hash = self.generate_content_hash(full_content)
        
        return DocumentMetadata(
            document_id=document_id,
            title=getattr(extracted_content, 'title', None),
            authors=getattr(extracted_content, 'authors', None),
            content_hash=content_hash,
            total_pages=getattr(extracted_content, 'total_pages', None),
            **file_info
        )
    
    def _ensure_nltk_data(self):
        """Download required NLTK data if not present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except:
                pass  # punkt_tab might not be available in older versions
    
    def chunk_extracted_content(self, extracted_content, file_path: str) -> List[TextChunk]:
        """
        Chunk extracted content from a paper.
        
        Args:
            extracted_content: ExtractedContent object
            file_path: Path to the source file
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        chunk_index = 0
        
        # Create document metadata
        document_metadata = self.create_document_metadata(extracted_content, file_path)
        
        # Chunk title and abstract together if both exist
        if extracted_content.title or extracted_content.abstract:
            title_abstract = ""
            if extracted_content.title:
                title_abstract += f"Title: {extracted_content.title}\n\n"
            if extracted_content.abstract:
                title_abstract += f"Abstract: {extracted_content.abstract}"
            
            title_chunks = self.chunk_text(
                text=title_abstract,
                document_id=document_metadata.document_id,
                section="Title_Abstract",
                start_index=chunk_index,
                document_metadata=document_metadata
            )
            chunks.extend(title_chunks)
            chunk_index += len(title_chunks)
        
        # Chunk each section separately
        for section_name, section_content in extracted_content.sections.items():
            if section_content and section_content.strip():
                section_chunks = self.chunk_text(
                    text=section_content,
                    document_id=document_metadata.document_id,
                    section=section_name,
                    start_index=chunk_index,
                    document_metadata=document_metadata
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
        
        # Chunk tables as structured content
        for table_idx, table in enumerate(extracted_content.tables):
            table_text = self._table_to_text(table)
            if table_text:
                table_chunks = self.chunk_text(
                    text=table_text,
                    document_id=document_metadata.document_id,
                    section=f"Table_{table_idx + 1}",
                    start_index=chunk_index,
                    document_metadata=document_metadata
                )
                chunks.extend(table_chunks)
                chunk_index += len(table_chunks)
        
        # Add metadata to all chunks
        for chunk in chunks:
            chunk.chunk_metadata.update({
                'total_chunks': len(chunks),
                'document_title': document_metadata.title,
                'document_authors': document_metadata.authors,
                'document_id': document_metadata.document_id,
                'document_hash': document_metadata.content_hash,
                'document_type': document_metadata.document_type,
                'document_path': document_metadata.file_path,
                'document_size': document_metadata.file_size,
                'document_pages': document_metadata.total_pages
            })
        
        self.logger.info(f"Generated {len(chunks)} chunks for document {document_metadata.document_id}")
        return chunks
    
    def chunk_text(
        self, 
        text: str, 
        document_id: str, 
        section: str, 
        start_index: int = 0,
        page: Optional[int] = None,
        document_metadata: Optional[DocumentMetadata] = None
    ) -> List[TextChunk]:
        """
        Chunk a single text into semantically meaningful pieces.
        
        Args:
            text: Text to chunk
            document_id: Document identifier
            section: Section name
            start_index: Starting chunk index
            page: Optional page number
            document_metadata: Optional DocumentMetadata object
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        text = self._preprocess_text(text)
        
        # Try different chunking strategies based on text characteristics
        if self._is_structured_content(text):
            chunks = self._chunk_structured_content(text)
        elif len(text) <= self.chunk_size:
            # Text is small enough to be a single chunk
            chunks = [text]
        else:
            # Use semantic chunking for longer texts
            chunks = self._semantic_chunk(text)
        
        # Convert to TextChunk objects
        text_chunks = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunk_id = f"{document_id}_{section}_{start_index + i}"
                
                text_chunk = TextChunk(
                    text=chunk_text.strip(),
                    chunk_id=chunk_id,
                    document_id=document_id,
                    section=section,
                    page=page,
                    chunk_index=start_index + i,
                    document_metadata=document_metadata,
                    chunk_metadata={
                        'chunk_method': 'semantic',
                        'original_length': len(text),
                        'chunk_length': len(chunk_text.strip())
                    }
                )
                text_chunks.append(text_chunk)
        
        return text_chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better chunking."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common formatting issues
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after sentences
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)     # Add space between words
        
        # Preserve important academic elements
        text = re.sub(self.citation_pattern, lambda m: f" {m.group()} ", text)
        text = re.sub(self.figure_table_ref, lambda m: f" {m.group()} ", text)
        
        return text.strip()
    
    def _is_structured_content(self, text: str) -> bool:
        """Check if text appears to be structured (lists, equations, etc.)."""
        # Check for bullet points or numbered lists
        list_pattern = r'^\s*[•\-\*\d+\.]\s+'
        list_lines = len(re.findall(list_pattern, text, re.MULTILINE))
        
        # Check for equations
        equations = len(re.findall(self.equation_pattern, text))
        
        # Check for short lines (might be structured)
        lines = text.split('\n')
        short_lines = sum(1 for line in lines if 0 < len(line.strip()) < 50)
        
        return (list_lines > 2 or equations > 0 or 
                short_lines > len(lines) * 0.3)
    
    def _chunk_structured_content(self, text: str) -> List[str]:
        """Chunk structured content preserving logical units."""
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = re.split(self.paragraph_break, text)
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed chunk size
            if (len(current_chunk) + len(paragraph) + 1 > self.chunk_size and 
                current_chunk):
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _semantic_chunk(self, text: str) -> List[str]:
        """Perform semantic chunking based on sentences and paragraphs."""
        # If semantic chunking is enabled, try section-based chunking first
        if self.use_semantic_chunking and self.semantic_break_patterns:
            semantic_chunks = self._semantic_section_chunk(text)
            if semantic_chunks:
                return semantic_chunks
        
        # Fall back to sentence-based chunking
        return self._sentence_based_chunk(text)
    
    def _semantic_section_chunk(self, text: str) -> List[str]:
        """Chunk text based on semantic section boundaries."""
        chunks = []
        
        # Find all section boundaries
        section_boundaries = []
        for pattern in self.semantic_break_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
            section_boundaries.extend(matches)
        
        # Sort boundaries by position
        section_boundaries.sort(key=lambda x: x.start())
        
        if not section_boundaries:
            # No section boundaries found, fall back to sentence-based chunking
            return []
        
        # Create chunks based on section boundaries
        current_pos = 0
        for i, boundary in enumerate(section_boundaries):
            # Get text from current position to this boundary
            section_text = text[current_pos:boundary.start()].strip()
            
            if section_text:
                # Chunk this section if it's too long
                if len(section_text) > self.max_section_chunk_size:
                    sub_chunks = self._sentence_based_chunk(section_text)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(section_text)
            
            # Start new section from boundary
            current_pos = boundary.start()
        
        # Handle the last section
        final_section = text[current_pos:].strip()
        if final_section:
            if len(final_section) > self.max_section_chunk_size:
                sub_chunks = self._sentence_based_chunk(final_section)
                chunks.extend(sub_chunks)
            else:
                chunks.append(final_section)
        
        return [chunk for chunk in chunks if len(chunk.strip()) >= self.min_chunk_size]
    
    def _sentence_based_chunk(self, text: str) -> List[str]:
        """Perform sentence-based chunking with overlap."""
        chunks = []
        
        # Split into sentences using NLTK
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            # Fallback to simple sentence splitting
            sentences = re.split(self.sentence_endings, text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            # If this sentence alone exceeds chunk size, split it further
            if sentence_length > self.chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_length = 0
                
                # Split long sentence into smaller parts
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding this sentence would exceed chunk size
            if (current_length + sentence_length + 1 > self.chunk_size and 
                current_chunk):
                
                # Save current chunk
                chunks.append(current_chunk)
                
                # Start new chunk with overlap if configured
                if self.chunk_overlap > 0 and chunks:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                    current_length = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_length = sentence_length
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length = len(current_chunk)
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a very long sentence into smaller parts."""
        # Try to split on commas, semicolons, or conjunctions
        split_points = [', ', '; ', ' and ', ' or ', ' but ', ' however ', ' therefore ']
        
        parts = [sentence]
        for delimiter in split_points:
            new_parts = []
            for part in parts:
                if len(part) > self.chunk_size:
                    subparts = part.split(delimiter)
                    current_subpart = ""
                    for subpart in subparts:
                        if len(current_subpart) + len(subpart) + len(delimiter) <= self.chunk_size:
                            if current_subpart:
                                current_subpart += delimiter + subpart
                            else:
                                current_subpart = subpart
                        else:
                            if current_subpart:
                                new_parts.append(current_subpart)
                            current_subpart = subpart
                    if current_subpart:
                        new_parts.append(current_subpart)
                else:
                    new_parts.append(part)
            parts = new_parts
        
        # If still too long, force split by character count
        final_parts = []
        for part in parts:
            if len(part) <= self.chunk_size:
                final_parts.append(part)
            else:
                # Force split into smaller chunks
                words = part.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                        if current_chunk:
                            current_chunk += " " + word
                        else:
                            current_chunk = word
                    else:
                        if current_chunk:
                            final_parts.append(current_chunk)
                        current_chunk = word
                if current_chunk:
                    final_parts.append(current_chunk)
        
        return final_parts
    
    def _get_overlap_text(self, chunk: str) -> str:
        """Get overlap text from the end of a chunk."""
        if self.chunk_overlap <= 0:
            return ""
        
        # For 50% overlap, we want to include the last portion of the chunk
        target_overlap = min(self.chunk_overlap, len(chunk) // 2)
        
        # Try to get last complete sentences for overlap
        sentences = re.split(self.sentence_endings, chunk)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            # If no sentence breaks, just take the last portion
            return chunk[-target_overlap:] if len(chunk) > target_overlap else chunk
        
        overlap_text = ""
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) + 1 <= target_overlap:
                if overlap_text:
                    overlap_text = sentence + " " + overlap_text
                else:
                    overlap_text = sentence
            else:
                # If adding this sentence would exceed overlap, try to include part of it
                remaining_space = target_overlap - len(overlap_text)
                if remaining_space > 20:  # Only if we have meaningful space
                    words = sentence.split()
                    for word in words:
                        if len(overlap_text) + len(word) + 1 <= target_overlap:
                            if overlap_text:
                                overlap_text += " " + word
                            else:
                                overlap_text = word
                        else:
                            break
                break
        
        return overlap_text
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table data to text representation."""
        if not table or 'data' not in table:
            return ""
        
        table_data = table['data']
        if not table_data:
            return ""
        
        # Format table as text
        text_lines = []
        text_lines.append(f"Table {table.get('table_index', '')} on page {table.get('page', '')}")
        
        # Add headers if available
        if table.get('headers'):
            headers = " | ".join(str(cell) for cell in table['headers'] if cell)
            text_lines.append(f"Headers: {headers}")
        
        # Add table rows
        for row_idx, row in enumerate(table_data):
            if row and any(cell for cell in row):  # Skip empty rows
                row_text = " | ".join(str(cell) for cell in row if cell)
                text_lines.append(f"Row {row_idx + 1}: {row_text}")
        
        return "\n".join(text_lines)
    
    def optimize_chunks_for_retrieval(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Optimize chunks for better retrieval performance."""
        optimized_chunks = []
        
        for chunk in chunks:
            # Skip very short chunks that might not be informative
            if len(chunk.text.split()) < 5:
                continue
            
            # Enhance chunk with context
            enhanced_text = self._enhance_chunk_context(chunk)
            chunk.text = enhanced_text
            
            # Add retrieval-specific metadata
            chunk.chunk_metadata.update({
                'word_count': len(chunk.text.split()),
                'char_count': len(chunk.text),
                'has_citations': bool(re.search(self.citation_pattern, chunk.text)),
                'has_figures': bool(re.search(self.figure_table_ref, chunk.text, re.IGNORECASE)),
                'has_equations': bool(re.search(self.equation_pattern, chunk.text))
            })
            
            optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _enhance_chunk_context(self, chunk: TextChunk) -> str:
        """Enhance chunk with contextual information for better retrieval."""
        context_parts = []
        
        # Add section context
        if chunk.section and chunk.section != "Title_Abstract":
            context_parts.append(f"Section: {chunk.section}")
        
        # Add paper context if available
        if chunk.document_metadata and chunk.document_metadata.title:
            context_parts.append(f"Paper: {chunk.document_metadata.title}")
        
        # Combine context with original text
        if context_parts:
            context_prefix = " | ".join(context_parts) + "\n\n"
            return context_prefix + chunk.text
        
        return chunk.text


class EnhancedTextChunker(TextChunker):
    """Enhanced text chunker with equation, figure, and table awareness."""
    
    def __init__(self, config=None):
        """Initialize enhanced text chunker."""
        super().__init__(config)
        
        # Enhanced patterns for academic content
        self.equation_patterns = [
            r'\$\$[^$]+\$\$',  # Display equations
            r'\$[^$]+\$',      # Inline equations
            r'\\begin\{equation\}.*?\\end\{equation\}',
            r'\\begin\{align\}.*?\\end\{align\}',
            r'\\begin\{eqnarray\}.*?\\end\{eqnarray\}',
            r'\\begin\{gather\}.*?\\end\{gather\}',
            r'\\begin\{multline\}.*?\\end\{multline\}',
            r'\\begin\{cases\}.*?\\end\{cases\}',
            r'\\begin\{matrix\}.*?\\end\{matrix\}',
            r'\\\[.*?\\\]',    # Display equations
            r'\\\(.*?\\\)',    # Inline equations
        ]
        
        # Figure and table reference patterns
        self.figure_ref_pattern = r'(?:Figure|Fig\.|figure|fig\.)\s+(\d+(?:\.\d+)*)'
        self.table_ref_pattern = r'(?:Table|Tab\.|table|tab\.)\s+(\d+(?:\.\d+)*)'
        self.equation_ref_pattern = r'(?:Equation|Eq\.|equation|eq\.)\s+(\d+(?:\.\d+)*)'
        
        # Context preservation settings
        self.equation_context_window = getattr(config, 'equation_context_window', 100) if config else 100
        self.preserve_equation_context = getattr(config, 'preserve_equation_context', True) if config else True
        
        # Compile patterns
        self.compiled_equation_patterns = [
            re.compile(p, re.DOTALL | re.IGNORECASE) 
            for p in self.equation_patterns
        ]
    
    def chunk_enhanced_content(
        self, 
        content: EnhancedContent, 
        document_id: str = None,
        file_path: str = None
    ) -> List[EnhancedTextChunk]:
        """
        Chunk enhanced content with equation, figure, and table awareness.
        
        Args:
            content: EnhancedContent object from enhanced PDF extraction
            document_id: Document identifier
            file_path: Path to source file
            
        Returns:
            List of EnhancedTextChunk objects
        """
        if document_id is None:
            document_id = self.generate_document_id(file_path or "", content.title or "")
        
        # Create document metadata
        doc_metadata = self.create_document_metadata(content, file_path, document_id)
        
        chunks = []
        chunk_index = 0
        
        # Process each section with enhanced awareness
        for section_name, section_text in content.sections.items():
            if not section_text or not section_text.strip():
                continue
            
            # Find equations, figures, and tables in this section
            section_equations = self._find_section_equations(content.equations, section_name)
            section_figures = self._find_section_figures(content.enhanced_figures, section_name)
            section_tables = self._find_section_tables(content.enhanced_tables, section_name)
            
            # Create enhanced chunks for this section
            section_chunks = self._chunk_section_enhanced(
                text=section_text,
                document_id=document_id,
                section=section_name,
                start_index=chunk_index,
                document_metadata=doc_metadata,
                equations=section_equations,
                figures=section_figures,
                tables=section_tables
            )
            
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        # Create standalone chunks for equations, figures, and tables
        standalone_chunks = self._create_standalone_content_chunks(
            content, document_id, doc_metadata, chunk_index
        )
        chunks.extend(standalone_chunks)
        
        # Optimize chunks for retrieval
        optimized_chunks = self.optimize_enhanced_chunks(chunks)
        
        return optimized_chunks
    
    def _chunk_section_enhanced(
        self,
        text: str,
        document_id: str,
        section: str,
        start_index: int,
        document_metadata: DocumentMetadata,
        equations: List[EnhancedEquation] = None,
        figures: List[EnhancedFigure] = None,
        tables: List[EnhancedTable] = None
    ) -> List[EnhancedTextChunk]:
        """Chunk a section with enhanced content awareness."""
        equations = equations or []
        figures = figures or []
        tables = tables or []
        
        # First, identify special content regions in the text
        content_regions = self._identify_content_regions(text, equations, figures, tables)
        
        # Create chunks considering content regions
        chunks = self._create_context_aware_chunks(
            text=text,
            document_id=document_id,
            section=section,
            start_index=start_index,
            document_metadata=document_metadata,
            content_regions=content_regions,
            equations=equations,
            figures=figures,
            tables=tables
        )
        
        return chunks
    
    def _identify_content_regions(
        self,
        text: str,
        equations: List[EnhancedEquation],
        figures: List[EnhancedFigure],
        tables: List[EnhancedTable]
    ) -> List[Dict[str, Any]]:
        """Identify regions in text that contain special content."""
        regions = []
        
        # Find equation regions
        for eq in equations:
            if eq.latex in text:
                start_pos = text.find(eq.latex)
                if start_pos != -1:
                    end_pos = start_pos + len(eq.latex)
                    regions.append({
                        'type': 'equation',
                        'start': start_pos,
                        'end': end_pos,
                        'content': eq,
                        'text': eq.latex
                    })
        
        # Find figure references
        for match in re.finditer(self.figure_ref_pattern, text, re.IGNORECASE):
            regions.append({
                'type': 'figure_ref',
                'start': match.start(),
                'end': match.end(),
                'text': match.group(),
                'reference': match.group(1)
            })
        
        # Find table references
        for match in re.finditer(self.table_ref_pattern, text, re.IGNORECASE):
            regions.append({
                'type': 'table_ref',
                'start': match.start(),
                'end': match.end(),
                'text': match.group(),
                'reference': match.group(1)
            })
        
        # Find equation references
        for match in re.finditer(self.equation_ref_pattern, text, re.IGNORECASE):
            regions.append({
                'type': 'equation_ref',
                'start': match.start(),
                'end': match.end(),
                'text': match.group(),
                'reference': match.group(1)
            })
        
        # Sort regions by position
        regions.sort(key=lambda x: x['start'])
        
        return regions
    
    def _create_context_aware_chunks(
        self,
        text: str,
        document_id: str,
        section: str,
        start_index: int,
        document_metadata: DocumentMetadata,
        content_regions: List[Dict[str, Any]],
        equations: List[EnhancedEquation],
        figures: List[EnhancedFigure],
        tables: List[EnhancedTable]
    ) -> List[EnhancedTextChunk]:
        """Create chunks while preserving context around special content."""
        chunks = []
        
        if not text.strip():
            return chunks
        
        # If no special content, use standard chunking
        if not content_regions:
            standard_chunks = self.chunk_text(
                text, document_id, section, start_index, 
                document_metadata=document_metadata
            )
            # Convert to enhanced chunks
            for chunk in standard_chunks:
                enhanced_chunk = EnhancedTextChunk(
                    text=chunk.text,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    section=chunk.section,
                    page=chunk.page,
                    chunk_index=chunk.chunk_index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    document_metadata=chunk.document_metadata,
                    chunk_metadata=chunk.chunk_metadata
                )
                chunks.append(enhanced_chunk)
            return chunks
        
        # Create chunks with special content awareness
        current_pos = 0
        chunk_index = start_index
        
        for region in content_regions:
            region_start = region['start']
            region_end = region['end']
            
            # Create chunk for text before special content (if any)
            if current_pos < region_start:
                before_text = text[current_pos:region_start].strip()
                if len(before_text) >= self.min_chunk_size:
                    chunk = self._create_enhanced_chunk(
                        text=before_text,
                        document_id=document_id,
                        section=section,
                        chunk_index=chunk_index,
                        start_char=current_pos,
                        end_char=region_start,
                        document_metadata=document_metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            
            # Create chunk for special content with context
            context_chunk = self._create_special_content_chunk(
                text=text,
                region=region,
                document_id=document_id,
                section=section,
                chunk_index=chunk_index,
                document_metadata=document_metadata,
                equations=equations,
                figures=figures,
                tables=tables
            )
            
            if context_chunk:
                chunks.append(context_chunk)
                chunk_index += 1
            
            current_pos = region_end
        
        # Create chunk for remaining text (if any)
        if current_pos < len(text):
            remaining_text = text[current_pos:].strip()
            if len(remaining_text) >= self.min_chunk_size:
                chunk = self._create_enhanced_chunk(
                    text=remaining_text,
                    document_id=document_id,
                    section=section,
                    chunk_index=chunk_index,
                    start_char=current_pos,
                    end_char=len(text),
                    document_metadata=document_metadata
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_special_content_chunk(
        self,
        text: str,
        region: Dict[str, Any],
        document_id: str,
        section: str,
        chunk_index: int,
        document_metadata: DocumentMetadata,
        equations: List[EnhancedEquation],
        figures: List[EnhancedFigure],
        tables: List[EnhancedTable]
    ) -> Optional[EnhancedTextChunk]:
        """Create a chunk for special content with appropriate context."""
        region_type = region['type']
        region_start = region['start']
        region_end = region['end']
        
        # Determine context window
        if region_type == 'equation':
            context_window = self.equation_context_window
        else:
            context_window = self.chunk_overlap
        
        # Calculate context boundaries
        context_start = max(0, region_start - context_window)
        context_end = min(len(text), region_end + context_window)
        
        # Extract text with context
        chunk_text = text[context_start:context_end].strip()
        
        if len(chunk_text) < self.min_chunk_size:
            return None
        
        # Create enhanced chunk
        chunk_id = f"{document_id}_{section}_{chunk_index}"
        
        enhanced_chunk = EnhancedTextChunk(
            text=chunk_text,
            chunk_id=chunk_id,
            document_id=document_id,
            section=section,
            chunk_index=chunk_index,
            start_char=context_start,
            end_char=context_end,
            document_metadata=document_metadata,
            chunk_metadata={
                'special_content_type': region_type,
                'has_context': True,
                'context_window': context_window,
                'region_reference': region.get('reference', ''),
                'chunk_method': 'context_aware'
            }
        )
        
        # Associate relevant content
        if region_type == 'equation' and 'content' in region:
            enhanced_chunk.equations = [region['content']]
        elif region_type == 'figure_ref':
            # Find associated figures
            ref_num = region.get('reference', '')
            associated_figures = [fig for fig in figures if ref_num in (fig.caption or '')]
            enhanced_chunk.figures = associated_figures
        elif region_type == 'table_ref':
            # Find associated tables
            ref_num = region.get('reference', '')
            associated_tables = [tab for tab in tables if ref_num in (tab.caption or '')]
            enhanced_chunk.tables = associated_tables
        
        return enhanced_chunk
    
    def _create_enhanced_chunk(
        self,
        text: str,
        document_id: str,
        section: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        document_metadata: DocumentMetadata,
        equations: List[EnhancedEquation] = None,
        figures: List[EnhancedFigure] = None,
        tables: List[EnhancedTable] = None
    ) -> EnhancedTextChunk:
        """Create an enhanced text chunk."""
        chunk_id = f"{document_id}_{section}_{chunk_index}"
        
        return EnhancedTextChunk(
            text=text,
            chunk_id=chunk_id,
            document_id=document_id,
            section=section,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            document_metadata=document_metadata,
            equations=equations or [],
            figures=figures or [],
            tables=tables or [],
            chunk_metadata={
                'chunk_method': 'enhanced',
                'original_length': len(text),
                'has_special_content': bool(equations or figures or tables)
            }
        )
    
    def _create_standalone_content_chunks(
        self,
        content: EnhancedContent,
        document_id: str,
        document_metadata: DocumentMetadata,
        start_index: int
    ) -> List[EnhancedTextChunk]:
        """Create standalone chunks for equations, figures, and tables."""
        chunks = []
        chunk_index = start_index
        
        # Create chunks for standalone equations
        for eq in content.equations:
            if eq.text_representation or eq.latex:
                chunk_text = self._create_equation_chunk_text(eq)
                chunk_id = f"{document_id}_equation_{chunk_index}"
                
                chunk = EnhancedTextChunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    section="Equations",
                    page=eq.page,
                    chunk_index=chunk_index,
                    document_metadata=document_metadata,
                    equations=[eq],
                    content_type="equation",
                    mathematical_content=True,
                    chunk_metadata={
                        'standalone_content': True,
                        'content_type': 'equation',
                        'equation_type': eq.equation_type,
                        'confidence': eq.confidence
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
        
        # Create chunks for standalone figures
        for fig in content.enhanced_figures:
            chunk_text = self._create_figure_chunk_text(fig)
            chunk_id = f"{document_id}_figure_{chunk_index}"
            
            chunk = EnhancedTextChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                document_id=document_id,
                section="Figures",
                page=fig.page,
                chunk_index=chunk_index,
                document_metadata=document_metadata,
                figures=[fig],
                content_type="figure",
                visual_content=True,
                chunk_metadata={
                    'standalone_content': True,
                    'content_type': 'figure',
                    'figure_type': fig.figure_type,
                    'has_extracted_text': bool(fig.extracted_text),
                    'contains_equations': fig.contains_equations
                }
            )
            chunks.append(chunk)
            chunk_index += 1
        
        # Create chunks for standalone tables
        for table in content.enhanced_tables:
            chunk_text = self._create_table_chunk_text(table)
            chunk_id = f"{document_id}_table_{chunk_index}"
            
            chunk = EnhancedTextChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                document_id=document_id,
                section="Tables",
                page=table.page,
                chunk_index=chunk_index,
                document_metadata=document_metadata,
                tables=[table],
                content_type="table",
                tabular_content=True,
                chunk_metadata={
                    'standalone_content': True,
                    'content_type': 'table',
                    'table_type': table.table_type,
                    'num_rows': table.num_rows,
                    'num_cols': table.num_cols,
                    'confidence': table.confidence
                }
            )
            chunks.append(chunk)
            chunk_index += 1
        
        return chunks
    
    def _create_equation_chunk_text(self, equation: EnhancedEquation) -> str:
        """Create searchable text for an equation chunk."""
        parts = []
        
        if equation.latex:
            parts.append(f"Equation (LaTeX): {equation.latex}")
        
        if equation.text_representation:
            parts.append(f"Equation (Text): {equation.text_representation}")
        
        if equation.equation_type:
            parts.append(f"Type: {equation.equation_type}")
        
        parts.append(f"Page: {equation.page}")
        
        return " | ".join(parts)
    
    def _create_figure_chunk_text(self, figure: EnhancedFigure) -> str:
        """Create searchable text for a figure chunk."""
        parts = []
        
        if figure.caption:
            parts.append(f"Figure Caption: {figure.caption}")
        
        if figure.description:
            parts.append(f"Description: {figure.description}")
        
        if figure.extracted_text:
            parts.append(f"Extracted Text: {figure.extracted_text}")
        
        if figure.figure_type:
            parts.append(f"Type: {figure.figure_type}")
        
        if figure.contains_equations:
            parts.append("Contains mathematical equations")
        
        parts.append(f"Page: {figure.page}")
        parts.append(f"Dimensions: {figure.width}x{figure.height}")
        
        return " | ".join(parts)
    
    def _create_table_chunk_text(self, table: EnhancedTable) -> str:
        """Create searchable text for a table chunk."""
        parts = []
        
        if table.caption:
            parts.append(f"Table Caption: {table.caption}")
        
        # Add headers
        if table.headers:
            parts.append(f"Headers: {', '.join(table.headers)}")
        
        # Add sample data (first few rows)
        if table.data:
            sample_rows = table.data[:3]  # First 3 rows
            for i, row in enumerate(sample_rows):
                row_text = ', '.join(str(cell) for cell in row if cell)
                parts.append(f"Row {i+1}: {row_text}")
        
        if table.table_type:
            parts.append(f"Type: {table.table_type}")
        
        parts.append(f"Page: {table.page}")
        parts.append(f"Size: {table.num_rows} rows x {table.num_cols} columns")
        
        return " | ".join(parts)
    
    def _find_section_equations(self, equations: List[EnhancedEquation], section_name: str) -> List[EnhancedEquation]:
        """Find equations that belong to a specific section."""
        # For now, return all equations - in practice, you'd match by position or content
        return equations
    
    def _find_section_figures(self, figures: List[EnhancedFigure], section_name: str) -> List[EnhancedFigure]:
        """Find figures that belong to a specific section."""
        # For now, return all figures - in practice, you'd match by position or content
        return figures
    
    def _find_section_tables(self, tables: List[EnhancedTable], section_name: str) -> List[EnhancedTable]:
        """Find tables that belong to a specific section."""
        # For now, return all tables - in practice, you'd match by position or content
        return tables
    
    def optimize_enhanced_chunks(self, chunks: List[EnhancedTextChunk]) -> List[EnhancedTextChunk]:
        """Optimize enhanced chunks for better retrieval performance."""
        optimized_chunks = []
        
        for chunk in chunks:
            # Skip very short chunks that might not be informative
            if len(chunk.text.split()) < 3:
                continue
            
            # Enhance chunk metadata for retrieval
            chunk.chunk_metadata.update({
                'word_count': len(chunk.text.split()),
                'char_count': len(chunk.text),
                'has_equations': len(chunk.equations) > 0,
                'has_figures': len(chunk.figures) > 0,
                'has_tables': len(chunk.tables) > 0,
                'content_diversity': self._calculate_content_diversity(chunk),
                'mathematical_score': self._calculate_mathematical_score(chunk),
                'visual_score': self._calculate_visual_score(chunk),
                'tabular_score': self._calculate_tabular_score(chunk)
            })
            
            optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _calculate_content_diversity(self, chunk: EnhancedTextChunk) -> float:
        """Calculate how diverse the content types are in this chunk."""
        content_types = 0
        if chunk.mathematical_content:
            content_types += 1
        if chunk.visual_content:
            content_types += 1
        if chunk.tabular_content:
            content_types += 1
        
        # Always has text content
        content_types += 1
        
        return content_types / 4.0  # Normalize to 0-1
    
    def _calculate_mathematical_score(self, chunk: EnhancedTextChunk) -> float:
        """Calculate how mathematical the chunk content is."""
        score = 0.0
        
        # Equation count contribution
        if chunk.equations:
            score += len(chunk.equations) * 0.3
        
        # Mathematical symbols in text
        math_symbols = r'[=<>≤≥≠±∞∫∑∏√α-ωΑ-Ω]'
        symbol_matches = len(re.findall(math_symbols, chunk.text))
        score += min(symbol_matches * 0.1, 0.5)
        
        # Mathematical functions
        math_functions = r'\b(sin|cos|tan|log|ln|exp|sqrt|sum|integral)\b'
        function_matches = len(re.findall(math_functions, chunk.text, re.IGNORECASE))
        score += min(function_matches * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def _calculate_visual_score(self, chunk: EnhancedTextChunk) -> float:
        """Calculate how visual the chunk content is."""
        score = 0.0
        
        # Figure count contribution
        if chunk.figures:
            score += len(chunk.figures) * 0.4
        
        # Visual description words
        visual_words = r'\b(figure|image|chart|graph|plot|diagram|illustration|visual|shown|depicted)\b'
        visual_matches = len(re.findall(visual_words, chunk.text, re.IGNORECASE))
        score += min(visual_matches * 0.1, 0.4)
        
        return min(score, 1.0)
    
    def _calculate_tabular_score(self, chunk: EnhancedTextChunk) -> float:
        """Calculate how tabular the chunk content is."""
        score = 0.0
        
        # Table count contribution
        if chunk.tables:
            score += len(chunk.tables) * 0.4
        
        # Tabular description words
        tabular_words = r'\b(table|column|row|data|results|values|statistics|comparison)\b'
        tabular_matches = len(re.findall(tabular_words, chunk.text, re.IGNORECASE))
        score += min(tabular_matches * 0.1, 0.4)
        
        return min(score, 1.0) 