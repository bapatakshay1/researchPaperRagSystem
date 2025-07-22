"""
Enhanced PDF content extraction with advanced equation, figure, and table processing.
"""

import json
import re
import fitz  # PyMuPDF
import pdfplumber
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import logging
from concurrent.futures import ThreadPoolExecutor
import base64
import io

from utils.logger import get_logger

# Optional advanced imports
try:
    import layoutparser as lp
    HAS_LAYOUTPARSER = True
except ImportError:
    lp = None
    HAS_LAYOUTPARSER = False

try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image, ImageEnhance, ImageFilter
    HAS_OCR = True
except ImportError:
    pytesseract = None
    convert_from_path = None
    Image = None
    ImageEnhance = None
    ImageFilter = None
    HAS_OCR = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    easyocr = None
    HAS_EASYOCR = False

try:
    from sympy import latex, sympify, parse_expr
    from sympy.parsing.latex import parse_latex
    HAS_SYMPY = True
except ImportError:
    latex = None
    sympify = None
    parse_expr = None
    parse_latex = None
    HAS_SYMPY = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False


@dataclass
class ExtractedContent:
    """Structure for extracted PDF content."""
    title: str = ""
    authors: List[str] = None
    abstract: str = ""
    sections: Dict[str, str] = None
    tables: List[Dict[str, Any]] = None
    figures: List[Dict[str, Any]] = None
    references: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.sections is None:
            self.sections = {}
        if self.tables is None:
            self.tables = []
        if self.figures is None:
            self.figures = []
        if self.references is None:
            self.references = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EnhancedEquation:
    """Enhanced equation representation with multiple formats."""
    latex: str = ""
    mathml: str = ""
    text_representation: str = ""
    image_data: Optional[bytes] = None
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    page: int = 0
    confidence: float = 0.0
    equation_type: str = "inline"  # inline, display, numbered


@dataclass
class EnhancedFigure:
    """Enhanced figure representation with content analysis."""
    image_data: bytes
    caption: str = ""
    figure_type: str = "unknown"  # chart, diagram, photo, plot, etc.
    extracted_text: str = ""
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    page: int = 0
    width: int = 0
    height: int = 0
    description: str = ""
    contains_equations: bool = False


@dataclass
class EnhancedTable:
    """Enhanced table representation with better structure preservation."""
    data: List[List[str]]
    headers: List[str] = None
    caption: str = ""
    table_type: str = "data"  # data, results, comparison, etc.
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    page: int = 0
    num_rows: int = 0
    num_cols: int = 0
    confidence: float = 0.0
    formatting_preserved: bool = False


@dataclass
class EnhancedContent(ExtractedContent):
    """Enhanced content structure with improved extraction."""
    equations: List[EnhancedEquation] = None
    enhanced_figures: List[EnhancedFigure] = None
    enhanced_tables: List[EnhancedTable] = None
    layout_analysis: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.equations is None:
            self.equations = []
        if self.enhanced_figures is None:
            self.enhanced_figures = []
        if self.enhanced_tables is None:
            self.enhanced_tables = []
        if self.layout_analysis is None:
            self.layout_analysis = {}


class PDFProcessor:
    """Processes PDF documents and extracts structured content."""
    
    def __init__(self, config=None):
        """Initialize PDF processor with configuration."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Common section headers in academic papers
        self.section_patterns = [
            r'^(abstract|introduction|background|literature\s+review|related\s+work)',
            r'^(methodology|methods|approach|implementation)',
            r'^(results|findings|experiments?|evaluation)',
            r'^(discussion|analysis|interpretation)',
            r'^(conclusion|conclusions?|summary|future\s+work)',
            r'^(references?|bibliography|acknowledgments?)',
            r'^(\d+\.?\s*[a-zA-Z])',  # Numbered sections
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.section_patterns]
    
    def extract_from_file(self, pdf_path: str) -> ExtractedContent:
        """
        Extract content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ExtractedContent object with extracted information
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Try PyMuPDF first for better performance
        try:
            content = self._extract_with_pymupdf(pdf_path)
            if not content.title and not content.sections:
                # Fallback to pdfplumber if PyMuPDF failed
                self.logger.warning("PyMuPDF extraction yielded minimal content, trying pdfplumber")
                content = self._extract_with_pdfplumber(pdf_path)
        except Exception as e:
            self.logger.warning(f"PyMuPDF extraction failed: {e}, trying pdfplumber")
            content = self._extract_with_pdfplumber(pdf_path)
        
        # OCR fallback if both methods failed to extract meaningful content
        if (not content.title and not content.sections and 
            self.config and self.config.ocr_fallback):
            self.logger.info("Attempting OCR extraction as fallback")
            try:
                ocr_content = self._extract_with_ocr(pdf_path)
                if ocr_content.title or ocr_content.sections:
                    content = ocr_content
            except Exception as e:
                self.logger.error(f"OCR extraction failed: {e}")
        
        # Post-process and enhance extracted content
        content = self._post_process_content(content, pdf_path)
        
        self.logger.info(f"Extraction completed for {pdf_path.name}")
        return content
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> ExtractedContent:
        """Extract content using PyMuPDF with improved text handling."""
        content = ExtractedContent()
        
        with fitz.open(str(pdf_path)) as doc:
            full_text = ""
            page_texts = []
            
            # Extract metadata
            content.metadata = {
                'page_count': len(doc),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
                'file_size': pdf_path.stat().st_size,
                'extraction_method': 'PyMuPDF'
            }
            
            # Extract text from all pages with better handling
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Try multiple extraction methods for better results
                page_text = self._extract_page_text_enhanced(page, page_num)
                
                if page_text:
                    page_texts.append(page_text)
                    full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                # Extract images if configured
                if self.config and getattr(self.config, 'preserve_images', False):
                    images = self._extract_images_from_page(page, page_num)
                    content.figures.extend(images)
        
        # Clean and process the extracted text
        full_text = self._clean_text(full_text)
        
        # Parse structured content from full text
        self._parse_structured_content(full_text, content)
        
        # Add page-level information
        content.metadata['page_texts'] = page_texts
        
        return content
    
    def _extract_page_text_enhanced(self, page, page_num: int) -> str:
        """Enhanced page text extraction with multiple methods."""
        page_text = ""
        
        try:
            # Method 1: Standard text extraction
            page_text = page.get_text()
            
            # Method 2: If standard extraction is poor, try alternative methods
            if not page_text or len(page_text.strip()) < 50:
                # Try getting text with different parameters
                page_text = page.get_text("text")
                
                if not page_text or len(page_text.strip()) < 50:
                    # Try getting text with HTML-like structure
                    page_text = page.get_text("html")
                    # Clean HTML tags
                    import re
                    page_text = re.sub(r'<[^>]+>', '', page_text)
            
            # Method 3: Extract text blocks for better structure preservation
            if not page_text or len(page_text.strip()) < 50:
                blocks = page.get_text("dict")
                if blocks and "blocks" in blocks:
                    text_parts = []
                    for block in blocks["blocks"]:
                        if "lines" in block:
                            for line in block["lines"]:
                                if "spans" in line:
                                    for span in line["spans"]:
                                        if "text" in span:
                                            text_parts.append(span["text"])
                    page_text = " ".join(text_parts)
            
            # Clean the extracted text
            if page_text:
                page_text = self._clean_text(page_text)
                
                # Add page separator for better structure
                page_text = f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
        except Exception as e:
            self.logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
            page_text = f"\n--- Page {page_num + 1} ---\n[Text extraction failed]\n"
        
        return page_text
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> ExtractedContent:
        """Extract content using pdfplumber (better for tables)."""
        content = ExtractedContent()
        
        with pdfplumber.open(str(pdf_path)) as pdf:
            full_text = ""
            
            # Extract metadata
            content.metadata = {
                'page_count': len(pdf.pages),
                'file_size': pdf_path.stat().st_size
            }
            
            # Extract text and tables from all pages
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                # Extract tables if configured
                if self.config and self.config.extract_tables:
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables or []):
                        if table:
                            content.tables.append({
                                'page': page_num + 1,
                                'table_index': table_idx,
                                'data': table,
                                'headers': table[0] if table else []
                            })
        
        # Parse structured content from full text
        self._parse_structured_content(full_text, content)
        
        return content
    
    def _extract_with_ocr(self, pdf_path: Path) -> ExtractedContent:
        """Extract content using OCR as fallback."""
        content = ExtractedContent()
        
        if not HAS_OCR or not convert_from_path or not pytesseract:
            self.logger.warning("OCR libraries not available - skipping OCR extraction")
            return content
        
        try:
            # Convert PDF to images
            images = convert_from_path(str(pdf_path), dpi=200)
            full_text = ""
            
            for page_num, image in enumerate(images):
                # Perform OCR on each page
                page_text = pytesseract.image_to_string(image, lang='eng')
                full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            # Parse structured content from OCR text
            self._parse_structured_content(full_text, content)
            
            content.metadata = {
                'page_count': len(images),
                'extraction_method': 'OCR',
                'file_size': pdf_path.stat().st_size
            }
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            raise
        
        return content
    
    def _parse_structured_content(self, full_text: str, content: ExtractedContent) -> None:
        """Parse structured content from extracted text."""
        lines = [line.strip() for line in full_text.split('\n') if line.strip()]
        
        # Extract title (usually the first significant line)
        if not content.title:
            content.title = self._extract_title(lines)
        
        # Extract authors
        content.authors = self._extract_authors(lines)
        
        # Extract abstract
        content.abstract = self._extract_abstract(lines)
        
        # Extract sections
        content.sections = self._extract_sections(lines)
        
        # Extract references
        content.references = self._extract_references(lines)
    
    def _extract_title(self, lines: List[str]) -> str:
        """Extract paper title from text lines."""
        # Look for the longest line in the first few lines that looks like a title
        candidates = []
        
        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            # Skip common header elements
            if any(skip in line.lower() for skip in ['page', 'doi:', 'arxiv:', 'abstract']):
                continue
            
            # Title is likely to be longer and contain meaningful words
            if len(line) > 20 and len(line.split()) > 3:
                candidates.append((i, line, len(line)))
        
        if candidates:
            # Return the longest candidate from the early lines
            candidates.sort(key=lambda x: (x[0] < 5, x[2]), reverse=True)
            return candidates[0][1]
        
        return ""
    
    def _extract_authors(self, lines: List[str]) -> List[str]:
        """Extract author names from text lines."""
        authors = []
        
        # Look for patterns common in author lines
        author_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?\s*)*[A-Z][a-z]+)(?:\s*[,;]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?\s*)*[A-Z][a-z]+))*',
            r'([A-Z]\.\s*[A-Z][a-z]+(?:\s+[A-Z]\.\s*[A-Z][a-z]+)*)',
        ]
        
        for line in lines[:50]:  # Check first 50 lines
            for pattern in author_patterns:
                matches = re.findall(pattern, line)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            authors.extend([m for m in match if m])
                        else:
                            authors.append(match)
        
        # Clean and deduplicate authors
        cleaned_authors = []
        for author in authors:
            author = author.strip(' ,;')
            if author and len(author) > 2 and author not in cleaned_authors:
                cleaned_authors.append(author)
        
        return cleaned_authors[:10]  # Limit to reasonable number
    
    def _extract_abstract(self, lines: List[str]) -> str:
        """Extract abstract from text lines."""
        abstract_start = None
        abstract_end = None
        
        # Find abstract section
        for i, line in enumerate(lines):
            if re.match(r'^\s*abstract\s*:?\s*$', line, re.IGNORECASE):
                abstract_start = i + 1
                break
            elif 'abstract' in line.lower() and len(line) < 50:
                abstract_start = i + 1
                break
        
        if abstract_start is None:
            return ""
        
        # Find end of abstract (next section or keywords)
        for i in range(abstract_start, min(abstract_start + 20, len(lines))):
            line = lines[i].lower()
            if any(keyword in line for keyword in ['introduction', 'keywords', '1.', 'i.']):
                abstract_end = i
                break
        
        if abstract_end is None:
            abstract_end = min(abstract_start + 15, len(lines))
        
        # Join abstract lines
        abstract_lines = lines[abstract_start:abstract_end]
        abstract = ' '.join(abstract_lines).strip()
        
        # Clean up abstract
        abstract = re.sub(r'\s+', ' ', abstract)
        
        return abstract
    
    def _extract_sections(self, lines: List[str]) -> Dict[str, str]:
        """Extract sections with improved academic paper structure detection."""
        sections = {}
        current_section = "Title_Abstract"
        current_content = []
        
        # Enhanced section patterns for academic papers
        section_patterns = [
            (r'^(abstract|ABSTRACT)\s*:?\s*$', 'Abstract'),
            (r'^(introduction|INTRODUCTION)\s*:?\s*$', 'Introduction'),
            (r'^(background|BACKGROUND)\s*:?\s*$', 'Background'),
            (r'^(related\s+work|RELATED\s+WORK|literature\s+review|LITERATURE\s+REVIEW)\s*:?\s*$', 'Related_Work'),
            (r'^(methodology|METHODOLOGY|methods|METHODS|approach|APPROACH)\s*:?\s*$', 'Methodology'),
            (r'^(experimental\s+setup|EXPERIMENTAL\s+SETUP|experimental\s+design|EXPERIMENTAL\s+DESIGN)\s*:?\s*$', 'Experimental_Setup'),
            (r'^(results|RESULTS|findings|FINDINGS|experiments?|EXPERIMENTS?)\s*:?\s*$', 'Results'),
            (r'^(discussion|DISCUSSION|analysis|ANALYSIS|interpretation|INTERPRETATION)\s*:?\s*$', 'Discussion'),
            (r'^(conclusion|CONCLUSIONS?|summary|SUMMARY|future\s+work|FUTURE\s+WORK)\s*:?\s*$', 'Conclusion'),
            (r'^(references?|REFERENCES?|bibliography|BIBLIOGRAPHY)\s*:?\s*$', 'References'),
            (r'^(acknowledgments?|ACKNOWLEDGMENTS?|acknowledgements?|ACKNOWLEDGEMENTS?)\s*:?\s*$', 'Acknowledgments'),
            (r'^(appendix|APPENDIX)\s*[A-Z]?\s*:?\s*$', 'Appendix'),
            (r'^(\d+\.?\s*[A-Z][a-zA-Z\s]+)\s*:?\s*$', 'Section'),  # Numbered sections
            (r'^([A-Z][A-Z\s]+)\s*:?\s*$', 'Section'),  # All caps sections
        ]
        
        # Compile patterns for efficiency
        compiled_patterns = [(re.compile(pattern, re.IGNORECASE), section_type) 
                           for pattern, section_type in section_patterns]
        
        for line in lines:
            # Check if this line is a section header
            section_found = False
            for pattern, section_type in compiled_patterns:
                if pattern.match(line.strip()):
                    # Save current section content
                    if current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    
                    # Start new section
                    if section_type == 'Section':
                        # Clean the section name
                        clean_name = self._clean_section_name(line.strip())
                        current_section = clean_name
                    else:
                        current_section = section_type
                    
                    current_content = []
                    section_found = True
                    break
            
            # If not a section header, add to current content
            if not section_found:
                current_content.append(line)
        
        # Save the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Post-process sections for better content organization
        sections = self._post_process_sections(sections)
        
        return sections
    
    def _post_process_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        """Post-process sections to improve content organization."""
        processed_sections = {}
        
        for section_name, content in sections.items():
            # Clean section content
            cleaned_content = self._clean_text(content)
            
            # Skip empty sections
            if not cleaned_content or len(cleaned_content.strip()) < 10:
                continue
            
            # Merge very short sections with previous ones
            if len(cleaned_content.strip()) < 50 and section_name != 'Title_Abstract':
                # Try to merge with previous section
                continue
            
            # Clean section name
            clean_section_name = self._clean_section_name(section_name)
            
            # Ensure we don't have duplicate section names
            if clean_section_name in processed_sections:
                # Append content to existing section
                processed_sections[clean_section_name] += f"\n\n{cleaned_content}"
            else:
                processed_sections[clean_section_name] = cleaned_content
        
        return processed_sections
    
    def _is_section_header(self, line: str) -> bool:
        """Check if a line is likely a section header."""
        line = line.strip()
        
        # Check for numbered sections
        if re.match(r'^\d+\.?\s+[A-Z]', line):
            return True
        
        # Check for common section headers
        headers = ['introduction', 'methodology', 'results', 'discussion', 'conclusion']
        if any(header in line.lower() for header in headers):
            return len(line) < 100  # Headers are usually short
        
        # Check for all caps short lines
        if line.isupper() and len(line.split()) <= 5 and len(line) > 3:
            return True
        
        return False
    
    def _clean_section_name(self, header: str) -> str:
        """Clean and normalize section header name."""
        # Remove numbers and special characters
        cleaned = re.sub(r'^\d+\.?\s*', '', header.strip())
        cleaned = re.sub(r'[^\w\s]', '', cleaned)
        
        # Normalize case
        cleaned = cleaned.lower().strip()
        
        # Map common variations
        mappings = {
            'abstract': 'Abstract',
            'intro': 'Introduction',
            'introduction': 'Introduction',
            'background': 'Background',
            'related work': 'Related Work',
            'methodology': 'Methodology',
            'methods': 'Methodology',
            'approach': 'Methodology',
            'implementation': 'Implementation',
            'results': 'Results',
            'findings': 'Results',
            'experiments': 'Results',
            'evaluation': 'Results',
            'discussion': 'Discussion',
            'analysis': 'Discussion',
            'conclusion': 'Conclusion',
            'conclusions': 'Conclusion',
            'summary': 'Conclusion',
            'future work': 'Future Work',
            'references': 'References',
            'bibliography': 'References'
        }
        
        return mappings.get(cleaned, cleaned.title())
    
    def _extract_references(self, lines: List[str]) -> List[str]:
        """Extract references from text lines."""
        references = []
        in_references = False
        
        for line in lines:
            # Check if we've reached the references section
            if re.match(r'^\s*references?\s*:?\s*$', line, re.IGNORECASE):
                in_references = True
                continue
            
            if in_references:
                # Look for reference patterns
                if re.match(r'^\[\d+\]', line.strip()) or re.match(r'^\d+\.', line.strip()):
                    references.append(line.strip())
                elif line.strip() and not self._is_section_header(line):
                    # Continuation of previous reference
                    if references:
                        references[-1] += " " + line.strip()
        
        return references[:50]  # Limit to reasonable number
    
    def _extract_images_from_page(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract images from a page."""
        images = []
        img_list = page.get_images()
        
        for img_idx, img in enumerate(img_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_data = {
                    'page': page_num + 1,
                    'image_index': img_idx,
                    'width': base_image['width'],
                    'height': base_image['height'],
                    'colorspace': base_image['colorspace'],
                    'size': len(base_image['image'])
                }
                images.append(image_data)
            except Exception as e:
                self.logger.warning(f"Failed to extract image {img_idx} from page {page_num}: {e}")
        
        return images
    
    def _post_process_content(self, content: ExtractedContent, pdf_path: Path) -> ExtractedContent:
        """Post-process extracted content for quality and consistency."""
        # Add file metadata
        from datetime import datetime
        content.metadata.update({
            'filename': pdf_path.name,
            'file_path': str(pdf_path),
            'extraction_timestamp': str(datetime.now())
        })
        
        # Clean up text content
        content.title = self._clean_text(content.title)
        content.abstract = self._clean_text(content.abstract)
        
        for section_name, section_content in content.sections.items():
            content.sections[section_name] = self._clean_text(section_content)
        
        return content
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content with improved encoding handling."""
        if not text:
            return ""
        
        # Handle encoding issues - fix common Unicode problems
        text = self._fix_encoding_issues(text)
        
        # Remove excessive whitespace but preserve meaningful spacing
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double newlines
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\bPage \d+\b', '', text)
        text = re.sub(r'\b\d+\s*$', '', text)  # Remove trailing page numbers
        
        # Fix common OCR errors and ligatures
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text = text.replace('ﬀ', 'ff').replace('ﬃ', 'ffi').replace('ﬄ', 'ffl')
        
        # Preserve mathematical content - don't over-clean
        text = self._preserve_mathematical_content(text)
        
        return text.strip()
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues in PDF text extraction."""
        if not text:
            return text
        
        # Fix common Unicode replacement characters
        text = text.replace('\ufffd', '')  # Remove replacement characters
        
        # Fix common mathematical symbol encoding issues
        # These are common replacements for Greek letters and math symbols
        math_replacements = {
            '\u03b1': 'α',  # Greek alpha
            '\u03b2': 'β',  # Greek beta
            '\u03b3': 'γ',  # Greek gamma
            '\u03b4': 'δ',  # Greek delta
            '\u03b5': 'ε',  # Greek epsilon
            '\u03b6': 'ζ',  # Greek zeta
            '\u03b7': 'η',  # Greek eta
            '\u03b8': 'θ',  # Greek theta
            '\u03b9': 'ι',  # Greek iota
            '\u03ba': 'κ',  # Greek kappa
            '\u03bb': 'λ',  # Greek lambda
            '\u03bc': 'μ',  # Greek mu
            '\u03bd': 'ν',  # Greek nu
            '\u03be': 'ξ',  # Greek xi
            '\u03bf': 'ο',  # Greek omicron
            '\u03c0': 'π',  # Greek pi
            '\u03c1': 'ρ',  # Greek rho
            '\u03c3': 'σ',  # Greek sigma
            '\u03c4': 'τ',  # Greek tau
            '\u03c5': 'υ',  # Greek upsilon
            '\u03c6': 'φ',  # Greek phi
            '\u03c7': 'χ',  # Greek chi
            '\u03c8': 'ψ',  # Greek psi
            '\u03c9': 'ω',  # Greek omega
            '\u0391': 'Α',  # Greek Alpha
            '\u0392': 'Β',  # Greek Beta
            '\u0393': 'Γ',  # Greek Gamma
            '\u0394': 'Δ',  # Greek Delta
            '\u0395': 'Ε',  # Greek Epsilon
            '\u0396': 'Ζ',  # Greek Zeta
            '\u0397': 'Η',  # Greek Eta
            '\u0398': 'Θ',  # Greek Theta
            '\u0399': 'Ι',  # Greek Iota
            '\u039a': 'Κ',  # Greek Kappa
            '\u039b': 'Λ',  # Greek Lambda
            '\u039c': 'Μ',  # Greek Mu
            '\u039d': 'Ν',  # Greek Nu
            '\u039e': 'Ξ',  # Greek Xi
            '\u039f': 'Ο',  # Greek Omicron
            '\u03a0': 'Π',  # Greek Pi
            '\u03a1': 'Ρ',  # Greek Rho
            '\u03a3': 'Σ',  # Greek Sigma
            '\u03a4': 'Τ',  # Greek Tau
            '\u03a5': 'Υ',  # Greek Upsilon
            '\u03a6': 'Φ',  # Greek Phi
            '\u03a7': 'Χ',  # Greek Chi
            '\u03a8': 'Ψ',  # Greek Psi
            '\u03a9': 'Ω',  # Greek Omega
            '\u2211': '∑',  # Summation
            '\u2212': '−',  # Minus sign
            '\u2217': '∗',  # Asterisk operator
            '\u2218': '∘',  # Ring operator
            '\u2219': '∙',  # Bullet operator
            '\u221a': '√',  # Square root
            '\u221b': '∛',  # Cube root
            '\u221c': '∜',  # Fourth root
            '\u221d': '∝',  # Proportional to
            '\u221e': '∞',  # Infinity
            '\u2220': '∠',  # Angle
            '\u2221': '∡',  # Measured angle
            '\u2222': '∢',  # Spherical angle
            '\u2223': '∣',  # Divides
            '\u2224': '∤',  # Does not divide
            '\u2225': '∥',  # Parallel to
            '\u2226': '∦',  # Not parallel to
            '\u2227': '∧',  # Logical and
            '\u2228': '∨',  # Logical or
            '\u2229': '∩',  # Intersection
            '\u222a': '∪',  # Union
            '\u222b': '∫',  # Integral
            '\u222c': '∬',  # Double integral
            '\u222d': '∭',  # Triple integral
            '\u222e': '∮',  # Contour integral
            '\u222f': '∯',  # Surface integral
            '\u2230': '∰',  # Volume integral
            '\u2231': '∱',  # Clockwise integral
            '\u2232': '∲',  # Clockwise contour integral
            '\u2233': '∳',  # Anticlockwise contour integral
            '\u2234': '∴',  # Therefore
            '\u2235': '∵',  # Because
            '\u2236': '∶',  # Ratio
            '\u2237': '∷',  # Proportion
            '\u2238': '∸',  # Dot minus
            '\u2239': '∹',  # Excess
            '\u223a': '∺',  # Geometric proportion
            '\u223b': '∻',  # Homothetic
            '\u223c': '∼',  # Tilde operator
            '\u223d': '∽',  # Reversed tilde
            '\u223e': '∾',  # Inverted lazy s
            '\u223f': '∿',  # Sine wave
            '\u2240': '≀',  # Wreath product
            '\u2241': '≁',  # Not tilde
            '\u2242': '≂',  # Minus tilde
            '\u2243': '≃',  # Asymptotically equal to
            '\u2244': '≄',  # Not asymptotically equal to
            '\u2245': '≅',  # Approximately equal to
            '\u2246': '≆',  # Approximately but not actually equal to
            '\u2247': '≇',  # Neither approximately nor actually equal to
            '\u2248': '≈',  # Almost equal to
            '\u2249': '≉',  # Not almost equal to
            '\u224a': '≊',  # Almost equal or equal to
            '\u224b': '≋',  # Triple tilde
            '\u224c': '≌',  # All equal to
            '\u224d': '≍',  # Equivalent to
            '\u224e': '≎',  # Geometrically equivalent to
            '\u224f': '≏',  # Difference between
            '\u2250': '≐',  # Approaches the limit
            '\u2251': '≑',  # Geometrically equal to
            '\u2252': '≒',  # Approximately equal to or the image of
            '\u2253': '≓',  # Image of or approximately equal to
            '\u2254': '≔',  # Colon equals
            '\u2255': '≕',  # Equals colon
            '\u2256': '≖',  # Ring in equal to
            '\u2257': '≗',  # Ring equal to
            '\u2258': '≘',  # Corresponds to
            '\u2259': '≙',  # Estimates
            '\u225a': '≚',  # Equiangular to
            '\u225b': '≛',  # Star equals
            '\u225c': '≜',  # Delta equal to
            '\u225d': '≝',  # Equal to by definition
            '\u225e': '≞',  # Measured by
            '\u225f': '≟',  # Questioned equal to
            '\u2260': '≠',  # Not equal to
            '\u2261': '≡',  # Identical to
            '\u2262': '≢',  # Not identical to
            '\u2263': '≣',  # Strictly equivalent to
            '\u2264': '≤',  # Less-than or equal to
            '\u2265': '≥',  # Greater-than or equal to
            '\u2266': '≦',  # Less-than over equal to
            '\u2267': '≧',  # Greater-than over equal to
            '\u2268': '≨',  # Less-than but not equal to
            '\u2269': '≩',  # Greater-than but not equal to
            '\u226a': '≪',  # Much less-than
            '\u226b': '≫',  # Much greater-than
            '\u226c': '≬',  # Between
            '\u226d': '≭',  # Not equivalent to
            '\u226e': '≮',  # Not less-than
            '\u226f': '≯',  # Not greater-than
            '\u2270': '≰',  # Neither less-than nor equal to
            '\u2271': '≱',  # Neither greater-than nor equal to
            '\u2272': '≲',  # Less-than or equivalent to
            '\u2273': '≳',  # Greater-than or equivalent to
            '\u2274': '≴',  # Neither less-than nor equivalent to
            '\u2275': '≵',  # Neither greater-than nor equivalent to
            '\u2276': '≶',  # Less-than or greater-than
            '\u2277': '≷',  # Greater-than or less-than
            '\u2278': '≸',  # Neither less-than nor greater-than
            '\u2279': '≹',  # Neither greater-than nor less-than
            '\u227a': '≺',  # Precedes
            '\u227b': '≻',  # Succeeds
            '\u227c': '≼',  # Precedes or equal to
            '\u227d': '≽',  # Succeeds or equal to
            '\u227e': '≾',  # Precedes or equivalent to
            '\u227f': '≿',  # Succeeds or equivalent to
            '\u2280': '⊀',  # Does not precede
            '\u2281': '⊁',  # Does not succeed
            '\u2282': '⊂',  # Subset of
            '\u2283': '⊃',  # Superset of
            '\u2284': '⊄',  # Not a subset of
            '\u2285': '⊅',  # Not a superset of
            '\u2286': '⊆',  # Subset of or equal to
            '\u2287': '⊇',  # Superset of or equal to
            '\u2288': '⊈',  # Neither a subset of nor equal to
            '\u2289': '⊉',  # Neither a superset of nor equal to
            '\u228a': '⊊',  # Subset of with not equal to
            '\u228b': '⊋',  # Superset of with not equal to
            '\u228c': '⊌',  # Multiset
            '\u228d': '⊍',  # Multiset multiplication
            '\u228e': '⊎',  # Multiset union
            '\u228f': '⊏',  # Square image of
            '\u2290': '⊐',  # Square original of
            '\u2291': '⊑',  # Square image of or equal to
            '\u2292': '⊒',  # Square original of or equal to
            '\u2293': '⊓',  # Square cap
            '\u2294': '⊔',  # Square cup
            '\u2295': '⊕',  # Circled plus
            '\u2296': '⊖',  # Circled minus
            '\u2297': '⊗',  # Circled times
            '\u2298': '⊘',  # Circled division slash
            '\u2299': '⊙',  # Circled dot operator
            '\u229a': '⊚',  # Circled ring operator
            '\u229b': '⊛',  # Circled asterisk operator
            '\u229c': '⊜',  # Circled equals
            '\u229d': '⊝',  # Circled dash
            '\u229e': '⊞',  # Squared plus
            '\u229f': '⊟',  # Squared minus
            '\u22a0': '⊠',  # Squared times
            '\u22a1': '⊡',  # Squared dot operator
            '\u22a2': '⊢',  # Right tack
            '\u22a3': '⊣',  # Left tack
            '\u22a4': '⊤',  # Down tack
            '\u22a5': '⊥',  # Up tack
            '\u22a6': '⊦',  # Assertion
            '\u22a7': '⊧',  # Models
            '\u22a8': '⊨',  # True
            '\u22a9': '⊩',  # Forces
            '\u22aa': '⊪',  # Triple vertical bar right turnstile
            '\u22ab': '⊫',  # Double vertical bar double right turnstile
            '\u22ac': '⊬',  # Does not prove
            '\u22ad': '⊭',  # Not true
            '\u22ae': '⊮',  # Does not force
            '\u22af': '⊯',  # Negated double vertical bar double right turnstile
            '\u22b0': '⊰',  # Precedes under relation
            '\u22b1': '⊱',  # Succeeds under relation
            '\u22b2': '⊲',  # Normal subgroup of
            '\u22b3': '⊳',  # Contains as normal subgroup
            '\u22b4': '⊴',  # Normal subgroup of or equal to
            '\u22b5': '⊵',  # Contains as normal subgroup or equal to
            '\u22b6': '⊶',  # Original of
            '\u22b7': '⊷',  # Image of
            '\u22b8': '⊸',  # Multimap
            '\u22b9': '⊹',  # Hermitian conjugate matrix
            '\u22ba': '⊺',  # Intercalate
            '\u22bb': '⊻',  # Xor
            '\u22bc': '⊼',  # Nand
            '\u22bd': '⊽',  # Nor
            '\u22be': '⊾',  # Right angle with arc
            '\u22bf': '⊿',  # Right triangle
            '\u22c0': '⋀',  # N-ary logical and
            '\u22c1': '⋁',  # N-ary logical or
            '\u22c2': '⋂',  # N-ary intersection
            '\u22c3': '⋃',  # N-ary union
            '\u22c4': '⋄',  # Diamond operator
            '\u22c5': '⋅',  # Dot operator
            '\u22c6': '⋆',  # Star operator
            '\u22c7': '⋇',  # Division times
            '\u22c8': '⋈',  # Bowtie
            '\u22c9': '⋉',  # Left normal factor semidirect product
            '\u22ca': '⋊',  # Right normal factor semidirect product
            '\u22cb': '⋋',  # Left semidirect product
            '\u22cc': '⋌',  # Right semidirect product
            '\u22cd': '⋍',  # Reversed tilde equals
            '\u22ce': '⋎',  # Curly logical or
            '\u22cf': '⋏',  # Curly logical and
            '\u22d0': '⋐',  # Double subset
            '\u22d1': '⋑',  # Double superset
            '\u22d2': '⋒',  # Double intersection
            '\u22d3': '⋓',  # Double union
            '\u22d4': '⋔',  # Pitchfork
            '\u22d5': '⋕',  # Equal and parallel to
            '\u22d6': '⋖',  # Less-than with dot
            '\u22d7': '⋗',  # Greater-than with dot
            '\u22d8': '⋘',  # Very much less-than
            '\u22d9': '⋙',  # Very much greater-than
            '\u22da': '⋚',  # Less-than equal to or greater-than
            '\u22db': '⋛',  # Greater-than equal to or less-than
            '\u22dc': '⋜',  # Equal to or less-than
            '\u22dd': '⋝',  # Equal to or greater-than
            '\u22de': '⋞',  # Equal to or precedes
            '\u22df': '⋟',  # Equal to or succeeds
            '\u22e0': '⋠',  # Does not precede or equal
            '\u22e1': '⋡',  # Does not succeed or equal
            '\u22e2': '⋢',  # Not square image of or equal to
            '\u22e3': '⋣',  # Not square original of or equal to
            '\u22e4': '⋤',  # Square image of or not equal to
            '\u22e5': '⋥',  # Square original of or not equal to
            '\u22e6': '⋦',  # Less-than but not equivalent to
            '\u22e7': '⋧',  # Greater-than but not equivalent to
            '\u22e8': '⋨',  # Precedes but not equivalent to
            '\u22e9': '⋩',  # Succeeds but not equivalent to
            '\u22ea': '⋪',  # Not normal subgroup of
            '\u22eb': '⋫',  # Does not contain as normal subgroup
            '\u22ec': '⋬',  # Not normal subgroup of or equal to
            '\u22ed': '⋭',  # Does not contain as normal subgroup or equal
            '\u22ee': '⋮',  # Vertical ellipsis
            '\u22ef': '⋯',  # Midline horizontal ellipsis
            '\u22f0': '⋰',  # Up right diagonal ellipsis
            '\u22f1': '⋱',  # Down right diagonal ellipsis
            '\u22f2': '⋲',  # Element of with long horizontal stroke
            '\u22f3': '⋳',  # Element of with vertical bar at end of horizontal stroke
            '\u22f4': '⋴',  # Small element of with vertical bar at end of horizontal stroke
            '\u22f5': '⋵',  # Element of with dot above
            '\u22f6': '⋶',  # Element of with overbar
            '\u22f7': '⋷',  # Small element of with overbar
            '\u22f8': '⋸',  # Element of with underbar
            '\u22f9': '⋹',  # Element of with two horizontal strokes
            '\u22fa': '⋺',  # Contains with long horizontal stroke
            '\u22fb': '⋻',  # Contains with vertical bar at end of horizontal stroke
            '\u22fc': '⋼',  # Small contains with vertical bar at end of horizontal stroke
            '\u22fd': '⋽',  # Contains with overbar
            '\u22fe': '⋾',  # Small contains with overbar
            '\u22ff': '⋿',  # Contains with underbar
        }
        
        for bad_char, good_char in math_replacements.items():
            text = text.replace(bad_char, good_char)
        
        # Fix common encoding issues with special characters
        text = text.replace('\u00a0', ' ')  # Non-breaking space to regular space
        text = text.replace('\u2013', '-')  # En dash to hyphen
        text = text.replace('\u2014', '--')  # Em dash to double hyphen
        text = text.replace('\u2018', "'")  # Left single quote to apostrophe
        text = text.replace('\u2019', "'")  # Right single quote to apostrophe
        text = text.replace('\u201c', '"')  # Left double quote to quote
        text = text.replace('\u201d', '"')  # Right double quote to quote
        
        return text
    
    def _preserve_mathematical_content(self, text: str) -> str:
        """Preserve mathematical content during cleaning."""
        if not text:
            return text
        
        # Don't over-clean mathematical expressions
        # Preserve common mathematical patterns
        math_patterns = [
            r'\$[^$]+\$',  # Inline math
            r'\\\([^)]+\\\)',  # Inline math
            r'\\\[[^\]]+\\\]',  # Display math
            r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}',  # LaTeX environments
            r'[α-ωΑ-Ω]',  # Greek letters
            r'[∑∫∏√∞≤≥≠≡]',  # Common math symbols
        ]
        
        # For now, just return the text as is to avoid breaking math
        # In a more sophisticated implementation, we would identify and protect math regions
        return text
    
    def save_extracted_content(self, content: ExtractedContent, output_path: str) -> None:
        """Save extracted content to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary for JSON serialization
        content_dict = asdict(content)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(content_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Extracted content saved to {output_path}")
    
    def process_multiple_files(self, pdf_paths: List[str], parallel: bool = False) -> List[ExtractedContent]:
        """Process multiple PDF files."""
        if parallel and self.config:
            max_workers = getattr(self.config, 'max_workers', 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self.extract_from_file, pdf_paths))
            return results
        else:
            return [self.extract_from_file(path) for path in pdf_paths]


class EnhancedPDFProcessor(PDFProcessor):
    """Enhanced PDF processor with advanced academic content extraction."""
    
    def __init__(self, config=None):
        """Initialize enhanced PDF processor."""
        super().__init__(config)
        
        # Initialize specialized models and tools
        self._initialize_models()
        
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
        
        # Compile equation patterns
        self.compiled_equation_patterns = [
            re.compile(p, re.DOTALL | re.IGNORECASE) 
            for p in self.equation_patterns
        ]
        
        # Figure and table detection patterns
        self.figure_keywords = [
            'figure', 'fig', 'chart', 'graph', 'plot', 'diagram', 
            'image', 'illustration', 'schema', 'flowchart'
        ]
        
        self.table_keywords = [
            'table', 'tab', 'matrix', 'data', 'results', 'comparison'
        ]
    
    def _initialize_models(self):
        """Initialize specialized models for enhanced extraction."""
        try:
            # Initialize LayoutParser for document layout analysis
            if HAS_LAYOUTPARSER:
                self.layout_model = lp.Detectron2LayoutModel(
                    'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
                )
            else:
                self.layout_model = None
                self.logger.warning("LayoutParser not available - layout analysis will be disabled")
            
            # Initialize EasyOCR for better text recognition in figures
            if HAS_EASYOCR:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            else:
                self.ocr_reader = None
                self.logger.warning("EasyOCR not available - advanced OCR will be disabled")
            
            self.logger.info("Enhanced models initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize all enhanced models: {e}")
            self.layout_model = None
            self.ocr_reader = None
    
    def extract_from_file(self, pdf_path: str) -> EnhancedContent:
        """
        Enhanced extraction from PDF file with advanced content detection.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            EnhancedContent object with comprehensive extraction
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Processing PDF with enhanced extraction: {pdf_path.name}")
        
        # Start with base extraction
        base_content = super().extract_from_file(pdf_path)
        
        # Convert to enhanced content
        enhanced_content = EnhancedContent(**asdict(base_content))
        
        # Perform layout analysis
        enhanced_content.layout_analysis = self._analyze_document_layout(pdf_path)
        
        # Enhanced equation extraction
        enhanced_content.equations = self._extract_equations_enhanced(pdf_path)
        
        # Enhanced figure extraction
        enhanced_content.enhanced_figures = self._extract_figures_enhanced(pdf_path)
        
        # Enhanced table extraction
        enhanced_content.enhanced_tables = self._extract_tables_enhanced(pdf_path)
        
        # Post-process and validate
        enhanced_content = self._post_process_enhanced_content(enhanced_content, pdf_path)
        
        self.logger.info(f"Enhanced extraction completed for {pdf_path.name}")
        return enhanced_content
    
    def _analyze_document_layout(self, pdf_path: Path) -> Dict[str, Any]:
        """Analyze document layout to identify content regions."""
        layout_info = {
            'pages': [],
            'content_regions': [],
            'reading_order': []
        }
        
        if not HAS_LAYOUTPARSER or self.layout_model is None:
            self.logger.debug("Layout analysis skipped - LayoutParser not available")
            return layout_info
        
        if not self.layout_model:
            return layout_info
        
        try:
            # Convert PDF to images for layout analysis
            images = convert_from_path(str(pdf_path), dpi=150)
            
            for page_num, image in enumerate(images):
                # Convert PIL image to numpy array
                image_np = np.array(image)
                
                # Detect layout elements
                layout = self.layout_model.detect(image_np)
                
                page_info = {
                    'page_number': page_num + 1,
                    'elements': []
                }
                
                for element in layout:
                    element_info = {
                        'type': element.type,
                        'confidence': float(element.score),
                        'bbox': [float(x) for x in element.block.coordinates],
                        'area': float(element.block.area)
                    }
                    page_info['elements'].append(element_info)
                
                layout_info['pages'].append(page_info)
            
        except Exception as e:
            self.logger.warning(f"Layout analysis failed: {e}")
        
        return layout_info
    
    def _extract_equations_enhanced(self, pdf_path: Path) -> List[EnhancedEquation]:
        """Enhanced equation extraction with multiple detection methods."""
        equations = []
        
        try:
            with fitz.open(str(pdf_path)) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Method 1: Text-based equation detection
                    text_equations = self._extract_text_equations(page, page_num)
                    equations.extend(text_equations)
                    
                    # Method 2: Image-based equation detection
                    if self.config and getattr(self.config, 'extract_equation_images', True):
                        image_equations = self._extract_equation_images(page, page_num)
                        equations.extend(image_equations)
                    
                    # Method 3: Vector graphics equation detection
                    vector_equations = self._extract_vector_equations(page, page_num)
                    equations.extend(vector_equations)
        
        except Exception as e:
            self.logger.error(f"Enhanced equation extraction failed: {e}")
        
        return self._deduplicate_equations(equations)
    
    def _extract_text_equations(self, page, page_num: int) -> List[EnhancedEquation]:
        """Extract equations from text content."""
        equations = []
        page_text = page.get_text()
        
        for pattern in self.compiled_equation_patterns:
            matches = pattern.finditer(page_text)
            
            for match in matches:
                latex_text = match.group().strip()
                
                # Clean up LaTeX text
                cleaned_latex = self._clean_latex_equation(latex_text)
                
                if len(cleaned_latex) > 3:  # Filter out very short matches
                    equation = EnhancedEquation(
                        latex=cleaned_latex,
                        text_representation=self._latex_to_text(cleaned_latex),
                        page=page_num + 1,
                        confidence=0.8,
                        equation_type="display" if any(delim in latex_text for delim in ['$$', '\\[', '\\begin']) else "inline"
                    )
                    equations.append(equation)
        
        return equations
    
    def _extract_equation_images(self, page, page_num: int) -> List[EnhancedEquation]:
        """Extract equations from images within the page."""
        equations = []
        
        if not self.ocr_reader:
            return equations
        
        try:
            # Get page as image
            mat = fitz.Matrix(2.0, 2.0)  # Higher resolution for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            image_np = np.array(image)
            
            # Use OCR to detect text regions that might contain equations
            ocr_results = self.ocr_reader.readtext(image_np)
            
            for (bbox, text, confidence) in ocr_results:
                if confidence > 0.5 and self._looks_like_equation(text):
                    # Extract the region containing the equation
                    x1, y1 = int(min([point[0] for point in bbox])), int(min([point[1] for point in bbox]))
                    x2, y2 = int(max([point[0] for point in bbox])), int(max([point[1] for point in bbox]))
                    
                    equation_image = image.crop((x1, y1, x2, y2))
                    
                    # Try to convert to LaTeX if possible
                    latex_text = self._image_to_latex(equation_image)
                    
                    equation = EnhancedEquation(
                        latex=latex_text,
                        text_representation=text,
                        image_data=self._pil_to_bytes(equation_image),
                        bounding_box=(x1/2, y1/2, x2/2, y2/2),  # Adjust for resolution scaling
                        page=page_num + 1,
                        confidence=confidence,
                        equation_type="display"
                    )
                    equations.append(equation)
        
        except Exception as e:
            self.logger.warning(f"Equation image extraction failed for page {page_num}: {e}")
        
        return equations
    
    def _extract_vector_equations(self, page, page_num: int) -> List[EnhancedEquation]:
        """Extract equations from vector graphics."""
        equations = []
        
        try:
            # Get drawing commands from the page
            drawings = page.get_drawings()
            
            for drawing in drawings:
                # Check if drawing might contain mathematical symbols
                if self._drawing_contains_math_symbols(drawing):
                    # Extract the region as image
                    rect = drawing.get('rect')
                    if rect:
                        # Get image of the specific region
                        mat = fitz.Matrix(3.0, 3.0)  # High resolution
                        pix = page.get_pixmap(matrix=mat, clip=rect)
                        img_data = pix.tobytes("png")
                        
                        # Try to recognize mathematical content
                        image = Image.open(io.BytesIO(img_data))
                        latex_text = self._image_to_latex(image)
                        
                        if latex_text and len(latex_text) > 3:
                            equation = EnhancedEquation(
                                latex=latex_text,
                                image_data=img_data,
                                bounding_box=(rect.x0, rect.y0, rect.x1, rect.y1),
                                page=page_num + 1,
                                confidence=0.6,
                                equation_type="display"
                            )
                            equations.append(equation)
        
        except Exception as e:
            self.logger.warning(f"Vector equation extraction failed for page {page_num}: {e}")
        
        return equations
    
    def _extract_figures_enhanced(self, pdf_path: Path) -> List[EnhancedFigure]:
        """Enhanced figure extraction with content analysis."""
        figures = []
        
        try:
            with fitz.open(str(pdf_path)) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Extract images using multiple methods
                    page_figures = []
                    
                    # Method 1: Direct image extraction
                    direct_figures = self._extract_direct_images(page, page_num)
                    page_figures.extend(direct_figures)
                    
                    # Method 2: Layout-based figure detection
                    if self.layout_model:
                        layout_figures = self._extract_layout_figures(page, page_num)
                        page_figures.extend(layout_figures)
                    
                    # Enhance each figure with content analysis
                    for figure in page_figures:
                        enhanced_figure = self._analyze_figure_content(figure)
                        figures.append(enhanced_figure)
        
        except Exception as e:
            self.logger.error(f"Enhanced figure extraction failed: {e}")
        
        return figures
    
    def _extract_direct_images(self, page, page_num: int) -> List[EnhancedFigure]:
        """Extract images directly from page."""
        figures = []
        img_list = page.get_images()
        
        for img_idx, img in enumerate(img_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image['image']
                
                # Create PIL Image for analysis
                image = Image.open(io.BytesIO(image_bytes))
                
                figure = EnhancedFigure(
                    image_data=image_bytes,
                    page=page_num + 1,
                    width=base_image['width'],
                    height=base_image['height']
                )
                figures.append(figure)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract image {img_idx} from page {page_num}: {e}")
        
        return figures
    
    def _extract_layout_figures(self, page, page_num: int) -> List[EnhancedFigure]:
        """Extract figures using layout analysis."""
        figures = []
        
        if not self.layout_model:
            return figures
        
        try:
            # Get page as image
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            image_np = np.array(image)
            
            # Detect layout elements
            layout = self.layout_model.detect(image_np)
            
            for element in layout:
                if element.type == "Figure" and element.score > 0.7:
                    # Extract the figure region
                    bbox = element.block.coordinates
                    x1, y1, x2, y2 = [int(coord/2) for coord in bbox]  # Adjust for resolution
                    
                    figure_image = image.crop((x1, y1, x2, y2))
                    figure_bytes = self._pil_to_bytes(figure_image)
                    
                    figure = EnhancedFigure(
                        image_data=figure_bytes,
                        bounding_box=(x1, y1, x2, y2),
                        page=page_num + 1,
                        width=x2 - x1,
                        height=y2 - y1
                    )
                    figures.append(figure)
        
        except Exception as e:
            self.logger.warning(f"Layout-based figure extraction failed for page {page_num}: {e}")
        
        return figures
    
    def _extract_tables_enhanced(self, pdf_path: Path) -> List[EnhancedTable]:
        """Enhanced table extraction with better structure preservation."""
        tables = []
        
        try:
            # Use both pdfplumber and layout detection
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Method 1: pdfplumber table extraction
                    plumber_tables = self._extract_pdfplumber_tables(page, page_num)
                    tables.extend(plumber_tables)
                    
                    # Method 2: Layout-based table detection
                    if self.layout_model:
                        layout_tables = self._extract_layout_tables(page, page_num)
                        tables.extend(layout_tables)
        
        except Exception as e:
            self.logger.error(f"Enhanced table extraction failed: {e}")
        
        return self._deduplicate_tables(tables)
    
    def _extract_pdfplumber_tables(self, page, page_num: int) -> List[EnhancedTable]:
        """Extract tables using pdfplumber with enhanced settings."""
        tables = []
        
        # Enhanced table extraction settings
        table_settings = {
            "vertical_strategy": "lines_strict",
            "horizontal_strategy": "lines_strict",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
            "intersection_tolerance": 3,
            "text_tolerance": 3,
        }
        
        extracted_tables = page.extract_tables(table_settings=table_settings)
        
        for table_idx, table_data in enumerate(extracted_tables or []):
            if table_data and len(table_data) > 1:  # At least header + one row
                # Clean and process table data
                cleaned_data = self._clean_table_data(table_data)
                
                if len(cleaned_data) > 0:
                    headers = cleaned_data[0] if len(cleaned_data) > 0 else []
                    data_rows = cleaned_data[1:] if len(cleaned_data) > 1 else []
                    
                    # Try to find table caption
                    caption = self._find_table_caption(page, table_idx)
                    
                    table = EnhancedTable(
                        data=data_rows,
                        headers=headers,
                        caption=caption,
                        page=page_num + 1,
                        num_rows=len(data_rows),
                        num_cols=len(headers) if headers else 0,
                        confidence=0.8,
                        formatting_preserved=True
                    )
                    
                    # Analyze table content to determine type
                    table.table_type = self._analyze_table_type(cleaned_data)
                    
                    tables.append(table)
        
        return tables
    
    # Helper methods for enhanced extraction
    
    def _clean_latex_equation(self, latex_text: str) -> str:
        """Clean and normalize LaTeX equation text."""
        # Remove outer delimiters
        latex_text = latex_text.strip()
        
        # Remove common delimiters
        if latex_text.startswith('$$') and latex_text.endswith('$$'):
            latex_text = latex_text[2:-2]
        elif latex_text.startswith('$') and latex_text.endswith('$'):
            latex_text = latex_text[1:-1]
        elif latex_text.startswith('\\[') and latex_text.endswith('\\]'):
            latex_text = latex_text[2:-2]
        elif latex_text.startswith('\\(') and latex_text.endswith('\\)'):
            latex_text = latex_text[2:-2]
        
        # Remove environment tags for simple cases
        for env in ['equation', 'align', 'eqnarray', 'gather', 'multline']:
            if latex_text.startswith(f'\\begin{{{env}}}') and latex_text.endswith(f'\\end{{{env}}}'):
                latex_text = latex_text[len(f'\\begin{{{env}}}'):-len(f'\\end{{{env}}}')]
        
        return latex_text.strip()
    
    def _latex_to_text(self, latex_text: str) -> str:
        """Convert LaTeX equation to readable text representation."""
        try:
            # Try to parse with sympy
            expr = parse_latex(latex_text)
            return str(expr)
        except:
            # Fallback: basic text conversion
            text = latex_text
            
            # Replace common LaTeX commands with text
            replacements = {
                r'\\frac\{([^}]+)\}\{([^}]+)\}': r'(\1)/(\2)',
                r'\\sqrt\{([^}]+)\}': r'sqrt(\1)',
                r'\\sum': 'sum',
                r'\\int': 'integral',
                r'\\alpha': 'alpha',
                r'\\beta': 'beta',
                r'\\gamma': 'gamma',
                r'\\delta': 'delta',
                r'\\epsilon': 'epsilon',
                r'\\theta': 'theta',
                r'\\lambda': 'lambda',
                r'\\mu': 'mu',
                r'\\pi': 'pi',
                r'\\sigma': 'sigma',
                r'\\phi': 'phi',
                r'\\psi': 'psi',
                r'\\omega': 'omega',
                r'\\_': '_',
                r'\\': '',
                r'\{': '(',
                r'\}': ')',
            }
            
            for pattern, replacement in replacements.items():
                text = re.sub(pattern, replacement, text)
            
            return text
    
    def _looks_like_equation(self, text: str) -> bool:
        """Check if text looks like a mathematical equation."""
        # Mathematical indicators
        math_indicators = [
            r'[=<>≤≥≠±∞∫∑∏√]',  # Mathematical symbols
            r'[α-ωΑ-Ω]',  # Greek letters
            r'\b\d+\.\d+\b',  # Decimal numbers
            r'[a-zA-Z]\^\d+',  # Exponents
            r'[a-zA-Z]_\d+',  # Subscripts
            r'\([^)]*[a-zA-Z][^)]*\)',  # Variables in parentheses
            r'\b(sin|cos|tan|log|ln|exp)\b',  # Mathematical functions
        ]
        
        # Check for mathematical patterns
        math_score = 0
        for pattern in math_indicators:
            if re.search(pattern, text):
                math_score += 1
        
        # Also check for ratio of non-alphabetic characters
        non_alpha = len([c for c in text if not c.isalpha() and not c.isspace()])
        total_chars = len(text.replace(' ', ''))
        
        if total_chars > 0:
            non_alpha_ratio = non_alpha / total_chars
        else:
            non_alpha_ratio = 0
        
        return math_score >= 2 or non_alpha_ratio > 0.3
    
    def _image_to_latex(self, image: Image.Image) -> str:
        """Convert equation image to LaTeX (placeholder for specialized service)."""
        # This would typically use a service like Mathpix OCR
        # For now, return empty string as placeholder
        # In a real implementation, you would:
        # 1. Use Mathpix API
        # 2. Use a local math OCR model
        # 3. Use computer vision to detect mathematical symbols
        
        try:
            # Basic OCR to get text representation
            if self.ocr_reader:
                results = self.ocr_reader.readtext(np.array(image))
                text = ' '.join([result[1] for result in results if result[2] > 0.5])
                
                # Try to convert basic mathematical expressions
                if self._looks_like_equation(text):
                    return self._text_to_basic_latex(text)
        except:
            pass
        
        return ""
    
    def _text_to_basic_latex(self, text: str) -> str:
        """Convert text to basic LaTeX representation."""
        # Basic conversions for common mathematical expressions
        latex = text
        
        # Convert fractions
        latex = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', latex)
        
        # Convert exponents
        latex = re.sub(r'([a-zA-Z0-9]+)\^([a-zA-Z0-9]+)', r'\1^{\2}', latex)
        
        # Convert subscripts
        latex = re.sub(r'([a-zA-Z0-9]+)_([a-zA-Z0-9]+)', r'\1_{\2}', latex)
        
        return latex
    
    def _drawing_contains_math_symbols(self, drawing: Dict) -> bool:
        """Check if a drawing might contain mathematical symbols."""
        # This is a simplified heuristic
        # In practice, you'd analyze the vector graphics more thoroughly
        
        # Check if drawing has multiple small elements (common in equations)
        items = drawing.get('items', [])
        
        if len(items) > 3:  # Multiple elements might indicate equation
            return True
        
        # Check for curved elements (common in mathematical symbols)
        for item in items:
            if item.get('type') == 'c':  # Curve
                return True
        
        return False
    
    def _pil_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes."""
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        return img_buffer.getvalue()
    
    def _analyze_figure_content(self, figure: EnhancedFigure) -> EnhancedFigure:
        """Analyze figure content to determine type and extract text."""
        try:
            image = Image.open(io.BytesIO(figure.image_data))
            
            # Extract text from figure using OCR
            if self.ocr_reader:
                ocr_results = self.ocr_reader.readtext(np.array(image))
                extracted_text = ' '.join([result[1] for result in ocr_results if result[2] > 0.5])
                figure.extracted_text = extracted_text
                
                # Check if figure contains equations
                figure.contains_equations = any(
                    self._looks_like_equation(result[1]) 
                    for result in ocr_results if result[2] > 0.5
                )
            
            # Classify figure type based on visual characteristics
            figure.figure_type = self._classify_figure_type(image)
            
            # Generate description based on content
            figure.description = self._generate_figure_description(figure)
            
        except Exception as e:
            self.logger.warning(f"Figure content analysis failed: {e}")
        
        return figure
    
    def _classify_figure_type(self, image: Image.Image) -> str:
        """Classify figure type based on visual characteristics."""
        # Convert to numpy array for analysis
        img_array = np.array(image.convert('RGB'))
        
        # Basic heuristics for figure classification
        height, width = img_array.shape[:2]
        
        # Check aspect ratio
        aspect_ratio = width / height
        
        # Check color distribution
        gray_image = image.convert('L')
        histogram = gray_image.histogram()
        
        # Simple classification based on characteristics
        if aspect_ratio > 2:
            return "chart"  # Wide images often charts
        elif len(set(histogram)) < 50:  # Low color diversity
            return "diagram"
        elif aspect_ratio < 0.5:
            return "plot"  # Tall images often plots
        else:
            return "figure"
    
    def _generate_figure_description(self, figure: EnhancedFigure) -> str:
        """Generate description for figure based on extracted content."""
        description_parts = []
        
        if figure.figure_type:
            description_parts.append(f"Type: {figure.figure_type}")
        
        if figure.extracted_text:
            # Summarize extracted text
            text_words = figure.extracted_text.split()
            if len(text_words) > 10:
                description_parts.append(f"Contains text with key terms: {', '.join(text_words[:5])}")
            else:
                description_parts.append(f"Text: {figure.extracted_text}")
        
        if figure.contains_equations:
            description_parts.append("Contains mathematical equations")
        
        return "; ".join(description_parts) if description_parts else "No description available"
    
    def _clean_table_data(self, table_data: List[List[str]]) -> List[List[str]]:
        """Clean and normalize table data."""
        cleaned_data = []
        
        for row in table_data:
            if row:  # Skip empty rows
                cleaned_row = []
                for cell in row:
                    if cell is not None:
                        # Clean cell content
                        cleaned_cell = str(cell).strip()
                        cleaned_cell = re.sub(r'\s+', ' ', cleaned_cell)
                        cleaned_row.append(cleaned_cell)
                    else:
                        cleaned_row.append("")
                
                if any(cell for cell in cleaned_row):  # Skip completely empty rows
                    cleaned_data.append(cleaned_row)
        
        return cleaned_data
    
    def _find_table_caption(self, page, table_idx: int) -> str:
        """Find caption for a table."""
        # Extract text from page and look for table captions
        text = page.extract_text()
        lines = text.split('\n')
        
        # Look for table caption patterns
        caption_patterns = [
            rf'Table\s+{table_idx + 1}[:.]\s*(.+)',
            rf'Tab\.\s+{table_idx + 1}[:.]\s*(.+)',
            r'Table\s+\d+[:.]\s*(.+)',
        ]
        
        for line in lines:
            for pattern in caption_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        return ""
    
    def _analyze_table_type(self, table_data: List[List[str]]) -> str:
        """Analyze table content to determine its type."""
        if not table_data:
            return "unknown"
        
        # Check for numerical data (results table)
        total_cells = sum(len(row) for row in table_data)
        numeric_cells = 0
        
        for row in table_data:
            for cell in row:
                if re.search(r'\d+\.?\d*%?', cell):
                    numeric_cells += 1
        
        numeric_ratio = numeric_cells / total_cells if total_cells > 0 else 0
        
        if numeric_ratio > 0.5:
            return "results"
        elif any("comparison" in str(cell).lower() for row in table_data for cell in row):
            return "comparison"
        else:
            return "data"
    
    def _extract_layout_tables(self, page, page_num: int) -> List[EnhancedTable]:
        """Extract tables using layout analysis."""
        tables = []
        
        if not self.layout_model:
            return tables
        
        # This would use layout detection to find table regions
        # Implementation would be similar to figure extraction
        # but focused on table-like structures
        
        return tables
    
    def _deduplicate_equations(self, equations: List[EnhancedEquation]) -> List[EnhancedEquation]:
        """Remove duplicate equations."""
        seen = set()
        unique_equations = []
        
        for eq in equations:
            # Create a signature for the equation
            signature = (eq.latex, eq.page, eq.equation_type)
            
            if signature not in seen:
                seen.add(signature)
                unique_equations.append(eq)
        
        return unique_equations
    
    def _deduplicate_tables(self, tables: List[EnhancedTable]) -> List[EnhancedTable]:
        """Remove duplicate tables."""
        seen = set()
        unique_tables = []
        
        for table in tables:
            # Create a signature for the table
            signature = (table.page, table.num_rows, table.num_cols, str(table.headers))
            
            if signature not in seen:
                seen.add(signature)
                unique_tables.append(table)
        
        return unique_tables
    
    def _post_process_enhanced_content(self, content: EnhancedContent, pdf_path: Path) -> EnhancedContent:
        """Post-process enhanced content for quality and consistency."""
        # Call parent post-processing
        content = super()._post_process_content(content, pdf_path)
        
        # Add enhanced metadata
        content.metadata.update({
            'enhanced_extraction': True,
            'equations_found': len(content.equations),
            'figures_found': len(content.enhanced_figures),
            'tables_found': len(content.enhanced_tables),
            'layout_analyzed': bool(content.layout_analysis.get('pages'))
        })
        
        return content 