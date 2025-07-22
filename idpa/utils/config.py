"""
Configuration management for IDPA.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from dotenv import load_dotenv


@dataclass
class Config:
    """Configuration settings for IDPA."""
    
    # Paths
    data_dir: str = "data"
    outputs_dir: str = "outputs"
    vector_store_dir: str = "vector_store"
    log_file: str = "logs/idpa.log"
    
    # PDF Processing
    max_file_size_mb: int = 100
    ocr_fallback: bool = True
    preserve_images: bool = True
    extract_tables: bool = True
    
    # Enhanced PDF Processing
    use_enhanced_extraction: bool = True
    extract_equations: bool = True
    extract_equation_images: bool = True
    equation_confidence_threshold: float = 0.5
    use_layout_analysis: bool = True
    layout_confidence_threshold: float = 0.7
    
    # Figure Processing
    extract_figure_content: bool = True
    figure_ocr_enabled: bool = True
    figure_classification_enabled: bool = True
    min_figure_size: int = 100  # Minimum size in pixels
    
    # Table Processing
    enhanced_table_extraction: bool = True
    table_structure_preservation: bool = True
    table_caption_extraction: bool = True
    
    # Text Processing
    chunk_size: int = 256  # Reduced from 512 for better precision
    chunk_overlap: int = 128  # 50% overlap (128/256)
    min_chunk_size: int = 50
    preserve_equation_context: bool = True
    equation_context_window: int = 100  # Characters around equations
    
    # Semantic Chunking Settings
    use_semantic_chunking: bool = True
    semantic_break_patterns: List[str] = None  # Will be set in __post_init__
    preserve_section_boundaries: bool = True
    max_section_chunk_size: int = 512  # Maximum size for section-based chunks
    
    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"  # or "cuda" if available
    use_multimodal_embeddings: bool = False
    
    # Vector Store
    similarity_threshold: float = 0.15
    max_retrieval_results: int = 10
    
    # LLM Settings
    llm_provider: str = "openai"  # "openai", "local", "huggingface"
    openai_model: str = "gpt-4"
    openai_api_key: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.1
    
    # Enhanced Query Processing
    equation_aware_search: bool = True
    figure_aware_search: bool = True
    table_aware_search: bool = True
    
    # Query Processing
    query_timeout: int = 30
    enable_caching: bool = True
    
    # Parallel Processing
    max_workers: int = 4
    batch_size: int = 10
    
    # Logging
    log_level: str = "INFO"
    console_logging: bool = True
    
    # External Services (for advanced features)
    mathpix_api_key: Optional[str] = None
    mathpix_api_id: Optional[str] = None
    use_mathpix_ocr: bool = False
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        # Load environment variables
        load_dotenv()
        
        # Initialize semantic break patterns for academic papers
        if self.semantic_break_patterns is None:
            self.semantic_break_patterns = [
                r'\b(?:Abstract|ABSTRACT)\b',
                r'\b(?:Introduction|INTRODUCTION)\b',
                r'\b(?:Related Work|RELATED WORK|Literature Review|LITERATURE REVIEW)\b',
                r'\b(?:Methodology|METHODOLOGY|Methods|METHODS|Approach|APPROACH)\b',
                r'\b(?:Experimental Setup|EXPERIMENTAL SETUP|Experimental Design|EXPERIMENTAL DESIGN)\b',
                r'\b(?:Results|RESULTS|Experimental Results|EXPERIMENTAL RESULTS)\b',
                r'\b(?:Discussion|DISCUSSION)\b',
                r'\b(?:Conclusion|CONCLUSION|Conclusions|CONCLUSIONS)\b',
                r'\b(?:Future Work|FUTURE WORK|Future Directions|FUTURE DIRECTIONS)\b',
                r'\b(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY)\b',
                r'\b(?:Appendix|APPENDIX|Appendices|APPENDICES)\b',
                r'\b(?:Acknowledgments|ACKNOWLEDGMENTS|Acknowledgements|ACKNOWLEDGEMENTS)\b',
                r'^\d+\.\s+[A-Z][^.]*$',  # Numbered sections like "1. Introduction"
                r'^[A-Z][A-Z\s]+\n',  # ALL CAPS section headers
                r'^\d+\.\d+\s+[A-Z][^.]*$',  # Subsections like "1.1 Background"
            ]
        
        # Override with environment variables if present
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        self.mathpix_api_key = os.getenv("MATHPIX_API_KEY", self.mathpix_api_key)
        self.mathpix_api_id = os.getenv("MATHPIX_API_ID", self.mathpix_api_id)
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.llm_provider = os.getenv("LLM_PROVIDER", self.llm_provider)
        
        # Auto-detect CUDA availability
        try:
            import torch
            if torch.cuda.is_available() and self.embedding_device == "auto":
                self.embedding_device = "cuda"
        except ImportError:
            pass
        
        # Create directories
        for dir_path in [self.data_dir, self.outputs_dir, self.vector_store_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Don't save sensitive information
        config_data = asdict(self)
        sensitive_keys = ['openai_api_key', 'mathpix_api_key', 'mathpix_api_id']
        for key in sensitive_keys:
            config_data.pop(key, None)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
    
    def validate(self) -> None:
        """Validate configuration settings."""
        errors = []
        
        # Check required directories
        if not Path(self.data_dir).exists():
            errors.append(f"Data directory does not exist: {self.data_dir}")
        
        # Check LLM configuration
        if self.llm_provider == "openai" and not self.openai_api_key:
            errors.append("OpenAI API key is required when using OpenAI provider")
        
        # Check numeric values
        if self.chunk_size <= 0:
            errors.append("Chunk size must be positive")
        
        if self.chunk_overlap >= self.chunk_size:
            errors.append("Chunk overlap must be less than chunk size")
        
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            errors.append("Similarity threshold must be between 0 and 1")
        
        # Validate enhanced extraction settings
        if self.equation_confidence_threshold < 0 or self.equation_confidence_threshold > 1:
            errors.append("Equation confidence threshold must be between 0 and 1")
        
        if self.layout_confidence_threshold < 0 or self.layout_confidence_threshold > 1:
            errors.append("Layout confidence threshold must be between 0 and 1")
        
        if self.min_figure_size < 0:
            errors.append("Minimum figure size must be non-negative")
        
        # Check Mathpix configuration if enabled
        if self.use_mathpix_ocr and (not self.mathpix_api_key or not self.mathpix_api_id):
            errors.append("Mathpix API credentials required when Mathpix OCR is enabled")
        
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert relative path to absolute path based on config."""
        return Path(relative_path).resolve()
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def get_extraction_config(self) -> Dict[str, Any]:
        """Get configuration specific to content extraction."""
        return {
            'enhanced_extraction': self.use_enhanced_extraction,
            'extract_equations': self.extract_equations,
            'extract_equation_images': self.extract_equation_images,
            'equation_confidence_threshold': self.equation_confidence_threshold,
            'use_layout_analysis': self.use_layout_analysis,
            'layout_confidence_threshold': self.layout_confidence_threshold,
            'extract_figure_content': self.extract_figure_content,
            'figure_ocr_enabled': self.figure_ocr_enabled,
            'figure_classification_enabled': self.figure_classification_enabled,
            'min_figure_size': self.min_figure_size,
            'enhanced_table_extraction': self.enhanced_table_extraction,
            'table_structure_preservation': self.table_structure_preservation,
            'table_caption_extraction': self.table_caption_extraction,
            'preserve_equation_context': self.preserve_equation_context,
            'equation_context_window': self.equation_context_window,
            'use_mathpix_ocr': self.use_mathpix_ocr,
            'mathpix_api_key': self.mathpix_api_key,
            'mathpix_api_id': self.mathpix_api_id,
        }
    
    def get_processing_mode(self) -> str:
        """Get the current processing mode based on configuration."""
        if self.use_enhanced_extraction:
            return "enhanced"
        else:
            return "standard"
    
    def enable_enhanced_mode(self) -> None:
        """Enable all enhanced extraction features."""
        self.use_enhanced_extraction = True
        self.extract_equations = True
        self.extract_equation_images = True
        self.use_layout_analysis = True
        self.extract_figure_content = True
        self.figure_ocr_enabled = True
        self.figure_classification_enabled = True
        self.enhanced_table_extraction = True
        self.table_structure_preservation = True
        self.table_caption_extraction = True
        self.preserve_equation_context = True
        self.equation_aware_search = True
        self.figure_aware_search = True
        self.table_aware_search = True
    
    def disable_enhanced_mode(self) -> None:
        """Disable enhanced extraction features for faster processing."""
        self.use_enhanced_extraction = False
        self.extract_equations = False
        self.extract_equation_images = False
        self.use_layout_analysis = False
        self.extract_figure_content = False
        self.figure_ocr_enabled = False
        self.figure_classification_enabled = False
        self.enhanced_table_extraction = False
        self.equation_aware_search = False
        self.figure_aware_search = False
        self.table_aware_search = False 