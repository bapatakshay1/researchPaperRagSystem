"""Text chunking and embedding module."""

# Import base classes from enhanced versions
from .multimodal_embedder import Embedder as DocumentEmbedder, MultiModalEmbedder
from .enhanced_chunker import TextChunker, EnhancedTextChunker, TextChunk, DocumentMetadata

__all__ = ['DocumentEmbedder', 'MultiModalEmbedder', 'TextChunker', 'EnhancedTextChunker', 'TextChunk', 'DocumentMetadata'] 