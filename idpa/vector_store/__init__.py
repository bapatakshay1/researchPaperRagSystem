"""Vector storage and retrieval module."""

# Import base class from enhanced version
from .enhanced_vector_store import FAISSVectorStore as FAISSStore, EnhancedVectorStore

__all__ = ['FAISSStore', 'EnhancedVectorStore'] 