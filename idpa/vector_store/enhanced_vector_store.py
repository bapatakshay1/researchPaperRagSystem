"""
Enhanced vector store for multi-modal academic content.
"""

import numpy as np
import faiss
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import asdict
import threading
from collections import defaultdict

from utils.logger import get_logger
from embedding.enhanced_chunker import EnhancedTextChunk, TextChunk, DocumentMetadata
from embedding.multimodal_embedder import MultiModalEmbedding, MultiModalEmbedder


class FAISSVectorStore:
    """FAISS-based vector store for semantic search with enhanced document tracking."""
    
    def __init__(self, store_path: str = None, embedding_dim: int = None, config=None):
        """Initialize FAISS store."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Store settings
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.15) if config else 0.15
        self.max_retrieval_results = getattr(config, 'max_retrieval_results', 10) if config else 10
        
        # FAISS components
        self.index = None
        self.chunks_metadata = {}  # maps vector index to TextChunk
        self.chunk_id_to_index = {}  # maps chunk_id to vector index
        self.is_trained = False
        
        # Enhanced document tracking
        self.document_registry = {}  # maps document_id to DocumentMetadata
        self.document_chunks = defaultdict(list)  # maps document_id to list of chunk indices
        self.document_hashes = {}  # maps content_hash to document_id for duplicate detection
        
        # Thread safety
        self._lock = threading.RLock()
        
        # File paths - use provided store_path or default
        if store_path:
            self.store_dir = Path(store_path)
        else:
            self.store_dir = Path("vector_store")
        self.index_file = self.store_dir / "faiss_store.index"
        self.metadata_file = self.store_dir / "faiss_store_metadata.pkl" 
        self.info_file = self.store_dir / "faiss_store_info.json"
        
    def create_index(self, dimension: int):
        """Create a new FAISS index."""
        self.logger.info(f"Creating FAISS index: flat, dimension: {dimension}")
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        self.is_trained = True
        
    def add_embeddings(self, embeddings: Dict[str, np.ndarray], chunks: List[TextChunk]) -> None:
        """Add embeddings and chunks to the index with enhanced document tracking."""
        if not embeddings or not chunks:
            self.logger.warning("No embeddings or chunks provided")
            return
            
        with self._lock:
            # Initialize index if needed
            if self.index is None:
                first_embedding = next(iter(embeddings.values()))
                self.create_index(len(first_embedding))
                
            # Prepare data
            chunk_id_to_chunk = {chunk.chunk_id: chunk for chunk in chunks}
            vectors = []
            chunk_ids = []
            
            # Track documents and detect duplicates
            documents_in_batch = {}
            
            for chunk_id, embedding in embeddings.items():
                if chunk_id in chunk_id_to_chunk:
                    chunk = chunk_id_to_chunk[chunk_id]
                    
                    # Register document metadata
                    if chunk.document_metadata and chunk.document_id not in self.document_registry:
                        # Check for duplicates by content hash
                        if chunk.document_metadata.content_hash in self.document_hashes:
                            existing_doc_id = self.document_hashes[chunk.document_metadata.content_hash]
                            self.logger.warning(f"Document with same content hash already exists: {existing_doc_id}")
                        else:
                            self.document_registry[chunk.document_id] = chunk.document_metadata
                            self.document_hashes[chunk.document_metadata.content_hash] = chunk.document_id
                            documents_in_batch[chunk.document_id] = chunk.document_metadata
                    
                    vectors.append(embedding)
                    chunk_ids.append(chunk_id)
                    
            if not vectors:
                self.logger.warning("No valid embedding-chunk pairs found")
                return
                
            # Convert and normalize for cosine similarity
            vectors_array = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(vectors_array)
            
            # Add to index
            start_idx = self.index.ntotal
            self.index.add(vectors_array)
            
            # Update metadata and document tracking
            for i, chunk_id in enumerate(chunk_ids):
                vector_idx = start_idx + i
                chunk = chunk_id_to_chunk[chunk_id]
                
                self.chunks_metadata[vector_idx] = chunk
                self.chunk_id_to_index[chunk_id] = vector_idx
                
                # Track chunks by document
                self.document_chunks[chunk.document_id].append(vector_idx)
                
            # Log document information
            for doc_id, doc_metadata in documents_in_batch.items():
                chunk_count = len([c for c in chunks if c.document_id == doc_id])
                self.logger.info(f"Added document '{doc_metadata.title}' (ID: {doc_id}) with {chunk_count} chunks")
                
            self.logger.info(f"Added {len(vectors)} embeddings to FAISS index. Total: {self.index.ntotal}")
            
    def search(self, query_embedding: np.ndarray, top_k: int = None, similarity_threshold: float = None, 
               document_filter: Set[str] = None) -> List[Tuple[TextChunk, float]]:
        """
        Search for similar chunks with optional document filtering.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            document_filter: Set of document IDs to restrict search to
            
        Returns:
            List of (TextChunk, similarity_score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            self.logger.warning("Index is empty or not initialized")
            return []
            
        top_k = top_k or self.max_retrieval_results
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        with self._lock:
            # Normalize query
            query_norm = query_embedding.copy().astype(np.float32)
            query_norm = query_norm.reshape(1, -1)
            faiss.normalize_L2(query_norm)
            
            # If document filter is specified, we need to search more broadly 
            # and then filter results
            search_k = top_k * 3 if document_filter else top_k
            
            # Search
            similarities, indices = self.index.search(query_norm, search_k)
            
            # Process results
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx == -1:
                    break
                    
                if similarity >= similarity_threshold:
                    chunk = self.chunks_metadata.get(idx)
                    if chunk:
                        # Apply document filter if specified
                        if document_filter is None or chunk.document_id in document_filter:
                            results.append((chunk, float(similarity)))
                            
                            # Stop when we have enough results
                            if len(results) >= top_k:
                                break
                        
            self.logger.debug(f"Found {len(results)} results above threshold {similarity_threshold}")
            return results
            
    def search_by_text(self, query_text: str, embedder, top_k: int = None, 
                      similarity_threshold: float = None, document_filter: Set[str] = None) -> List[Tuple[TextChunk, float]]:
        """Search using text query with optional document filtering."""
        query_embedding = embedder.embed_query(query_text)
        return self.search(query_embedding, top_k, similarity_threshold, document_filter)
    
    def search_within_document(self, query_text: str, document_id: str, embedder, 
                              top_k: int = None) -> List[Tuple[TextChunk, float]]:
        """Search within a specific document."""
        if document_id not in self.document_registry:
            self.logger.warning(f"Document {document_id} not found in registry")
            return []
        
        return self.search_by_text(query_text, embedder, top_k, 
                                  document_filter={document_id})
    
    def get_document_chunks(self, document_id: str) -> List[TextChunk]:
        """Get all chunks from a specific document."""
        if document_id not in self.document_chunks:
            return []
        
        chunks = []
        for vector_idx in self.document_chunks[document_id]:
            chunk = self.chunks_metadata.get(vector_idx)
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def get_documents_list(self) -> List[DocumentMetadata]:
        """Get list of all registered documents."""
        return list(self.document_registry.values())
    
    def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get metadata for a specific document."""
        return self.document_registry.get(document_id)
    
    def is_document_duplicate(self, content_hash: str) -> Optional[str]:
        """Check if a document with the same content hash already exists."""
        return self.document_hashes.get(content_hash)
        
    def save(self):
        """Save the FAISS index and metadata."""
        if self.index is None:
            self.logger.warning("No index to save")
            return
            
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata including enhanced document tracking
            metadata_to_save = {
                'chunks_metadata': self.chunks_metadata,
                'chunk_id_to_index': self.chunk_id_to_index,
                'document_registry': self.document_registry,
                'document_chunks': dict(self.document_chunks),
                'document_hashes': self.document_hashes
            }
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata_to_save, f)
                
            # Save info with document statistics
            info = {
                'total_vectors': self.index.ntotal,
                'dimension': self.index.d if self.index else 0,
                'similarity_threshold': self.similarity_threshold,
                'is_trained': self.is_trained,
                'total_documents': len(self.document_registry),
                'document_list': [
                    {
                        'document_id': doc_id,
                        'title': metadata.title,
                        'chunk_count': len(self.document_chunks.get(doc_id, []))
                    }
                    for doc_id, metadata in self.document_registry.items()
                ]
            }
            with open(self.info_file, 'w') as f:
                json.dump(info, f, indent=2)
                
        self.logger.info(f"Saved FAISS store to {self.store_dir}/faiss_store.* with {len(self.document_registry)} documents")
        
    def load(self):
        """Load the FAISS index and metadata."""
        if not self.index_file.exists() or not self.metadata_file.exists():
            raise FileNotFoundError("FAISS store files not found")
            
        with self._lock:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))
            
            # Load metadata
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                self.chunks_metadata = metadata['chunks_metadata']
                self.chunk_id_to_index = metadata['chunk_id_to_index']
                
                # Load enhanced document tracking (with backward compatibility)
                self.document_registry = metadata.get('document_registry', {})
                self.document_chunks = defaultdict(list, metadata.get('document_chunks', {}))
                self.document_hashes = metadata.get('document_hashes', {})
                
            # Load info if available
            if self.info_file.exists():
                with open(self.info_file, 'r') as f:
                    info = json.load(f)
                    self.is_trained = info.get('is_trained', True)
                    
        doc_count = len(self.document_registry)
        self.logger.info(f"Loaded FAISS store from {self.store_dir}/faiss_store.* with {self.index.ntotal} vectors from {doc_count} documents")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector store."""
        if self.index is None:
            return {'total_vectors': 0, 'dimension': 0, 'total_documents': 0}
        
        # Calculate document statistics
        doc_chunk_counts = {doc_id: len(chunks) for doc_id, chunks in self.document_chunks.items()}
        
        stats = {
            'total_vectors': self.index.ntotal,
            'dimension': self.index.d,
            'similarity_threshold': self.similarity_threshold,
            'chunks_count': len(self.chunks_metadata),
            'total_documents': len(self.document_registry),
            'documents': [
                {
                    'document_id': doc_id,
                    'title': metadata.title,
                    'chunk_count': doc_chunk_counts.get(doc_id, 0),
                    'file_name': metadata.file_name,
                    'content_hash': metadata.content_hash[:16] + '...' if metadata.content_hash else None,
                    'processing_timestamp': metadata.processing_timestamp
                }
                for doc_id, metadata in self.document_registry.items()
            ]
        }
        
        return stats

    def get_chunk_info(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get chunk information by vector index."""
        if chunk_id in self.chunks_metadata:
            chunk = self.chunks_metadata[chunk_id]
            return {
                'content': chunk.text,
                'source': chunk.document_id,
                'chunk_type': chunk.section or 'text',
                'paper_id': chunk.paper_id,
                'page': chunk.page,
                'metadata': chunk.chunk_metadata
            }
        return None


class EnhancedVectorStore(FAISSVectorStore):
    """Enhanced vector store with multi-modal content support."""
    
    def __init__(self, store_path: str, embedding_dim: int = 384, config=None):
        """Initialize enhanced vector store."""
        super().__init__(store_path, embedding_dim, config)
        
        # Multi-modal specific storage
        self.multimodal_embeddings = {}  # chunk_id -> MultiModalEmbedding
        self.content_type_index = {}     # content_type -> list of chunk_ids
        self.equation_index = {}         # equation signatures -> chunk_ids
        self.figure_index = {}          # figure characteristics -> chunk_ids
        self.table_index = {}           # table characteristics -> chunk_ids
        
        # Enhanced metadata
        self.enhanced_metadata_path = Path(store_path) / "enhanced_metadata.pkl"
        self.content_index_path = Path(store_path) / "content_index.pkl"
        
        # Load existing enhanced data
        self._load_enhanced_data()
    
    def add_enhanced_chunks(self, chunks: List[EnhancedTextChunk], embeddings: List[MultiModalEmbedding]) -> None:
        """Add enhanced chunks with multi-modal embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Prepare data for base vector store
        texts = [chunk.text for chunk in chunks]
        combined_embeddings = [emb.combined_embedding for emb in embeddings]
        
        # Convert None embeddings to text embeddings
        for i, emb in enumerate(combined_embeddings):
            if emb is None:
                combined_embeddings[i] = embeddings[i].text_embedding
        
        combined_embeddings = np.array(combined_embeddings)
        
        # Add to base vector store
        chunk_ids = self.add_texts(texts, combined_embeddings)
        
        # Store enhanced information
        for chunk, embedding, chunk_id in zip(chunks, embeddings, chunk_ids):
            # Store multi-modal embedding
            self.multimodal_embeddings[chunk_id] = embedding
            
            # Update content type index
            content_type = getattr(chunk, 'content_type', 'text')
            if content_type not in self.content_type_index:
                self.content_type_index[content_type] = []
            self.content_type_index[content_type].append(chunk_id)
            
            # Index equations
            if chunk.equations:
                self._index_equations(chunk.equations, chunk_id)
            
            # Index figures
            if chunk.figures:
                self._index_figures(chunk.figures, chunk_id)
            
            # Index tables
            if chunk.tables:
                self._index_tables(chunk.tables, chunk_id)
        
        # Save enhanced data
        self._save_enhanced_data()
    
    def search_multimodal(
        self,
        query: str,
        embedder: MultiModalEmbedder,
        content_types: List[str] = None,
        content_preference: Dict[str, float] = None,
        top_k: int = 10
    ) -> List[Tuple[EnhancedTextChunk, float]]:
        """Search with multi-modal awareness and content type filtering."""
        
        # Create query embedding
        query_embedding = embedder.embed_text(query)
        
        # Get candidate chunks
        if content_types:
            candidate_ids = self._get_candidates_by_content_type(content_types)
        else:
            candidate_ids = list(self.multimodal_embeddings.keys())
        
        if not candidate_ids:
            return []
        
        # Calculate similarities with multi-modal awareness
        similarities = []
        for chunk_id in candidate_ids:
            if chunk_id in self.multimodal_embeddings:
                multimodal_emb = self.multimodal_embeddings[chunk_id]
                similarity = embedder.similarity_multimodal(
                    query_embedding, 
                    multimodal_emb, 
                    content_preference
                )
                similarities.append((chunk_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top results
        results = []
        for chunk_id, similarity in similarities[:top_k]:
            chunk = self._reconstruct_enhanced_chunk(chunk_id)
            if chunk:
                results.append((chunk, similarity))
        
        return results
    
    def search_equations(
        self,
        query: str,
        embedder: MultiModalEmbedder,
        equation_type: str = None,
        top_k: int = 10
    ) -> List[Tuple[EnhancedTextChunk, float]]:
        """Search specifically for equations."""
        content_preference = {'equation': 2.0, 'text': 1.0, 'figure': 0.5, 'table': 0.5}
        
        # Filter by equation-containing chunks
        equation_chunks = self.content_type_index.get('equation', [])
        mixed_chunks = self.content_type_index.get('mixed', [])
        candidate_ids = equation_chunks + mixed_chunks
        
        if not candidate_ids:
            # Fallback to all chunks with equations
            candidate_ids = [cid for cid in self.multimodal_embeddings.keys() 
                           if self.multimodal_embeddings[cid].equation_embedding is not None]
        
        # Enhanced query for equations
        enhanced_query = f"Mathematical equation formula: {query}"
        query_embedding = embedder.embed_text(enhanced_query)
        
        # Calculate similarities
        similarities = []
        for chunk_id in candidate_ids:
            if chunk_id in self.multimodal_embeddings:
                multimodal_emb = self.multimodal_embeddings[chunk_id]
                similarity = embedder.similarity_multimodal(
                    query_embedding, 
                    multimodal_emb, 
                    content_preference
                )
                similarities.append((chunk_id, similarity))
        
        # Sort and return results
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk_id, similarity in similarities[:top_k]:
            chunk = self._reconstruct_enhanced_chunk(chunk_id)
            if chunk:
                results.append((chunk, similarity))
        
        return results
    
    def search_figures(
        self,
        query: str,
        embedder: MultiModalEmbedder,
        figure_type: str = None,
        top_k: int = 10
    ) -> List[Tuple[EnhancedTextChunk, float]]:
        """Search specifically for figures."""
        content_preference = {'figure': 2.0, 'text': 1.0, 'equation': 0.5, 'table': 0.5}
        
        # Filter by figure-containing chunks
        figure_chunks = self.content_type_index.get('figure', [])
        mixed_chunks = self.content_type_index.get('mixed', [])
        candidate_ids = figure_chunks + mixed_chunks
        
        if not candidate_ids:
            # Fallback to all chunks with figures
            candidate_ids = [cid for cid in self.multimodal_embeddings.keys() 
                           if self.multimodal_embeddings[cid].figure_embedding is not None]
        
        # Enhanced query for figures
        enhanced_query = f"Figure image chart diagram: {query}"
        query_embedding = embedder.embed_text(enhanced_query)
        
        # Calculate similarities
        similarities = []
        for chunk_id in candidate_ids:
            if chunk_id in self.multimodal_embeddings:
                multimodal_emb = self.multimodal_embeddings[chunk_id]
                similarity = embedder.similarity_multimodal(
                    query_embedding, 
                    multimodal_emb, 
                    content_preference
                )
                similarities.append((chunk_id, similarity))
        
        # Sort and return results
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk_id, similarity in similarities[:top_k]:
            chunk = self._reconstruct_enhanced_chunk(chunk_id)
            if chunk:
                results.append((chunk, similarity))
        
        return results
    
    def search_tables(
        self,
        query: str,
        embedder: MultiModalEmbedder,
        table_type: str = None,
        top_k: int = 10
    ) -> List[Tuple[EnhancedTextChunk, float]]:
        """Search specifically for tables."""
        content_preference = {'table': 2.0, 'text': 1.0, 'equation': 0.5, 'figure': 0.5}
        
        # Filter by table-containing chunks
        table_chunks = self.content_type_index.get('table', [])
        mixed_chunks = self.content_type_index.get('mixed', [])
        candidate_ids = table_chunks + mixed_chunks
        
        if not candidate_ids:
            # Fallback to all chunks with tables
            candidate_ids = [cid for cid in self.multimodal_embeddings.keys() 
                           if self.multimodal_embeddings[cid].table_embedding is not None]
        
        # Enhanced query for tables
        enhanced_query = f"Table data results comparison: {query}"
        query_embedding = embedder.embed_text(enhanced_query)
        
        # Calculate similarities
        similarities = []
        for chunk_id in candidate_ids:
            if chunk_id in self.multimodal_embeddings:
                multimodal_emb = self.multimodal_embeddings[chunk_id]
                similarity = embedder.similarity_multimodal(
                    query_embedding, 
                    multimodal_emb, 
                    content_preference
                )
                similarities.append((chunk_id, similarity))
        
        # Sort and return results
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk_id, similarity in similarities[:top_k]:
            chunk = self._reconstruct_enhanced_chunk(chunk_id)
            if chunk:
                results.append((chunk, similarity))
        
        return results
    
    def get_content_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored content."""
        stats = {
            'total_chunks': len(self.metadata),
            'content_type_distribution': {},
            'equation_count': len(self.equation_index),
            'figure_count': len(self.figure_index),
            'table_count': len(self.table_index),
            'multimodal_chunks': len(self.multimodal_embeddings)
        }
        
        # Content type distribution
        for content_type, chunk_ids in self.content_type_index.items():
            stats['content_type_distribution'][content_type] = len(chunk_ids)
        
        return stats
    
    def _get_candidates_by_content_type(self, content_types: List[str]) -> List[str]:
        """Get candidate chunk IDs by content type."""
        candidates = set()
        
        for content_type in content_types:
            if content_type in self.content_type_index:
                candidates.update(self.content_type_index[content_type])
        
        return list(candidates)
    
    def _index_equations(self, equations: List, chunk_id: str) -> None:
        """Index equations for specialized search."""
        for eq in equations:
            # Create signatures for equations
            signatures = []
            
            if hasattr(eq, 'latex') and eq.latex:
                # Normalize LaTeX for indexing
                normalized_latex = self._normalize_equation(eq.latex)
                signatures.append(normalized_latex)
            
            if hasattr(eq, 'equation_type') and eq.equation_type:
                signatures.append(eq.equation_type)
            
            # Index by signatures
            for sig in signatures:
                if sig not in self.equation_index:
                    self.equation_index[sig] = []
                if chunk_id not in self.equation_index[sig]:
                    self.equation_index[sig].append(chunk_id)
    
    def _index_figures(self, figures: List, chunk_id: str) -> None:
        """Index figures for specialized search."""
        for fig in figures:
            # Index by figure characteristics
            characteristics = []
            
            if hasattr(fig, 'figure_type') and fig.figure_type:
                characteristics.append(fig.figure_type)
            
            if hasattr(fig, 'contains_equations') and fig.contains_equations:
                characteristics.append('contains_equations')
            
            # Index by characteristics
            for char in characteristics:
                if char not in self.figure_index:
                    self.figure_index[char] = []
                if chunk_id not in self.figure_index[char]:
                    self.figure_index[char].append(chunk_id)
    
    def _index_tables(self, tables: List, chunk_id: str) -> None:
        """Index tables for specialized search."""
        for table in tables:
            # Index by table characteristics
            characteristics = []
            
            if hasattr(table, 'table_type') and table.table_type:
                characteristics.append(table.table_type)
            
            # Index by size categories
            if hasattr(table, 'num_rows') and hasattr(table, 'num_cols'):
                if table.num_rows > 10:
                    characteristics.append('large_table')
                elif table.num_rows > 5:
                    characteristics.append('medium_table')
                else:
                    characteristics.append('small_table')
            
            # Index by characteristics
            for char in characteristics:
                if char not in self.table_index:
                    self.table_index[char] = []
                if chunk_id not in self.table_index[char]:
                    self.table_index[char].append(chunk_id)
    
    def _normalize_equation(self, latex: str) -> str:
        """Normalize LaTeX equation for indexing."""
        # Remove whitespace and common variations
        normalized = latex.strip().replace(' ', '').replace('\\,', '').replace('\\!', '')
        
        # Normalize common patterns
        normalized = normalized.replace('\\left', '').replace('\\right', '')
        
        return normalized.lower()
    
    def _reconstruct_enhanced_chunk(self, chunk_id: str) -> Optional[EnhancedTextChunk]:
        """Reconstruct an enhanced chunk from stored data."""
        if chunk_id not in self.metadata:
            return None
        
        # Get base metadata
        base_metadata = self.metadata[chunk_id]
        
        # Reconstruct enhanced chunk
        # Note: This is a simplified reconstruction
        # In practice, you'd need to store and restore all enhanced properties
        chunk = EnhancedTextChunk(
            text=base_metadata.get('text', ''),
            chunk_id=chunk_id,
            document_id=base_metadata.get('document_id', ''),
            section=base_metadata.get('section', ''),
            page=base_metadata.get('page'),
            chunk_index=base_metadata.get('chunk_index', 0),
            start_char=base_metadata.get('start_char', 0),
            end_char=base_metadata.get('end_char', 0),
            chunk_metadata=base_metadata.get('chunk_metadata', {})
        )
        
        return chunk
    
    def _save_enhanced_data(self) -> None:
        """Save enhanced vector store data."""
        try:
            # Save multi-modal embeddings (excluding numpy arrays for now)
            embeddings_data = {}
            for chunk_id, embedding in self.multimodal_embeddings.items():
                embeddings_data[chunk_id] = {
                    'content_weights': embedding.content_weights,
                    'has_equation': embedding.equation_embedding is not None,
                    'has_figure': embedding.figure_embedding is not None,
                    'has_table': embedding.table_embedding is not None
                }
            
            with open(self.enhanced_metadata_path, 'wb') as f:
                pickle.dump({
                    'embeddings_metadata': embeddings_data,
                    'content_type_index': self.content_type_index,
                    'equation_index': self.equation_index,
                    'figure_index': self.figure_index,
                    'table_index': self.table_index
                }, f)
            
            self.logger.info("Enhanced vector store data saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save enhanced data: {e}")
    
    def _load_enhanced_data(self) -> None:
        """Load enhanced vector store data."""
        try:
            if self.enhanced_metadata_path.exists():
                with open(self.enhanced_metadata_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Restore indices
                self.content_type_index = data.get('content_type_index', {})
                self.equation_index = data.get('equation_index', {})
                self.figure_index = data.get('figure_index', {})
                self.table_index = data.get('table_index', {})
                
                self.logger.info("Enhanced vector store data loaded")
                
        except Exception as e:
            self.logger.warning(f"Could not load enhanced data: {e}")
    
    def clear_enhanced_data(self) -> None:
        """Clear all enhanced data."""
        self.multimodal_embeddings.clear()
        self.content_type_index.clear()
        self.equation_index.clear()
        self.figure_index.clear()
        self.table_index.clear()
        
        # Remove files
        if self.enhanced_metadata_path.exists():
            self.enhanced_metadata_path.unlink()
        
        self.logger.info("Enhanced vector store data cleared")
    
    def export_enhanced_data(self, output_path: str) -> None:
        """Export enhanced data for analysis."""
        stats = self.get_content_statistics()
        
        export_data = {
            'statistics': stats,
            'content_type_index': self.content_type_index,
            'equation_signatures': list(self.equation_index.keys()),
            'figure_characteristics': list(self.figure_index.keys()),
            'table_characteristics': list(self.table_index.keys())
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Enhanced data exported to {output_path}") 