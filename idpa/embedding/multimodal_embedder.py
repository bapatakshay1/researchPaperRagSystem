"""
Multi-modal embedder for text, equations, figures, and tables.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import base64
import pickle
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from utils.logger import get_logger
from .enhanced_chunker import EnhancedTextChunk, TextChunk


class Embedder:
    """Generates embeddings for document chunks using sentence transformers."""
    
    def __init__(self, config=None):
        """Initialize document embedder with configuration."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Configuration parameters
        self.model_name = getattr(config, 'embedding_model', 'all-MiniLM-L6-v2') if config else 'all-MiniLM-L6-v2'
        self.device = getattr(config, 'embedding_device', 'cpu') if config else 'cpu'
        self.batch_size = getattr(config, 'batch_size', 32) if config else 32
        
        # Initialize the model
        self.model = None
        self.embedding_dim = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            
            # Check if CUDA is available and requested
            if self.device == 'cuda' and not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = 'cpu'
            
            # Load the model
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_chunks(self, chunks: List[TextChunk]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            Dictionary mapping chunk_id to embedding vector
        """
        if not chunks:
            return {}
        
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract texts and chunk IDs
        texts = [chunk.text for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = self._generate_embeddings_batch(texts)
        
        # Create mapping from chunk_id to embedding
        embedding_dict = {}
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            embedding_dict[chunk_id] = embedding
        
        self.logger.info(f"Generated embeddings for {len(embedding_dict)} chunks")
        return embedding_dict
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text.strip():
            return np.zeros(self.embedding_dim)
        
        embedding = self.model.encode([text], batch_size=1)[0]
        return embedding
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query (same as embed_text but with explicit naming).
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.embed_text(query)
    
    def _generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts."""
        try:
            # Use the model's encode method with batching
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_tensor=False,
                normalize_embeddings=True  # Normalize for better similarity computation
            )
            
            # Ensure embeddings are numpy arrays
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            
            return [emb for emb in embeddings]
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def save_embeddings(
        self, 
        embeddings: Dict[str, np.ndarray], 
        chunks: List[TextChunk],
        output_path: str
    ) -> None:
        """
        Save embeddings and associated metadata to disk.
        
        Args:
            embeddings: Dictionary mapping chunk_id to embedding
            chunks: List of TextChunk objects
            output_path: Path to save the embeddings
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        embedding_data = {
            'embeddings': embeddings,
            'chunks_metadata': [asdict(chunk) for chunk in chunks],
            'model_info': {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'device': self.device
            }
        }
        
        # Save as pickle for efficiency with numpy arrays
        pickle_path = output_path.with_suffix('.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        # Also save metadata as JSON for human readability
        json_path = output_path.with_suffix('.json')
        json_data = {
            'chunks_metadata': embedding_data['chunks_metadata'],
            'model_info': embedding_data['model_info'],
            'embedding_shape': f"({len(embeddings)}, {self.embedding_dim})"
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Embeddings saved to {pickle_path} and metadata to {json_path}")
    
    def load_embeddings(self, input_path: str) -> tuple[Dict[str, np.ndarray], List[TextChunk]]:
        """
        Load embeddings and chunks from disk.
        
        Args:
            input_path: Path to the saved embeddings
            
        Returns:
            Tuple of (embeddings_dict, chunks_list)
        """
        input_path = Path(input_path)
        pickle_path = input_path.with_suffix('.pkl')
        
        if not pickle_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {pickle_path}")
        
        with open(pickle_path, 'rb') as f:
            embedding_data = pickle.load(f)
        
        # Reconstruct TextChunk objects
        chunks = []
        for chunk_dict in embedding_data['chunks_metadata']:
            chunk = TextChunk(**chunk_dict)
            chunks.append(chunk)
        
        embeddings = embedding_data['embeddings']
        
        self.logger.info(f"Loaded {len(embeddings)} embeddings from {pickle_path}")
        return embeddings, chunks
    
    def compute_similarity(
        self, 
        query_embedding: np.ndarray, 
        document_embeddings: Dict[str, np.ndarray],
        top_k: int = 10
    ) -> List[tuple[str, float]]:
        """
        Compute similarity between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: Dictionary of document embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_id, similarity_score) tuples, sorted by similarity
        """
        if not document_embeddings:
            return []
        
        similarities = []
        
        # Normalize query embedding
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        for chunk_id, doc_embedding in document_embeddings.items():
            # Compute cosine similarity
            doc_norm = doc_embedding / np.linalg.norm(doc_embedding)
            similarity = np.dot(query_norm, doc_norm)
            similarities.append((chunk_id, float(similarity)))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def batch_similarity_search(
        self,
        queries: List[str],
        document_embeddings: Dict[str, np.ndarray],
        top_k: int = 10
    ) -> List[List[tuple[str, float]]]:
        """
        Perform similarity search for multiple queries.
        
        Args:
            queries: List of query strings
            document_embeddings: Dictionary of document embeddings
            top_k: Number of top results per query
            
        Returns:
            List of similarity results for each query
        """
        # Generate embeddings for all queries
        query_embeddings = self._generate_embeddings_batch(queries)
        
        results = []
        for query_embedding in query_embeddings:
            similarities = self.compute_similarity(
                query_embedding, document_embeddings, top_k
            )
            results.append(similarities)
        
        return results
    
    def get_embedding_statistics(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compute statistics about the embeddings.
        
        Args:
            embeddings: Dictionary of embeddings
            
        Returns:
            Dictionary with embedding statistics
        """
        if not embeddings:
            return {}
        
        embedding_matrix = np.array(list(embeddings.values()))
        
        stats = {
            'count': len(embeddings),
            'dimension': embedding_matrix.shape[1],
            'mean_norm': float(np.mean(np.linalg.norm(embedding_matrix, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embedding_matrix, axis=1))),
            'mean_values': embedding_matrix.mean(axis=0).tolist(),
            'std_values': embedding_matrix.std(axis=0).tolist()
        }
        
        return stats
    
    def update_model(self, new_model_name: str) -> None:
        """
        Update the embedding model.
        
        Args:
            new_model_name: Name of the new model to load
        """
        self.logger.info(f"Updating model from {self.model_name} to {new_model_name}")
        self.model_name = new_model_name
        self._load_model()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': self.device,
            'batch_size': self.batch_size
        }


@dataclass
class MultiModalEmbedding:
    """Container for multi-modal embeddings."""
    text_embedding: np.ndarray
    equation_embedding: Optional[np.ndarray] = None
    figure_embedding: Optional[np.ndarray] = None
    table_embedding: Optional[np.ndarray] = None
    combined_embedding: Optional[np.ndarray] = None
    content_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.content_weights is None:
            self.content_weights = {'text': 1.0}


class MultiModalEmbedder(Embedder):
    """Enhanced embedder for multi-modal academic content."""
    
    def __init__(self, config=None):
        """Initialize multi-modal embedder."""
        super().__init__(config)
        
        # Initialize specialized models
        self._initialize_multimodal_models()
        
        # Embedding weights for different content types
        self.content_weights = {
            'text': 1.0,
            'equation': 0.8,
            'figure': 0.6,
            'table': 0.7
        }
        
        # Combination strategy
        self.combination_strategy = getattr(config, 'embedding_combination_strategy', 'weighted_average') if config else 'weighted_average'
    
    def _initialize_multimodal_models(self):
        """Initialize models for different content types."""
        try:
            # Text embeddings (inherited from parent)
            # self.model is already initialized in parent class
            
            # Equation embeddings - use specialized model or text model
            self.equation_model = self.model  # For now, use same model
            
            # Figure embeddings - would need vision model
            self.figure_model = None  # Placeholder for vision model
            
            # Table embeddings - use text model with special processing
            self.table_model = self.model
            
            self.logger.info("Multi-modal models initialized")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize all multi-modal models: {e}")
    
    def embed_enhanced_chunk(self, chunk: EnhancedTextChunk) -> MultiModalEmbedding:
        """Create multi-modal embedding for an enhanced text chunk."""
        embeddings = {}
        
        # Text embedding (always present)
        text_embedding = self.embed_text(chunk.text)
        embeddings['text'] = text_embedding
        
        # Equation embeddings
        equation_embedding = None
        if chunk.equations:
            equation_embedding = self._embed_equations(chunk.equations)
            embeddings['equation'] = equation_embedding
        
        # Figure embeddings
        figure_embedding = None
        if chunk.figures:
            figure_embedding = self._embed_figures(chunk.figures)
            embeddings['figure'] = figure_embedding
        
        # Table embeddings
        table_embedding = None
        if chunk.tables:
            table_embedding = self._embed_tables(chunk.tables)
            embeddings['table'] = table_embedding
        
        # Combine embeddings
        combined_embedding = self._combine_embeddings(embeddings, chunk)
        
        # Calculate content weights based on chunk
        content_weights = self._calculate_content_weights(chunk)
        
        return MultiModalEmbedding(
            text_embedding=text_embedding,
            equation_embedding=equation_embedding,
            figure_embedding=figure_embedding,
            table_embedding=table_embedding,
            combined_embedding=combined_embedding,
            content_weights=content_weights
        )
    
    def _embed_equations(self, equations: List) -> np.ndarray:
        """Generate embeddings for equations."""
        if not equations:
            return np.zeros(self.embedding_dim)
        
        # Extract equation texts
        equation_texts = []
        for eq in equations:
            if hasattr(eq, 'text_representation') and eq.text_representation:
                equation_texts.append(eq.text_representation)
            elif hasattr(eq, 'latex') and eq.latex:
                equation_texts.append(eq.latex)
            elif hasattr(eq, 'text') and eq.text:
                equation_texts.append(eq.text)
        
        if not equation_texts:
            return np.zeros(self.embedding_dim)
        
        # Generate embeddings using parent class method
        embeddings = self._generate_embeddings_batch(equation_texts)
        
        # Return average embedding
        return np.mean(embeddings, axis=0)
    
    def _embed_figures(self, figures: List) -> np.ndarray:
        """Generate embeddings for figures."""
        if not figures:
            return np.zeros(self.embedding_dim)
        
        # Extract figure texts
        figure_texts = []
        for fig in figures:
            if hasattr(fig, 'caption') and fig.caption:
                figure_texts.append(fig.caption)
            if hasattr(fig, 'description') and fig.description:
                figure_texts.append(fig.description)
            if hasattr(fig, 'extracted_text') and fig.extracted_text:
                figure_texts.append(fig.extracted_text)
        
        if not figure_texts:
            return np.zeros(self.embedding_dim)
        
        # Generate embeddings using parent class method
        embeddings = self._generate_embeddings_batch(figure_texts)
        
        # Return average embedding
        return np.mean(embeddings, axis=0)
    
    def _embed_tables(self, tables: List) -> np.ndarray:
        """Generate embeddings for tables."""
        if not tables:
            return np.zeros(self.embedding_dim)
        
        # Extract table texts
        table_texts = []
        for table in tables:
            if hasattr(table, 'caption') and table.caption:
                table_texts.append(table.caption)
            if hasattr(table, 'data') and table.data:
                # Convert table data to text
                table_text = self._table_to_text(table)
                if table_text:
                    table_texts.append(table_text)
        
        if not table_texts:
            return np.zeros(self.embedding_dim)
        
        # Generate embeddings using parent class method
        embeddings = self._generate_embeddings_batch(table_texts)
        
        # Return average embedding
        return np.mean(embeddings, axis=0)
    
    def _table_to_text(self, table) -> str:
        """Convert table data to text representation."""
        if not table or not hasattr(table, 'data') or not table.data:
            return ""
        
        table_data = table.data
        if not table_data:
            return ""
        
        # Format table as text
        text_lines = []
        
        # Add caption if available
        if hasattr(table, 'caption') and table.caption:
            text_lines.append(f"Table: {table.caption}")
        
        # Add headers if available
        if hasattr(table, 'headers') and table.headers:
            headers = " | ".join(str(cell) for cell in table.headers if cell)
            text_lines.append(f"Headers: {headers}")
        
        # Add table rows
        for row_idx, row in enumerate(table_data):
            if row and any(cell for cell in row):  # Skip empty rows
                row_text = " | ".join(str(cell) for cell in row if cell)
                text_lines.append(f"Row {row_idx + 1}: {row_text}")
        
        return "\n".join(text_lines)
    
    def _combine_embeddings(self, embeddings: Dict[str, np.ndarray], chunk: EnhancedTextChunk) -> np.ndarray:
        """Combine different types of embeddings into a single representation."""
        if self.combination_strategy == 'weighted_average':
            return self._weighted_average_combination(embeddings, chunk)
        elif self.combination_strategy == 'concatenation':
            return self._concatenation_combination(embeddings)
        elif self.combination_strategy == 'attention':
            return self._attention_combination(embeddings, chunk)
        else:
            # Default to weighted average
            return self._weighted_average_combination(embeddings, chunk)
    
    def _weighted_average_combination(self, embeddings: Dict[str, np.ndarray], chunk: EnhancedTextChunk) -> np.ndarray:
        """Combine embeddings using weighted average."""
        combined = None
        total_weight = 0
        
        for content_type, embedding in embeddings.items():
            if embedding is not None:
                weight = self.content_weights.get(content_type, 1.0)
                
                # Adjust weight based on content importance in chunk
                if content_type == 'equation' and chunk.mathematical_content:
                    weight *= 1.5
                elif content_type == 'figure' and chunk.visual_content:
                    weight *= 1.3
                elif content_type == 'table' and chunk.tabular_content:
                    weight *= 1.4
                
                if combined is None:
                    combined = embedding * weight
                else:
                    combined += embedding * weight
                
                total_weight += weight
        
        if combined is not None and total_weight > 0:
            combined /= total_weight
        
        return combined
    
    def _concatenation_combination(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine embeddings by concatenation."""
        valid_embeddings = [emb for emb in embeddings.values() if emb is not None]
        
        if not valid_embeddings:
            return None
        
        # Ensure all embeddings have the same dimension
        target_dim = valid_embeddings[0].shape[0]
        normalized_embeddings = []
        
        for emb in valid_embeddings:
            if emb.shape[0] != target_dim:
                # Pad or truncate to match target dimension
                if emb.shape[0] < target_dim:
                    padded = np.zeros(target_dim)
                    padded[:emb.shape[0]] = emb
                    normalized_embeddings.append(padded)
                else:
                    normalized_embeddings.append(emb[:target_dim])
            else:
                normalized_embeddings.append(emb)
        
        return np.concatenate(normalized_embeddings)
    
    def _attention_combination(self, embeddings: Dict[str, np.ndarray], chunk: EnhancedTextChunk) -> np.ndarray:
        """Combine embeddings using attention mechanism (simplified)."""
        # Simplified attention - in practice would use learned attention weights
        content_importance = {
            'text': 1.0,
            'equation': chunk.chunk_metadata.get('mathematical_score', 0.0) if hasattr(chunk, 'chunk_metadata') else 0.0,
            'figure': chunk.chunk_metadata.get('visual_score', 0.0) if hasattr(chunk, 'chunk_metadata') else 0.0,
            'table': chunk.chunk_metadata.get('tabular_score', 0.0) if hasattr(chunk, 'chunk_metadata') else 0.0
        }
        
        # Normalize importance scores
        total_importance = sum(content_importance.values())
        if total_importance > 0:
            content_importance = {k: v/total_importance for k, v in content_importance.items()}
        
        # Apply attention weights
        combined = None
        for content_type, embedding in embeddings.items():
            if embedding is not None:
                weight = content_importance.get(content_type, 0.0)
                if combined is None:
                    combined = embedding * weight
                else:
                    combined += embedding * weight
        
        return combined
    
    def _calculate_content_weights(self, chunk: EnhancedTextChunk) -> Dict[str, float]:
        """Calculate dynamic content weights based on chunk characteristics."""
        weights = {'text': 1.0}
        
        if chunk.mathematical_content:
            weights['equation'] = min(len(chunk.equations) * 0.3 + 0.5, 1.0)
        
        if chunk.visual_content:
            weights['figure'] = min(len(chunk.figures) * 0.4 + 0.4, 1.0)
        
        if chunk.tabular_content:
            weights['table'] = min(len(chunk.tables) * 0.3 + 0.5, 1.0)
        
        return weights
    
    def embed_enhanced_chunks(self, chunks: List[EnhancedTextChunk]) -> List[MultiModalEmbedding]:
        """Create multi-modal embeddings for a list of enhanced chunks."""
        embeddings = []
        
        for chunk in chunks:
            embedding = self.embed_enhanced_chunk(chunk)
            embeddings.append(embedding)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the combined embeddings."""
        if self.combination_strategy == 'concatenation':
            # Would be sum of all embedding dimensions
            return self.model.get_sentence_embedding_dimension() * 4  # text + equation + figure + table
        else:
            # For other strategies, same as text embedding dimension
            return self.model.get_sentence_embedding_dimension()
    
    def embed_query_multimodal(self, query: str, query_type: str = "text") -> np.ndarray:
        """Create embedding for a query with multi-modal awareness."""
        if query_type == "equation":
            # Enhance query for equation search
            enhanced_query = f"Mathematical equation: {query}"
        elif query_type == "figure":
            # Enhance query for figure search
            enhanced_query = f"Figure or image: {query}"
        elif query_type == "table":
            # Enhance query for table search
            enhanced_query = f"Table or data: {query}"
        else:
            enhanced_query = query
        
        return self.embed_text(enhanced_query)
    
    def similarity_multimodal(
        self, 
        query_embedding: np.ndarray, 
        chunk_embedding: MultiModalEmbedding,
        content_preference: Dict[str, float] = None
    ) -> float:
        """Calculate similarity with multi-modal awareness."""
        if content_preference is None:
            content_preference = {'text': 1.0, 'equation': 1.0, 'figure': 1.0, 'table': 1.0}
        
        # Use combined embedding for similarity
        if chunk_embedding.combined_embedding is not None:
            similarity = self._cosine_similarity(query_embedding, chunk_embedding.combined_embedding)
        else:
            similarity = self._cosine_similarity(query_embedding, chunk_embedding.text_embedding)
        
        # Apply content preference adjustments
        for content_type, preference in content_preference.items():
            if content_type in chunk_embedding.content_weights:
                weight = chunk_embedding.content_weights[content_type]
                similarity *= (1.0 + (preference - 1.0) * weight)
        
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if a is None or b is None:
            return 0.0
        
        # Handle dimension mismatch
        if a.shape[0] != b.shape[0]:
            min_dim = min(a.shape[0], b.shape[0])
            a = a[:min_dim]
            b = b[:min_dim]
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b) 