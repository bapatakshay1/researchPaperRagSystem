"""
Main query processor for handling different types of academic paper queries.
"""

import re
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache
from collections import defaultdict

# Import OpenAI with backward compatibility
try:
    from openai import OpenAI, AsyncOpenAI
    HAS_ASYNC_OPENAI = True
except ImportError:
    try:
        from openai import OpenAI
        AsyncOpenAI = None
        HAS_ASYNC_OPENAI = False
    except ImportError:
        OpenAI = None
        AsyncOpenAI = None
        HAS_ASYNC_OPENAI = False

from .query_types import QueryClassifier, QueryType, QueryAnalysis
from embedding.enhanced_chunker import TextChunk
from utils.logger import get_logger

@dataclass
class QueryResponse:
    """Response to a user query."""
    answer: str
    sources: List[TextChunk]
    confidence: float
    query_type: QueryType
    processing_time: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ResponseCache:
    """Simple in-memory cache for LLM responses."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _generate_key(self, prompt: str, query_type: str) -> str:
        """Generate cache key from prompt and query type."""
        content = f"{query_type}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt: str, query_type: str) -> Optional[str]:
        """Get cached response if available and not expired."""
        key = self._generate_key(prompt, query_type)
        
        if key in self.cache:
            timestamp = self.timestamps[key]
            if time.time() - timestamp < self.ttl_seconds:
                return self.cache[key]
            else:
                # Remove expired entry
                del self.cache[key]
                del self.timestamps[key]
        
        return None
    
    def put(self, prompt: str, query_type: str, response: str):
        """Cache a response."""
        key = self._generate_key(prompt, query_type)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = response
        self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.timestamps.clear()


class PerformanceMetrics:
    """Track performance metrics for monitoring and optimization."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_processing_time = 0.0
        self.slow_queries = 0
        self.query_types = defaultdict(int)
        self.search_times = []
        self.llm_times = []
        self.start_time = time.time()
    
    def record_query(self, query_type: QueryType, processing_time: float, 
                    cache_hit: bool, search_time: float, llm_time: float):
        """Record metrics for a query."""
        self.total_queries += 1
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        self.total_processing_time += processing_time
        if processing_time > 5.0:  # Consider > 5s as slow
            self.slow_queries += 1
        
        self.query_types[query_type.value] += 1
        self.search_times.append(search_time)
        self.llm_times.append(llm_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.total_queries == 0:
            return {}
        
        uptime = time.time() - self.start_time
        
        return {
            'total_queries': self.total_queries,
            'queries_per_second': self.total_queries / max(uptime, 1),
            'cache_hit_rate': self.cache_hits / self.total_queries,
            'avg_processing_time': self.total_processing_time / self.total_queries,
            'slow_query_rate': self.slow_queries / self.total_queries,
            'avg_search_time': sum(self.search_times) / len(self.search_times) if self.search_times else 0,
            'avg_llm_time': sum(self.llm_times) / len(self.llm_times) if self.llm_times else 0,
            'query_types': dict(self.query_types),
            'uptime_seconds': uptime
        }


class QueryProcessor:
    """Main processor for handling different types of queries."""
    
    def __init__(self, config, vector_store, embedder):
        """
        Initialize query processor.
        
        Args:
            config: Configuration object
            vector_store: FAISS vector store instance
            embedder: Document embedder instance
        """
        self.config = config
        self.vector_store = vector_store
        self.embedder = embedder
        self.logger = get_logger(__name__)
        
        # Initialize query classifier
        self.classifier = QueryClassifier()
        
        # LLM configuration
        self.llm_provider = getattr(config, 'llm_provider', 'openai')
        self.max_tokens = getattr(config, 'max_tokens', 2000)
        self.temperature = getattr(config, 'temperature', 0.1)
        self.query_timeout = getattr(config, 'query_timeout', 30)
        
        # Performance optimizations
        self.max_context_length = getattr(config, 'max_context_length', 8000)
        self.enable_caching = getattr(config, 'enable_caching', True)
        self.cache_ttl = getattr(config, 'cache_ttl_seconds', 3600)
        
        # Initialize caching and metrics
        self.response_cache = ResponseCache(ttl_seconds=self.cache_ttl) if self.enable_caching else None
        self.metrics = PerformanceMetrics()
        
        # Shared thread pool for any blocking operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='query_processor')
        
        # Initialize LLM client
        self._setup_llm()
        
        # Prompt templates for different query types
        self._setup_prompts()
        
        # Dynamic similarity thresholds
        self._setup_dynamic_thresholds()
    
    def _setup_llm(self):
        """Set up the LLM client based on configuration."""
        if self.llm_provider == 'openai':
            if not OpenAI:
                raise ValueError("OpenAI library is not installed. Please install with: pip install openai")
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key is required")
            
            self.model_name = getattr(self.config, 'openai_model', 'gpt-4')
            
            # Create sync client (always available)
            self.sync_client = OpenAI(api_key=self.config.openai_api_key)
            
            # Create async client if available
            if HAS_ASYNC_OPENAI and AsyncOpenAI:
                self.async_client = AsyncOpenAI(
                    api_key=self.config.openai_api_key,
                    max_retries=3,
                    timeout=self.query_timeout
                )
                self.logger.info("Using async OpenAI client for optimal performance")
            else:
                self.async_client = None
                self.logger.warning("AsyncOpenAI not available, falling back to sync client with threading")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _setup_dynamic_thresholds(self):
        """Set up dynamic similarity thresholds for different query types."""
        self.similarity_thresholds = {
            QueryType.DIRECT_LOOKUP: 0.4,      # High precision needed
            QueryType.COMPARISON: 0.25,        # Need diverse sources  
            QueryType.SUMMARIZATION: 0.2,      # Need broad coverage
            QueryType.METRIC_EXTRACTION: 0.35, # Need specific data
            QueryType.GENERAL_SEARCH: 0.3,     # Default
            QueryType.UNKNOWN: 0.3
        }
    
    def _get_similarity_threshold(self, query_type: QueryType) -> float:
        """Get appropriate similarity threshold for query type."""
        return self.similarity_thresholds.get(query_type, 0.3)

    def _setup_prompts(self):
        """Set up prompt templates for different query types."""
        self.prompts = {
            QueryType.DIRECT_LOOKUP: """
Based on the following context from academic papers, answer the user's question directly and concisely.

Context:
{context}

Question: {query}

Instructions:
- Provide a direct, factual answer based on the context
- Use all available information from the context, even if it's partial
- If specific information is missing, acknowledge what's available and what's not
- Cite specific papers or sections when possible
- Keep the answer focused and relevant
- Only say "information is not available" if the context contains NO relevant information at all

Answer:""",

            QueryType.SUMMARIZATION: """
Based on the following context from academic papers, provide a comprehensive summary addressing the user's request.

Context:
{context}

Request: {query}

Instructions:
- Create a well-structured summary covering the key points
- Organize information logically
- Highlight the most important findings or aspects
- Maintain academic tone and accuracy
- Include relevant details while being concise

Summary:""",

            QueryType.COMPARISON: """
Based on the following context from academic papers, provide a detailed comparison addressing the user's request.

Context:
{context}

Comparison Request: {query}

Instructions:
- Compare the specified aspects systematically
- Highlight similarities and differences
- Use clear structure (e.g., bullet points or sections)
- Be objective and factual
- Point out advantages and disadvantages where applicable
- If insufficient information for comparison, state this clearly

Comparison:""",

            QueryType.METRIC_EXTRACTION: """
Based on the following context from academic papers, extract and present the requested metrics or numerical results.

Context:
{context}

Metrics Request: {query}

Instructions:
- Identify and extract all relevant numerical metrics
- Present metrics in a clear, organized format
- Include context for each metric (which paper, what conditions, etc.)
- If metrics are not available, state this clearly
- Ensure accuracy in reporting numbers

Metrics:""",

            QueryType.GENERAL_SEARCH: """
Based on the following context from academic papers, provide a comprehensive response to the user's query.

Context:
{context}

Query: {query}

Instructions:
- Address the query comprehensively using the available context
- Organize information logically
- Provide relevant details and examples
- Maintain academic accuracy
- If the context doesn't fully address the query, mention this

Response:"""
        }
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _remove_duplicate_chunks(self, chunks: List[Tuple[TextChunk, float]], 
                                similarity_threshold: float = 0.8) -> List[Tuple[TextChunk, float]]:
        """Remove duplicate chunks based on text similarity."""
        filtered = []
        
        for chunk, score in chunks:
            is_duplicate = False
            for existing_chunk, _ in filtered:
                text_similarity = self._calculate_text_similarity(chunk.text, existing_chunk.text)
                if text_similarity > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append((chunk, score))
        
        return filtered
    
    def _calculate_content_quality(self, chunk: TextChunk, query: str) -> float:
        """Calculate content quality score for a chunk."""
        text_lower = chunk.text.lower()
        query_lower = query.lower()
        
        # 1. Query relevance (how many query terms appear)
        query_terms = set(query_lower.split())
        text_terms = set(text_lower.split())
        relevance_score = len(query_terms & text_terms) / max(len(query_terms), 1)
        
        # 2. Academic content indicators
        academic_indicators = [
            'algorithm', 'method', 'results', 'analysis', 'evaluation',
            'performance', 'comparison', 'experiment', 'theorem', 'proof',
            'approach', 'technique', 'implementation', 'study', 'research'
        ]
        academic_score = sum(1 for term in academic_indicators if term in text_lower)
        academic_score = min(academic_score / len(academic_indicators), 1.0)
        
        # 3. Avoid metadata
        metadata_indicators = ['email', 'contact', 'university', 'copyright', 'license', 'doi']
        metadata_penalty = sum(1 for term in metadata_indicators if term in text_lower)
        metadata_penalty = min(metadata_penalty * 0.2, 0.8)  # Max 80% penalty
        
        # 4. Length quality (prefer substantial content)
        length_score = min(len(chunk.text) / 200, 1.0)  # Optimal around 200 chars
        
        # 5. Section quality bonus
        section_bonus = 0.0
        if chunk.section:
            important_sections = ['abstract', 'introduction', 'results', 'conclusion', 'discussion']
            if any(sec in chunk.section.lower() for sec in important_sections):
                section_bonus = 0.1
        
        final_score = (
            0.4 * relevance_score + 
            0.25 * academic_score + 
            0.15 * length_score + 
            0.1 * section_bonus -
            0.1 * metadata_penalty
        )
        
        return max(final_score, 0.0)

    async def process_query(self, query: str, top_k: int = None) -> QueryResponse:
        """
        Process a user query and return a comprehensive response.
        
        Args:
            query: User query string
            top_k: Number of top chunks to retrieve
            
        Returns:
            QueryResponse object with answer and metadata
        """
        start_time = time.time()
        search_time = 0.0
        llm_time = 0.0
        cache_hit = False
        
        try:
            # Validate query
            is_valid, error_msg = self.classifier.validate_query(query)
            if not is_valid:
                return QueryResponse(
                    answer=f"Invalid query: {error_msg}",
                    sources=[],
                    confidence=0.0,
                    query_type=QueryType.UNKNOWN,
                    processing_time=time.time() - start_time
                )
            
            # Classify query
            analysis = self.classifier.classify(query)
            self.logger.info(f"Query classified as {analysis.query_type.value} with confidence {analysis.confidence:.2f}")
            
            # Retrieve relevant chunks with improved filtering
            search_start = time.time()
            top_k = top_k or 5  # Default to 5 results
            search_results = self._search_similar_chunks(query, analysis.query_type, top_k=top_k)
            search_time = time.time() - search_start
            
            # Convert to expected format
            relevant_chunks = []
            for result in search_results:
                # Create TextChunk object from search result
                chunk = TextChunk(
                    text=result.text,
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    section=result.section,
                    page=None,
                    chunk_metadata={'document_title': result.document_id}
                )
                # Get similarity from the search results
                relevant_chunks.append((chunk, 0.8))  # Default similarity score
            
            if not relevant_chunks:
                processing_time = time.time() - start_time
                self.metrics.record_query(analysis.query_type, processing_time, False, search_time, 0.0)
                return QueryResponse(
                    answer="No relevant information found in the document collection.",
                    sources=[],
                    confidence=0.0,
                    query_type=analysis.query_type,
                    processing_time=processing_time
                )
            
            # Extract chunks and scores
            chunks = [chunk for chunk, score in relevant_chunks]
            scores = [score for chunk, score in relevant_chunks]
            
            # Generate response based on query type
            llm_start = time.time()
            answer, cache_hit = await self._generate_response_with_cache(query, analysis, chunks)
            llm_time = time.time() - llm_start
            
            # Calculate overall confidence
            confidence = self._calculate_response_confidence(analysis.confidence, scores)
            
            processing_time = time.time() - start_time
            
            # Record metrics
            self.metrics.record_query(analysis.query_type, processing_time, cache_hit, search_time, llm_time)
            
            return QueryResponse(
                answer=answer,
                sources=chunks,
                confidence=confidence,
                query_type=analysis.query_type,
                processing_time=processing_time,
                metadata={
                    'analysis': asdict(analysis),
                    'retrieval_scores': scores,
                    'num_sources': len(chunks),
                    'cache_hit': cache_hit,
                    'search_time': search_time,
                    'llm_time': llm_time
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing query: {e}")
            self.metrics.record_query(QueryType.UNKNOWN, processing_time, False, search_time, llm_time)
            return QueryResponse(
                answer=f"An error occurred while processing your query: {str(e)}",
                sources=[],
                confidence=0.0,
                query_type=QueryType.UNKNOWN,
                processing_time=processing_time
            )
    
    async def _generate_response_with_cache(
        self, 
        query: str, 
        analysis: QueryAnalysis, 
        chunks: List[TextChunk]
    ) -> Tuple[str, bool]:
        """Generate response with caching support."""
        
        # Prepare context from chunks
        context = self._prepare_context(chunks, analysis.query_type)
        
        # Get appropriate prompt template
        prompt_template = self.prompts.get(analysis.query_type, self.prompts[QueryType.GENERAL_SEARCH])
        
        # Format prompt
        prompt = prompt_template.format(context=context, query=query)
        
        # Check cache first
        cache_hit = False
        if self.response_cache:
            cached_response = self.response_cache.get(prompt, analysis.query_type.value)
            if cached_response:
                cache_hit = True
                return cached_response, cache_hit
        
        # Generate response using LLM
        if self.llm_provider == 'openai':
            response = await self._call_openai_async(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        
        # Cache the response
        if self.response_cache:
            self.response_cache.put(prompt, analysis.query_type.value, response)
        
        return response, cache_hit

    async def _generate_response(
        self, 
        query: str, 
        analysis: QueryAnalysis, 
        chunks: List[TextChunk]
    ) -> str:
        """Generate response using LLM based on query type and context."""
        response, _ = await self._generate_response_with_cache(query, analysis, chunks)
        return response
    
    def _prepare_context(self, chunks: List[TextChunk], query_type: QueryType) -> str:
        """Prepare context string from retrieved chunks with length management."""
        context_parts = []
        total_length = 0
        
        for i, chunk in enumerate(chunks):
            # Create concise header with metadata
            header = f"[Source {i+1}]"
            if chunk.chunk_metadata.get('document_title'):
                header += f" {chunk.chunk_metadata['document_title']}"
            if chunk.section:
                header += f" - {chunk.section}"
            if chunk.page:
                header += f" (Page {chunk.page})"
            
            # Add chunk content
            chunk_content = f"{header}\n{chunk.text}"
            
            # Check if adding this chunk would exceed context limit
            if total_length + len(chunk_content) > self.max_context_length:
                remaining_space = self.max_context_length - total_length
                if remaining_space > 100:  # Only add if meaningful space left
                    truncated_content = chunk_content[:remaining_space-10] + "..."
                    context_parts.append(truncated_content)
                break
            
            context_parts.append(chunk_content)
            total_length += len(chunk_content)
        
        # For comparison queries, try to group by papers
        if query_type == QueryType.COMPARISON and len(context_parts) > 2:
            return self._organize_context_for_comparison(chunks)
        
        return "\n\n".join(context_parts)
    
    def _organize_context_for_comparison(self, chunks: List[TextChunk]) -> str:
        """Organize context specifically for comparison queries."""
        # Group chunks by paper
        papers = {}
        for chunk in chunks:
            paper_id = chunk.paper_id
            if paper_id not in papers:
                papers[paper_id] = []
            papers[paper_id].append(chunk)
        
        context_parts = []
        total_length = 0
        
        for paper_id, paper_chunks in papers.items():
            paper_title = paper_chunks[0].chunk_metadata.get('document_title', paper_id)
            paper_section = f"=== {paper_title} ==="
            
            if total_length + len(paper_section) > self.max_context_length:
                break
                
            context_parts.append(paper_section)
            total_length += len(paper_section)
            
            for chunk in paper_chunks:
                section_info = f"Section: {chunk.section}" if chunk.section else ""
                chunk_text = f"{section_info}\n{chunk.text}\n"
                
                if total_length + len(chunk_text) > self.max_context_length:
                    break
                    
                context_parts.append(chunk_text)
                total_length += len(chunk_text)
        
        return "\n".join(context_parts)
    
    async def _call_openai_async(self, prompt: str) -> str:
        """Call OpenAI API asynchronously with proper error handling."""
        try:
            # Use async client if available
            if self.async_client:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert academic research assistant. Provide accurate, well-structured responses based on the given context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                    frequency_penalty=0.1,
                    presence_penalty=0.1
                )
                return response.choices[0].message.content.strip()
            else:
                # Fallback to sync client in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self.thread_pool,
                    self._sync_openai_call,
                    prompt
                )
                return response
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return f"Error generating response: {str(e)}"

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API with timeout and error handling (backward compatibility)."""
        return await self._call_openai_async(prompt)
    
    async def _openai_api_call(self, prompt: str) -> str:
        """Make the actual OpenAI API call (backward compatibility)."""
        return await self._call_openai_async(prompt)
    
    def _sync_openai_call(self, prompt: str) -> str:
        """Synchronous OpenAI API call (backward compatibility)."""
        response = self.sync_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert academic research assistant. Provide accurate, well-structured responses based on the given context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        return response.choices[0].message.content.strip()
    
    def _calculate_response_confidence(self, query_confidence: float, retrieval_scores: List[float]) -> float:
        """Calculate overall confidence for the response."""
        if not retrieval_scores:
            return 0.0
        
        # Average retrieval confidence
        avg_retrieval_confidence = sum(retrieval_scores) / len(retrieval_scores)
        
        # Combine query classification confidence with retrieval confidence
        overall_confidence = (query_confidence + avg_retrieval_confidence) / 2
        
        # Adjust based on number of sources
        if len(retrieval_scores) >= 3:
            overall_confidence *= 1.1  # Boost for multiple sources
        elif len(retrieval_scores) == 1:
            overall_confidence *= 0.9  # Reduce for single source
        
        return min(overall_confidence, 1.0)
    
    def process_batch_queries(self, queries: List[str]) -> List[QueryResponse]:
        """Process multiple queries in batch."""
        # Use asyncio to process queries concurrently
        async def process_all():
            tasks = [self.process_query(query) for query in queries]
            return await asyncio.gather(*tasks)
        
        return asyncio.run(process_all())
    
    def extract_metrics_from_text(self, text: str) -> Dict[str, List[float]]:
        """Extract numerical metrics from text using regex patterns."""
        metrics = {}
        
        # Define metric patterns
        metric_patterns = {
            'accuracy': r'\b(?:accuracy|acc)\s*[:\-=]?\s*(\d+\.?\d*%?|\d*\.\d+%?)',
            'precision': r'\bprecision\s*[:\-=]?\s*(\d+\.?\d*%?|\d*\.\d+%?)',
            'recall': r'\brecall\s*[:\-=]?\s*(\d+\.?\d*%?|\d*\.\d+%?)',
            'f1': r'\bf1[\-\s]?score\s*[:\-=]?\s*(\d+\.?\d*%?|\d*\.\d+%?)',
            'auc': r'\bauc\s*[:\-=]?\s*(\d+\.?\d*%?|\d*\.\d+%?)',
            'loss': r'\bloss\s*[:\-=]?\s*(\d+\.?\d*|\d*\.\d+)'
        }
        
        for metric_name, pattern in metric_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Convert to float, handling percentages
                values = []
                for match in matches:
                    try:
                        if '%' in match:
                            value = float(match.replace('%', '')) / 100.0
                        else:
                            value = float(match)
                        values.append(value)
                    except ValueError:
                        continue
                if values:
                    metrics[metric_name] = values
        
        return metrics
    
    def get_query_suggestions(self, context: str = "") -> List[str]:
        """Get query suggestions based on available content."""
        suggestions = []
        
        # Get suggestions for each query type
        for query_type in QueryType:
            if query_type != QueryType.UNKNOWN:
                type_suggestions = self.classifier.get_query_suggestions(query_type)
                suggestions.extend(type_suggestions)
        
        return suggestions[:10]  # Return top 10 suggestions 

    def _search_similar_chunks(self, query: str, query_type: QueryType, top_k: int = 5) -> List[TextChunk]:
        """Search for similar chunks with optimized filtering and scoring."""
        logger = get_logger(__name__)
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Get dynamic threshold based on query type
        threshold = self._get_similarity_threshold(query_type)
        
        # Optimize candidate pool size - get fewer candidates initially
        candidate_k = min(top_k * 2, 15)  # More efficient than top_k * 3
        search_results = self.vector_store.search(query_embedding, top_k=candidate_k)
        
        # Define priority sections (most important first)
        priority_sections = [
            'Abstract', 'Introduction', 'Results', 'Conclusions', 'Conclusion',
            'Background', 'Related Work', 'Experimental Evaluation', 'Discussion'
        ]
        
        # Define metadata sections to filter out
        metadata_sections = [
            'Title_Abstract', 'References', 'Bibliography', 'Appendix',
            '𝐴1', '𝐴2', '𝐵1', '𝐵2', '𝐵3', '𝐵4', '𝐵5', '𝐵6',  # Mathematical symbols
            'Nnf', 'Th'  # Abbreviated sections
        ]
        
        # Filter and score chunks
        scored_chunks = []
        
        for chunk, similarity in search_results:
            if similarity < threshold:
                continue
                
            section = chunk.section
            
            # Skip metadata-only sections
            if any(metadata_section in section for metadata_section in metadata_sections):
                continue
                
            # Skip very short chunks (likely headers)
            if len(chunk.text.strip()) < 50:
                continue
            
            # Calculate content quality score
            content_quality = self._calculate_content_quality(chunk, query)
            
            # Skip low-quality content
            if content_quality < 0.1:
                continue
                
            # Calculate priority score
            priority_score = 0
            for i, priority_section in enumerate(priority_sections):
                if priority_section.lower() in section.lower():
                    priority_score = len(priority_sections) - i
                    break
            
            # Calculate composite score
            length_bonus = min(len(chunk.text) / 200, 1.0)
            total_score = (
                0.4 * similarity +
                0.3 * content_quality +
                0.2 * priority_score / len(priority_sections) +
                0.1 * length_bonus
            )
            
            scored_chunks.append({
                'chunk': chunk,
                'similarity': similarity,
                'content_quality': content_quality,
                'total_score': total_score
            })
        
        # Remove duplicates before sorting
        chunk_pairs = [(item['chunk'], item['total_score']) for item in scored_chunks]
        deduplicated = self._remove_duplicate_chunks(chunk_pairs, similarity_threshold=0.85)
        
        # Update scored_chunks with deduplicated results
        deduplicated_dict = {id(chunk): score for chunk, score in deduplicated}
        scored_chunks = [
            item for item in scored_chunks 
            if id(item['chunk']) in deduplicated_dict
        ]
        
        # Sort by total score (highest first)
        scored_chunks.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Take top_k results
        selected_chunks = scored_chunks[:top_k]
        
        logger.info(f"Found {len(selected_chunks)} high-quality chunks above {threshold} threshold")
        
        # Log what we found for debugging
        for i, chunk_data in enumerate(selected_chunks):
            chunk = chunk_data['chunk']
            logger.info(f"  {i+1}. Section: '{chunk.section}' (score: {chunk_data['total_score']:.2f}, quality: {chunk_data['content_quality']:.2f})")
            logger.info(f"     Preview: {chunk.text[:100]}...")
        
        return [chunk_data['chunk'] for chunk_data in selected_chunks]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.metrics.get_stats()
    
    def clear_cache(self):
        """Clear the response cache."""
        if self.response_cache:
            self.response_cache.clear()
            self.logger.info("Response cache cleared")
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics.reset()
        self.logger.info("Performance metrics reset")
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False) 