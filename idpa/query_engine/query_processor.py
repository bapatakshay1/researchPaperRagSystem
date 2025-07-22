"""
Main query processor for handling different types of academic paper queries.
"""

import re
import json
from openai import OpenAI
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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
        
        # Initialize LLM client
        self._setup_llm()
        
        # Prompt templates for different query types
        self._setup_prompts()
    
    def _setup_llm(self):
        """Set up the LLM client based on configuration."""
        if self.llm_provider == 'openai':
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key is required")
            # API key is automatically read from environment by OpenAI client
            self.model_name = getattr(self.config, 'openai_model', 'gpt-4')
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
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
    
    async def process_query(self, query: str, top_k: int = None) -> QueryResponse:
        """
        Process a user query and return a comprehensive response.
        
        Args:
            query: User query string
            top_k: Number of top chunks to retrieve
            
        Returns:
            QueryResponse object with answer and metadata
        """
        import time
        start_time = time.time()
        
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
            top_k = top_k or 5  # Default to 5 results
            search_results = self._search_similar_chunks(query, top_k=top_k, threshold=0.3)
            
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
                return QueryResponse(
                    answer="No relevant information found in the document collection.",
                    sources=[],
                    confidence=0.0,
                    query_type=analysis.query_type,
                    processing_time=time.time() - start_time
                )
            
            # Extract chunks and scores
            chunks = [chunk for chunk, score in relevant_chunks]
            scores = [score for chunk, score in relevant_chunks]
            
            # Generate response based on query type
            answer = await self._generate_response(query, analysis, chunks)
            
            # Calculate overall confidence
            confidence = self._calculate_response_confidence(analysis.confidence, scores)
            
            processing_time = time.time() - start_time
            
            return QueryResponse(
                answer=answer,
                sources=chunks,
                confidence=confidence,
                query_type=analysis.query_type,
                processing_time=processing_time,
                metadata={
                    'analysis': asdict(analysis),
                    'retrieval_scores': scores,
                    'num_sources': len(chunks)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return QueryResponse(
                answer=f"An error occurred while processing your query: {str(e)}",
                sources=[],
                confidence=0.0,
                query_type=QueryType.UNKNOWN,
                processing_time=time.time() - start_time
            )
    
    async def _generate_response(
        self, 
        query: str, 
        analysis: QueryAnalysis, 
        chunks: List[TextChunk]
    ) -> str:
        """Generate response using LLM based on query type and context."""
        
        # Prepare context from chunks
        context = self._prepare_context(chunks, analysis.query_type)
        
        # Get appropriate prompt template
        prompt_template = self.prompts.get(analysis.query_type, self.prompts[QueryType.GENERAL_SEARCH])
        
        # Format prompt
        prompt = prompt_template.format(context=context, query=query)
        
        # Generate response using LLM
        if self.llm_provider == 'openai':
            response = await self._call_openai(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        
        return response
    
    def _prepare_context(self, chunks: List[TextChunk], query_type: QueryType) -> str:
        """Prepare context string from retrieved chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            # Add chunk header with metadata
            header = f"--- Source {i+1} ---"
            if chunk.chunk_metadata.get('document_title'):
                header += f"\nPaper: {chunk.chunk_metadata['document_title']}"
            if chunk.section:
                header += f"\nSection: {chunk.section}"
            if chunk.page:
                header += f"\nPage: {chunk.page}"
            
            # Add chunk content
            context_part = f"{header}\n{chunk.text}\n"
            context_parts.append(context_part)
        
        # For comparison queries, try to group by papers
        if query_type == QueryType.COMPARISON:
            return self._organize_context_for_comparison(chunks)
        
        return "\n".join(context_parts)
    
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
        for paper_id, paper_chunks in papers.items():
            paper_title = paper_chunks[0].chunk_metadata.get('document_title', paper_id)
            context_parts.append(f"=== {paper_title} ===")
            
            for chunk in paper_chunks:
                section_info = f"Section: {chunk.section}" if chunk.section else ""
                context_parts.append(f"{section_info}\n{chunk.text}\n")
        
        return "\n".join(context_parts)
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API with timeout and error handling."""
        try:
            # Use asyncio to add timeout
            response = await asyncio.wait_for(
                self._openai_api_call(prompt),
                timeout=self.query_timeout
            )
            return response
        except TimeoutError:
            self.logger.error("OpenAI API call timed out")
            return "Response generation timed out. Please try again with a simpler query."
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return f"Error generating response: {str(e)}"
    
    async def _openai_api_call(self, prompt: str) -> str:
        """Make the actual OpenAI API call."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(
                executor, 
                self._sync_openai_call, 
                prompt
            )
        return response
    
    def _sync_openai_call(self, prompt: str) -> str:
        """Synchronous OpenAI API call."""
        client = OpenAI(api_key=self.config.openai_api_key)
        response = client.chat.completions.create(
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

    def _search_similar_chunks(self, query: str, top_k: int = 5, threshold: float = 0.3) -> List[TextChunk]:
        """Search for similar chunks with improved section prioritization."""
        logger = get_logger(__name__)
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search in vector store
        search_results = self.vector_store.search(query_embedding, top_k=top_k * 3)  # Get more candidates
        
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
        
        # Filter and prioritize chunks
        filtered_chunks = []
        
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
                
            # Check for content vs metadata
            text_lower = chunk.text.lower()
            
            # Content keywords (what we want)
            content_keywords = ['sat', 'smt', 'cnf', 'formula', 'solver', 'enumeration', 'transformation', 'theorem', 'proof', 'algorithm', 'conjunctive', 'normal', 'boolean', 'logic', 'state-of-the-art', 'efficiently', 'conversion', 'approach', 'method', 'technique']
            content_count = sum(1 for keyword in content_keywords if keyword in text_lower)
            
            # Metadata keywords (what we want to avoid)
            metadata_keywords = ['contact', 'email', 'license', 'creative commons', 'authors', 'university', 'disi', 'italy', 'rice university', 'copyright', 'doi.org']
            metadata_count = sum(1 for keyword in metadata_keywords if keyword in text_lower)
            
            # Skip if it's clearly just metadata (no content keywords and has metadata)
            if content_count == 0 and metadata_count > 0:
                continue
                
            # Calculate content ratio
            has_content = content_count > 0
            content_ratio = content_count / max(metadata_count, 1)  # Higher is better
                
            # Calculate priority score
            priority_score = 0
            
            # High priority for important sections
            for i, priority_section in enumerate(priority_sections):
                if priority_section.lower() in section.lower():
                    priority_score = len(priority_sections) - i  # Higher score for earlier sections
                    break
            
            # Bonus for longer content
            content_length_bonus = min(len(chunk.text) / 100, 2)  # Cap at 2 points
            
            # Bonus for higher similarity
            similarity_bonus = similarity * 3
            
            # Bonus for content-rich chunks
            content_bonus = content_ratio * 5  # Scale the content ratio
            
            total_score = priority_score + content_length_bonus + similarity_bonus + content_bonus
            
            filtered_chunks.append({
                'chunk': chunk,
                'similarity': similarity,
                'priority_score': priority_score,
                'total_score': total_score
            })
        
        # Sort by total score (highest first)
        filtered_chunks.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Take top_k results
        selected_chunks = filtered_chunks[:top_k]
        
        logger.info(f"Found {len(selected_chunks)} chunks above {threshold} threshold")
        
        # Log what we found
        for i, chunk_data in enumerate(selected_chunks):
            chunk = chunk_data['chunk']
            logger.info(f"  {i+1}. Section: '{chunk.section}' (score: {chunk_data['total_score']:.2f}, sim: {chunk_data['similarity']:.3f})")
            logger.info(f"     Preview: {chunk.text[:100]}...")
        
        return [chunk_data['chunk'] for chunk_data in selected_chunks] 