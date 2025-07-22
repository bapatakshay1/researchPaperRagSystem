"""
Query type detection and classification for academic paper queries.
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from utils.logger import get_logger


class QueryType(Enum):
    """Types of queries supported by the system."""
    DIRECT_LOOKUP = "direct_lookup"
    SUMMARIZATION = "summarization"
    COMPARISON = "comparison"
    METRIC_EXTRACTION = "metric_extraction"
    GENERAL_SEARCH = "general_search"
    UNKNOWN = "unknown"


@dataclass
class QueryAnalysis:
    """Analysis result for a user query."""
    query_type: QueryType
    confidence: float
    entities: List[str]  # Papers, sections, metrics mentioned
    keywords: List[str]
    intent: str
    parameters: Dict[str, any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class QueryClassifier:
    """Classifies user queries into different types for appropriate handling."""
    
    def __init__(self):
        """Initialize the query classifier."""
        self.logger = get_logger(__name__)
        
        # Patterns for different query types
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Set up regex patterns for query classification."""
        
        # Direct lookup patterns
        self.lookup_patterns = [
            r'\b(?:what is|what are|what does)\b.*\b(?:conclusion|abstract|result|finding|summary)\b',
            r'\b(?:show me|give me|find|get)\b.*\b(?:from|in|of)\b.*\b(?:paper|study|article)\b',
            r'\b(?:title|author|abstract|conclusion|introduction|methodology|results?)\b.*\b(?:paper|study)\b',
            r'\b(?:where|how|when)\b.*\b(?:mentioned|discussed|described|explained)\b'
        ]
        
        # Summarization patterns
        self.summarization_patterns = [
            r'\b(?:summarize|summary|overview|brief)\b',
            r'\b(?:key points|main findings|highlights)\b',
            r'\b(?:what|how) (?:is|are|does).*(?:methodology|approach|method)\b',
            r'\bgive me (?:a|an) (?:overview|summary|brief)\b'
        ]
        
        # Comparison patterns
        self.comparison_patterns = [
            r'\b(?:compare|comparison|vs|versus|against|between)\b',
            r'\b(?:difference|differences|similar|similarities)\b.*\b(?:between|and)\b',
            r'\b(?:which|what) (?:is )?(?:better|best|worse|different)\b',
            r'\b(?:how do|how does).*(?:differ|compare)\b'
        ]
        
        # Metric extraction patterns
        self.metric_patterns = [
            r'\b(?:accuracy|precision|recall|f1|f-score|auc|roc)\b',
            r'\b(?:performance|score|metric|result|outcome)\b',
            r'\b(?:what|how) (?:is|are) the (?:accuracy|results?|scores?|metrics?)\b',
            r'\b(?:\d+\.?\d*%?|\d+\.\d+)\b.*\b(?:accuracy|precision|recall|f1)\b'
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = {
            QueryType.DIRECT_LOOKUP: [re.compile(p, re.IGNORECASE) for p in self.lookup_patterns],
            QueryType.SUMMARIZATION: [re.compile(p, re.IGNORECASE) for p in self.summarization_patterns],
            QueryType.COMPARISON: [re.compile(p, re.IGNORECASE) for p in self.comparison_patterns],
            QueryType.METRIC_EXTRACTION: [re.compile(p, re.IGNORECASE) for p in self.metric_patterns]
        }
        
        # Keywords for different categories
        self.category_keywords = {
            'papers': ['paper', 'study', 'article', 'research', 'publication', 'document'],
            'sections': ['abstract', 'introduction', 'methodology', 'results', 'conclusion', 
                        'discussion', 'background', 'related work', 'future work'],
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'f-score', 'auc', 'roc',
                       'performance', 'score', 'metric', 'result'],
            'comparison': ['compare', 'versus', 'vs', 'against', 'between', 'difference',
                          'similar', 'different', 'better', 'worse', 'best'],
            'summarization': ['summarize', 'summary', 'overview', 'brief', 'key points',
                             'main findings', 'highlights']
        }
    
    def classify(self, query: str) -> QueryAnalysis:
        """
        Classify a user query and return analysis.
        
        Args:
            query: User query string
            
        Returns:
            QueryAnalysis object with classification results
        """
        query = query.strip()
        if not query:
            return QueryAnalysis(
                query_type=QueryType.UNKNOWN,
                confidence=0.0,
                entities=[],
                keywords=[],
                intent="Empty query"
            )
        
        # Calculate scores for each query type
        type_scores = {}
        
        for query_type, patterns in self.compiled_patterns.items():
            score = self._calculate_pattern_score(query, patterns)
            type_scores[query_type] = score
        
        # Determine best match
        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]
        
        # If no pattern matches well, classify as general search
        if best_score < 0.3:
            best_type = QueryType.GENERAL_SEARCH
            best_score = 0.5  # Default confidence for general search
        
        # Extract entities and keywords
        entities = self._extract_entities(query)
        keywords = self._extract_keywords(query)
        
        # Generate intent description
        intent = self._generate_intent(query, best_type)
        
        # Extract parameters specific to query type
        parameters = self._extract_parameters(query, best_type)
        
        return QueryAnalysis(
            query_type=best_type,
            confidence=best_score,
            entities=entities,
            keywords=keywords,
            intent=intent,
            parameters=parameters
        )
    
    def _calculate_pattern_score(self, query: str, patterns: List[re.Pattern]) -> float:
        """Calculate how well a query matches a set of patterns."""
        if not patterns:
            return 0.0
        
        scores = []
        for pattern in patterns:
            match = pattern.search(query)
            if match:
                # Score based on match coverage
                match_length = len(match.group())
                coverage = match_length / len(query)
                scores.append(coverage)
        
        return max(scores) if scores else 0.0
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract mentioned entities (papers, sections, etc.) from query."""
        entities = []
        
        # Extract paper references
        paper_patterns = [
            r'paper [A-Z]',
            r'study [A-Z]',
            r'article [A-Z]',
            r'(?:paper|study|article) (?:titled|called|named) ["\']([^"\']+)["\']',
            r'the ([A-Z][a-z]+ et al\.?) (?:paper|study)'
        ]
        
        for pattern in paper_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        # Extract section references
        section_pattern = r'\b(' + '|'.join(self.category_keywords['sections']) + r')\b'
        section_matches = re.findall(section_pattern, query, re.IGNORECASE)
        entities.extend(section_matches)
        
        # Extract metric references
        metric_pattern = r'\b(' + '|'.join(self.category_keywords['metrics']) + r')\b'
        metric_matches = re.findall(metric_pattern, query, re.IGNORECASE)
        entities.extend(metric_matches)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Simple keyword extraction - remove stop words and get meaningful terms
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Remove very common academic terms that don't add much meaning
        academic_stop_words = {'paper', 'study', 'research', 'article', 'work'}
        keywords = [word for word in keywords if word not in academic_stop_words]
        
        return keywords[:10]  # Limit to most relevant
    
    def _generate_intent(self, query: str, query_type: QueryType) -> str:
        """Generate a human-readable intent description."""
        intent_templates = {
            QueryType.DIRECT_LOOKUP: "Find specific information from papers",
            QueryType.SUMMARIZATION: "Provide a summary or overview",
            QueryType.COMPARISON: "Compare aspects between papers or studies",
            QueryType.METRIC_EXTRACTION: "Extract performance metrics or numerical results",
            QueryType.GENERAL_SEARCH: "Search for relevant information",
            QueryType.UNKNOWN: "Unable to determine intent"
        }
        
        base_intent = intent_templates.get(query_type, "Process query")
        
        # Add more specific details based on entities found
        entities = self._extract_entities(query)
        if entities:
            base_intent += f" (related to: {', '.join(entities[:3])})"
        
        return base_intent
    
    def _extract_parameters(self, query: str, query_type: QueryType) -> Dict[str, any]:
        """Extract parameters specific to the query type."""
        parameters = {}
        
        if query_type == QueryType.COMPARISON:
            # Extract comparison subjects
            comparison_words = ['between', 'and', 'vs', 'versus', 'against']
            for word in comparison_words:
                if word in query.lower():
                    # Try to extract what's being compared
                    parts = re.split(f'\\b{word}\\b', query, flags=re.IGNORECASE)
                    if len(parts) >= 2:
                        parameters['comparison_subjects'] = [p.strip() for p in parts]
                    break
        
        elif query_type == QueryType.METRIC_EXTRACTION:
            # Extract specific metrics mentioned
            metric_mentions = []
            for metric in self.category_keywords['metrics']:
                if metric in query.lower():
                    metric_mentions.append(metric)
            parameters['requested_metrics'] = metric_mentions
        
        elif query_type == QueryType.SUMMARIZATION:
            # Extract what to summarize
            summarize_patterns = [
                r'summarize (?:the )?([^.?!]+)',
                r'summary of ([^.?!]+)',
                r'overview of ([^.?!]+)'
            ]
            for pattern in summarize_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    parameters['summarization_target'] = match.group(1).strip()
                    break
        
        elif query_type == QueryType.DIRECT_LOOKUP:
            # Extract what information is being looked up
            lookup_patterns = [
                r'what (?:is|are) (?:the )?([^.?!]+)',
                r'(?:show|find|get) (?:me )?(?:the )?([^.?!]+)',
                r'where (?:is|are) ([^.?!]+)'
            ]
            for pattern in lookup_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    parameters['lookup_target'] = match.group(1).strip()
                    break
        
        return parameters
    
    def get_query_suggestions(self, query_type: QueryType) -> List[str]:
        """Get example queries for a given query type."""
        suggestions = {
            QueryType.DIRECT_LOOKUP: [
                "What is the conclusion of Paper A?",
                "Show me the abstract from the transformer study",
                "What methodology is used in the vision paper?"
            ],
            QueryType.SUMMARIZATION: [
                "Summarize the methodology of Paper B",
                "Give me an overview of the results section",
                "What are the key findings in this research?"
            ],
            QueryType.COMPARISON: [
                "Compare the results of Paper A and Paper B",
                "What are the differences between CNN and RNN approaches?",
                "Which model performs better on accuracy?"
            ],
            QueryType.METRIC_EXTRACTION: [
                "What is the F1-score reported in Paper C?",
                "Show me all accuracy measurements",
                "What are the performance metrics for the proposed model?"
            ],
            QueryType.GENERAL_SEARCH: [
                "Machine learning optimization techniques",
                "Neural network architectures",
                "Computer vision applications"
            ]
        }
        
        return suggestions.get(query_type, [])
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a query can be processed.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query) > 1000:
            return False, "Query is too long (max 1000 characters)"
        
        # Check for potentially problematic patterns
        if re.search(r'[<>{}[\]\\|`~]', query):
            return False, "Query contains invalid characters"
        
        return True, None 