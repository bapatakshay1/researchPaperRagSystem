# 🧠 Intelligent Document Processing Agent (IDPA)

A comprehensive AI-powered system for processing, analyzing, and querying academic PDF papers using advanced NLP techniques.

## 🎯 Overview

IDPA is designed to help researchers and academics efficiently work with large collections of research papers by providing:

- **PDF Content Extraction**: Intelligent parsing of academic papers with support for text, tables, figures, and references
- **Semantic Search**: Vector-based similarity search using state-of-the-art embeddings
- **Natural Language Queries**: Ask questions in plain English about your research collection
- **Multi-Paper Comparisons**: Compare methodologies, results, and findings across papers
- **Metric Extraction**: Automatically extract performance metrics and numerical results
- **Interactive CLI**: User-friendly command-line interface with rich formatting

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Input     │    │  Text Chunking  │    │   Embeddings    │
│   (PyMuPDF +    │───▶│   (Semantic)    │───▶│ (Sentence-BERT) │
│    OCR Fallback)│    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Structured JSON │    │ Optimized Chunks│    │  FAISS Vector  │
│   (Sections,    │    │  (Contextual)   │    │     Store       │
│ Tables, Refs)   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Query Engine   │    │  LLM Integration│
                       │ (Classification │◀───│   (GPT-4 /      │
                       │  & Routing)     │    │   Local Model)  │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB RAM
- Optional: CUDA-compatible GPU for faster processing

### Installation

1. **Clone or extract the IDPA package**:
   ```bash
   cd idpa/
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   echo "LOG_LEVEL=INFO" >> .env
   ```

4. **Download NLTK data** (automatic on first run):
   ```python
   import nltk
   nltk.download('punkt')
   ```

### Basic Usage

1. **Upload and process PDFs**:
   ```bash
   python main.py upload /path/to/pdf/files/
   ```

2. **Extract content**:
   ```bash
   python main.py extract
   ```

3. **Generate embeddings**:
   ```bash
   python main.py embed
   ```

4. **Ask questions**:
   ```bash
   python main.py query "What is the conclusion of the transformer paper?"
   ```

5. **Interactive mode**:
   ```bash
   python main.py interactive
   ```

## 📖 Detailed Usage

### Command Line Interface

#### Upload PDFs
```bash
# Upload single file
python main.py upload paper.pdf

# Upload directory of PDFs
python main.py upload /research/papers/

# Upload with parallel processing
python main.py upload /papers/ --parallel
```

#### Extract Content
```bash
# Extract all new PDFs
python main.py extract

# Force re-extraction of all files
python main.py extract --force
```

#### Generate Embeddings
```bash
# Use default model (all-MiniLM-L6-v2)
python main.py embed

# Use specific model
python main.py embed --model sentence-transformers/all-mpnet-base-v2
```

#### Query Documents
```bash
# Single query
python main.py query "Compare CNN and RNN performance"

# Query with more results
python main.py query "What accuracy is reported?" --top-k 10
```

### Query Types

IDPA automatically detects and handles different types of queries:

#### 1. Direct Lookup
```
"What is the conclusion of Paper A?"
"Show me the abstract from the transformer study"
"What methodology is used in the vision paper?"
```

#### 2. Summarization
```
"Summarize the methodology of Paper B"
"Give me an overview of the results section"
"What are the key findings in this research?"
```

#### 3. Comparison
```
"Compare the results of Paper A and Paper B"
"What are the differences between CNN and RNN approaches?"
"Which model performs better on accuracy?"
```

#### 4. Metric Extraction
```
"What is the F1-score reported in Paper C?"
"Show me all accuracy measurements"
"What are the performance metrics for the proposed model?"
```

### Interactive Mode

Start an interactive session for multiple queries:

```bash
python main.py interactive
```

In interactive mode, you can use special commands:
- `help` - Show available commands
- `stats` - Display collection statistics
- `papers` - List loaded papers
- `exit` - Exit interactive mode

## ⚙️ Configuration

The system can be configured via:

1. **Environment variables** (`.env` file)
2. **Configuration file** (`config.json`)
3. **Command line arguments**

### Key Configuration Options

```python
# Text Processing
chunk_size = 512           # Size of text chunks
chunk_overlap = 50         # Overlap between chunks
min_chunk_size = 50        # Minimum chunk size

# Embedding Model
embedding_model = "all-MiniLM-L6-v2"
embedding_device = "cpu"   # or "cuda"

# LLM Settings
llm_provider = "openai"    # "openai", "local"
openai_model = "gpt-4"
max_tokens = 2000
temperature = 0.1

# Vector Store
similarity_threshold = 0.7
max_retrieval_results = 10

# Performance
max_workers = 4            # Parallel processing
batch_size = 32            # Embedding batch size
```

## 📊 System Components

### 1. PDF Processor (`extraction/enhanced_pdf_processor.py`)
- **Primary**: PyMuPDF for fast text extraction
- **Fallback**: pdfplumber for better table extraction
- **OCR**: pytesseract + pdf2image for scanned PDFs
- **Output**: Structured JSON with sections, tables, figures, references

### 2. Text Chunker (`embedding/enhanced_chunker.py`)
- **Semantic Chunking**: Preserves sentence and paragraph boundaries
- **Contextual Enhancement**: Adds section and paper context
- **Optimization**: Filters and enhances chunks for retrieval

### 3. Document Embedder (`embedding/multimodal_embedder.py`)
- **Model**: Sentence-BERT (configurable)
- **Batch Processing**: Efficient embedding generation
- **Normalization**: Cosine similarity optimization
- **Caching**: Save/load embeddings to disk

### 4. Vector Store (`vector_store/enhanced_vector_store.py`)
- **Engine**: FAISS for fast similarity search
- **Index Types**: Flat, IVF, HNSW (configurable)
- **Persistence**: Save/load vector indices
- **Thread Safety**: Concurrent query support

### 5. Query Engine (`query_engine/`)
- **Classification**: Automatic query type detection
- **Routing**: Type-specific processing pipelines
- **LLM Integration**: GPT-4 or local models
- **Response Generation**: Contextual, accurate answers

### 6. CLI Interface (`interface/cli.py`)
- **Rich Formatting**: Tables, panels, progress bars
- **Interactive Mode**: Real-time query processing
- **Error Handling**: User-friendly error messages
- **Statistics**: Collection and performance metrics

## 🔧 Advanced Features

### Custom Models

Use different embedding models:
```bash
# High-quality but slower
python main.py embed --model sentence-transformers/all-mpnet-base-v2

# Multilingual support
python main.py embed --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### Batch Processing

Process multiple PDFs efficiently:
```python
from idpa.extraction.enhanced_pdf_processor import PDFProcessor

processor = PDFProcessor(config)
results = processor.process_multiple_files(pdf_paths, parallel=True)
```

### Custom Query Processing

Extend the query engine:
```python
from idpa.query_engine.query_processor import QueryProcessor

# Add custom query handlers
processor = QueryProcessor(config, vector_store, embedder)
response = await processor.process_query("Your custom query")
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/idpa"
```

#### 2. PDF Processing Failures
```bash
# Install additional dependencies for OCR
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install tesseract-ocr poppler-utils

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

#### 3. Memory Issues
- Reduce `batch_size` in configuration
- Use CPU instead of GPU if running out of VRAM
- Process fewer files at once

#### 4. OpenAI API Issues
- Check API key in `.env` file
- Verify API quota and billing
- Use local models as fallback

#### 5. FAISS Installation
```bash
# CPU version
pip install faiss-cpu

# GPU version (if needed)
pip install faiss-gpu
```

### Performance Optimization

1. **Use GPU for embeddings**:
   ```python
   embedding_device = "cuda"
   ```

2. **Increase batch sizes**:
   ```python
   batch_size = 64  # If you have enough memory
   ```

3. **Use IVF index for large collections**:
   ```python
   # Automatically switches to IVF for >10k documents
   ```

4. **Parallel processing**:
   ```bash
   python main.py upload /papers/ --parallel
   ```

## 📝 Logging and Debugging

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
python main.py query "your question"
```

Log files are saved to `logs/idpa.log` by default.

## 🤝 Contributing

To extend IDPA:

1. **Add new PDF extractors**: Implement in `extraction/`
2. **Custom embedders**: Extend `embedding/multimodal_embedder.py`
3. **New query types**: Add to `query_engine/query_types.py`
4. **Additional interfaces**: Create in `interface/`

## 📄 License

This project is provided as-is for academic and research purposes.

## 🙏 Acknowledgments

- **PyMuPDF**: Fast PDF processing
- **Sentence Transformers**: State-of-the-art embeddings
- **FAISS**: Efficient similarity search
- **Rich**: Beautiful CLI formatting
- **OpenAI**: Advanced language understanding

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review log files in `logs/`
3. Ensure all dependencies are correctly installed
4. Verify your PDF files are not corrupted

---

**Happy Researching! 🎓📚** 