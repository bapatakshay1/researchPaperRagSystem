"""
Command Line Interface for the Intelligent Document Processing Agent.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.markdown import Markdown
import json

from extraction.enhanced_pdf_processor import PDFProcessor
from embedding.enhanced_chunker import TextChunker
from embedding.multimodal_embedder import Embedder as DocumentEmbedder
from vector_store.enhanced_vector_store import FAISSVectorStore as FAISSStore
from query_engine.query_processor import QueryProcessor
from utils.logger import get_logger


class CLIInterface:
    """Command Line Interface for IDPA."""
    
    def __init__(self, config, logger):
        """Initialize CLI interface."""
        self.config = config
        self.logger = logger
        self.console = Console()
        
        # Initialize components
        self.pdf_processor = None
        self.chunker = None
        self.embedder = None
        self.vector_store = None
        self.query_processor = None
        
        # State tracking
        self.initialized = False
        self.papers_loaded = []
        
    def display_welcome(self):
        """Display welcome message."""
        welcome_text = """
# 🧠 Intelligent Document Processing Agent (IDPA)

Welcome to IDPA! An AI-powered system for processing and analyzing academic PDF papers.

## Features:
- 📄 **PDF Processing**: Extract structured content from academic papers
- 🔍 **Intelligent Search**: Query papers using natural language
- 📊 **Comparisons**: Compare findings across multiple papers
- 📈 **Metric Extraction**: Extract performance metrics and results
- 💬 **Interactive Queries**: Ask questions about your research collection

Type `help` at any time for available commands.
        """
        
        self.console.print(Panel(Markdown(welcome_text), title="IDPA", border_style="blue"))
    
    def _initialize_components(self):
        """Initialize all IDPA components."""
        if self.initialized:
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Initializing IDPA components...", total=None)
            
            try:
                # Initialize PDF processor
                progress.update(task, description="Loading PDF processor...")
                self.pdf_processor = PDFProcessor(self.config)
                
                # Initialize text chunker
                progress.update(task, description="Loading text chunker...")
                self.chunker = TextChunker(self.config)
                
                # Initialize embedder
                progress.update(task, description="Loading embedding model...")
                self.embedder = DocumentEmbedder(self.config)
                
                # Initialize vector store
                progress.update(task, description="Setting up vector store...")
                self.vector_store = FAISSStore(config=self.config)
                
                # Try to load existing vector store
                try:
                    self.vector_store.load()
                    self.console.print("✅ Loaded existing vector store", style="green")
                except FileNotFoundError:
                    self.console.print("📝 Will create new vector store", style="yellow")
                
                # Initialize query processor
                progress.update(task, description="Setting up query processor...")
                self.query_processor = QueryProcessor(self.config, self.vector_store, self.embedder)
                
                self.initialized = True
                progress.update(task, description="✅ Initialization complete!")
                
            except Exception as e:
                progress.update(task, description=f"❌ Initialization failed: {e}")
                self.logger.error(f"Failed to initialize components: {e}")
                raise
    
    def handle_upload(self, path: str, parallel: bool = False):
        """Handle PDF upload and processing."""
        self._initialize_components()
        
        path = Path(path)
        if not path.exists():
            self.console.print(f"❌ Path not found: {path}", style="red")
            return
        
        # Collect PDF files
        pdf_files = []
        if path.is_file() and path.suffix.lower() == '.pdf':
            pdf_files = [path]
        elif path.is_dir():
            pdf_files = list(path.rglob("*.pdf"))
        else:
            self.console.print(f"❌ Invalid path or not a PDF: {path}", style="red")
            return
        
        if not pdf_files:
            self.console.print("❌ No PDF files found", style="red")
            return
        
        self.console.print(f"📄 Found {len(pdf_files)} PDF file(s)")
        
        # Copy files to data directory
        data_dir = Path(self.config.data_dir)
        copied_files = []
        
        with Progress(console=self.console) as progress:
            task = progress.add_task("Copying files...", total=len(pdf_files))
            
            for pdf_file in pdf_files:
                try:
                    dest_file = data_dir / pdf_file.name
                    if not dest_file.exists():
                        import shutil
                        shutil.copy2(pdf_file, dest_file)
                        copied_files.append(dest_file)
                    else:
                        self.console.print(f"⚠️  File already exists: {pdf_file.name}", style="yellow")
                        copied_files.append(dest_file)
                    
                    progress.advance(task)
                except Exception as e:
                    self.console.print(f"❌ Failed to copy {pdf_file.name}: {e}", style="red")
        
        self.console.print(f"✅ Uploaded {len(copied_files)} files to {data_dir}")
        
        # Automatically trigger extraction
        if copied_files and Confirm.ask("Start content extraction now?"):
            self.handle_extract()
    
    def handle_extract(self, force: bool = False):
        """Handle content extraction from PDFs."""
        self._initialize_components()
        
        data_dir = Path(self.config.data_dir)
        outputs_dir = Path(self.config.outputs_dir)
        
        pdf_files = list(data_dir.glob("*.pdf"))
        if not pdf_files:
            self.console.print("❌ No PDF files found in data directory", style="red")
            return
        
        # Filter files that need extraction
        files_to_extract = []
        for pdf_file in pdf_files:
            output_file = outputs_dir / f"{pdf_file.stem}.json"
            if force or not output_file.exists():
                files_to_extract.append(pdf_file)
        
        if not files_to_extract:
            self.console.print("✅ All files already extracted", style="green")
            return
        
        self.console.print(f"📄 Extracting content from {len(files_to_extract)} files...")
        
        with Progress(console=self.console) as progress:
            task = progress.add_task("Extracting content...", total=len(files_to_extract))
            
            for pdf_file in files_to_extract:
                try:
                    progress.update(task, description=f"Processing {pdf_file.name}...")
                    
                    # Extract content
                    content = self.pdf_processor.extract_from_file(str(pdf_file))
                    
                    # Save extracted content
                    output_file = outputs_dir / f"{pdf_file.stem}.json"
                    self.pdf_processor.save_extracted_content(content, str(output_file))
                    
                    self.papers_loaded.append({
                        'file': pdf_file.name,
                        'title': content.title,
                        'output': output_file
                    })
                    
                    progress.advance(task)
                    
                except Exception as e:
                    self.console.print(f"❌ Failed to extract {pdf_file.name}: {e}", style="red")
                    progress.advance(task)
        
        self.console.print(f"✅ Extracted content from {len(files_to_extract)} files")
        
        # Automatically trigger embedding generation
        if files_to_extract and Confirm.ask("Generate embeddings now?"):
            self.handle_embed()
    
    def handle_embed(self, model: str = None):
        """Handle embedding generation."""
        self._initialize_components()
        
        outputs_dir = Path(self.config.outputs_dir)
        json_files = list(outputs_dir.glob("*.json"))
        
        if not json_files:
            self.console.print("❌ No extracted content found. Run extraction first.", style="red")
            return
        
        self.console.print(f"🔮 Generating embeddings for {len(json_files)} papers...")
        
        all_chunks = []
        
        with Progress(console=self.console) as progress:
            chunk_task = progress.add_task("Chunking text...", total=len(json_files))
            
            for json_file in json_files:
                try:
                    progress.update(chunk_task, description=f"Chunking {json_file.name}...")
                    
                    # Load extracted content
                    with open(json_file, 'r', encoding='utf-8') as f:
                        content_dict = json.load(f)
                    
                    # Reconstruct ExtractedContent object
                    from extraction.enhanced_pdf_processor import ExtractedContent
                    content = ExtractedContent(**content_dict)
                    
                    # Generate chunks using file path for better document identification
                    source_pdf_path = Path(self.config.data_dir) / f"{json_file.stem}.pdf"
                    chunks = self.chunker.chunk_extracted_content(content, str(source_pdf_path))
                    chunks = self.chunker.optimize_chunks_for_retrieval(chunks)
                    
                    all_chunks.extend(chunks)
                    progress.advance(chunk_task)
                    
                except Exception as e:
                    self.console.print(f"❌ Failed to chunk {json_file.name}: {e}", style="red")
                    progress.advance(chunk_task)
            
            # Generate embeddings
            embed_task = progress.add_task("Generating embeddings...", total=len(all_chunks))
            embeddings = {}
            
            batch_size = 50  # Process in batches
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                batch_embeddings = self.embedder.embed_chunks(batch)
                embeddings.update(batch_embeddings)
                
                progress.advance(embed_task, advance=len(batch))
            
            # Add to vector store
            store_task = progress.add_task("Building vector index...", total=None)
            self.vector_store.add_embeddings(embeddings, all_chunks)
            
            # Save vector store
            self.vector_store.save()
        
        self.console.print(f"✅ Generated embeddings for {len(all_chunks)} chunks")
        self._display_collection_stats()
    
    def handle_query(self, question: str, top_k: int = 5):
        """Handle single query."""
        self._initialize_components()
        
        if self.vector_store.index is None or self.vector_store.index.ntotal == 0:
            self.console.print("❌ No document collection loaded. Please upload and process documents first.", style="red")
            return
        
        self.console.print(f"🔍 Processing query: [bold]{question}[/bold]")
        
        with self.console.status("Generating response..."):
            try:
                # Process query asynchronously
                response = asyncio.run(self.query_processor.process_query(question, top_k))
                
                # Display response
                self._display_query_response(response)
                
            except Exception as e:
                self.console.print(f"❌ Query failed: {e}", style="red")
                self.logger.error(f"Query processing failed: {e}")
    
    def start_interactive_session(self):
        """Start interactive query session."""
        self._initialize_components()
        
        if self.vector_store.index is None or self.vector_store.index.ntotal == 0:
            self.console.print("❌ No document collection loaded. Please upload and process documents first.", style="red")
            return
        
        self.console.print(Panel(
            "🚀 Interactive Query Session Started\n\n"
            "Enter your questions about the document collection.\n"
            "Commands: 'help', 'stats', 'papers', 'exit'", 
            title="Interactive Mode"
        ))
        
        while True:
            try:
                query = Prompt.ask("\n[bold blue]Query[/bold blue]")
                
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                elif query.lower() == 'help':
                    self._display_interactive_help()
                elif query.lower() == 'stats':
                    self._display_collection_stats()
                elif query.lower() == 'papers':
                    self._display_papers_list()
                elif query.lower().startswith('search in '):
                    # Extract document identifier from command
                    doc_query = query[10:].strip()  # Remove 'search in '
                    if ' for ' in doc_query:
                        doc_identifier, search_query = doc_query.split(' for ', 1)
                        self._handle_document_search(doc_identifier.strip(), search_query.strip())
                    else:
                        self.console.print("❌ Usage: 'search in <document_name> for <query>'", style="red")
                elif query.strip():
                    self.handle_query(query)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"❌ Error: {e}", style="red")
        
        self.console.print("👋 Interactive session ended")
    
    def _display_query_response(self, response):
        """Display formatted query response."""
        # Response header
        header_table = Table(show_header=False, box=None)
        header_table.add_column(style="cyan")
        header_table.add_column()
        
        header_table.add_row("Query Type:", response.query_type.value.title())
        header_table.add_row("Confidence:", f"{response.confidence:.2%}")
        header_table.add_row("Processing Time:", f"{response.processing_time:.2f}s")
        header_table.add_row("Sources:", str(len(response.sources)))
        
        self.console.print(Panel(header_table, title="Query Analysis"))
        
        # Main answer
        self.console.print(Panel(
            Markdown(response.answer),
            title="Answer",
            border_style="green"
        ))
        
        # Sources
        if response.sources:
            sources_table = Table(title="Sources", show_header=True)
            sources_table.add_column("Paper", style="cyan")
            sources_table.add_column("Section", style="magenta")
            sources_table.add_column("Preview", style="white")
            
            for i, chunk in enumerate(response.sources[:5]):  # Show top 5 sources
                paper_title = chunk.chunk_metadata.get('document_title', chunk.document_id)[:30] + "..."
                section = chunk.section or "Unknown"
                preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
                
                sources_table.add_row(paper_title, section, preview)
            
            self.console.print(sources_table)
    
    def _display_interactive_help(self):
        """Display help for interactive mode."""
        help_text = """
[bold blue]🔍 Query Commands:[/bold blue]
• Type any question to search across all documents
• [cyan]search in <document> for <query>[/cyan] - Search within a specific document
• [cyan]stats[/cyan] - Show collection statistics
• [cyan]papers[/cyan] - List all documents in the collection
• [cyan]help[/cyan] - Show this help message
• [cyan]exit[/cyan] or [cyan]quit[/cyan] - Exit interactive mode

[bold blue]📝 Examples:[/bold blue]
• "What are the main findings about AI in marketing?"
• "search in marketing_paper for performance metrics"
• "Compare methodologies across papers"
        """
        self.console.print(Panel(help_text, title="Interactive Help"))
    
    def _display_collection_stats(self):
        """Display comprehensive collection statistics."""
        if self.vector_store is None:
            self.console.print("❌ Vector store not initialized", style="red")
            return
        
        stats = self.vector_store.get_stats()
        
        # Main statistics table
        stats_table = Table(title="📊 Collection Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Documents", str(stats.get('total_documents', 0)))
        stats_table.add_row("Total Chunks", str(stats.get('total_vectors', 0)))
        stats_table.add_row("Embedding Dimension", str(stats.get('dimension', 0)))
        stats_table.add_row("Similarity Threshold", str(stats.get('similarity_threshold', 0)))
        
        self.console.print(stats_table)
        
        # Document details table
        if stats.get('documents'):
            doc_table = Table(title="📄 Document Details")
            doc_table.add_column("Title", style="cyan", no_wrap=False, max_width=40)
            doc_table.add_column("File", style="yellow", max_width=25)
            doc_table.add_column("Chunks", style="green", justify="right")
            doc_table.add_column("Added", style="dim", max_width=20)
            
            for doc in stats['documents']:
                title = doc.get('title', 'Unknown') or 'Unknown'
                file_name = doc.get('file_name', 'Unknown') or 'Unknown'
                chunk_count = str(doc.get('chunk_count', 0))
                
                # Format timestamp
                timestamp = doc.get('processing_timestamp', '')
                if timestamp:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        formatted_time = timestamp[:16]
                else:
                    formatted_time = 'Unknown'
                
                doc_table.add_row(title, file_name, chunk_count, formatted_time)
            
            self.console.print(doc_table)
    
    def _display_papers_list(self):
        """Display list of available papers with search identifiers."""
        if self.vector_store is None:
            self.console.print("❌ Vector store not initialized", style="red")
            return
        
        documents = self.vector_store.get_documents_list()
        
        if not documents:
            self.console.print("❌ No documents found in collection", style="yellow")
            return
        
        self.console.print(f"📚 Available Documents ({len(documents)} total):")
        
        papers_table = Table()
        papers_table.add_column("#", style="dim", width=3)
        papers_table.add_column("Title", style="cyan", no_wrap=False)
        papers_table.add_column("File Name", style="yellow")
        papers_table.add_column("Document ID", style="dim", max_width=25)
        papers_table.add_column("Chunks", style="green", justify="right")
        
        for i, doc in enumerate(documents, 1):
            title = doc.title or "Unknown Title"
            file_name = doc.file_name or "Unknown File"
            doc_id = doc.document_id
            
            # Get chunk count
            chunks = self.vector_store.get_document_chunks(doc.document_id)
            chunk_count = len(chunks)
            
            papers_table.add_row(
                str(i), 
                title, 
                file_name, 
                doc_id, 
                str(chunk_count)
            )
        
        self.console.print(papers_table)
        
        self.console.print("\n💡 [dim]To search within a specific document, use: 'search in <title/filename> for <query>'[/dim]")

    def _handle_document_search(self, doc_identifier: str, search_query: str):
        """Handle search within a specific document."""
        # Find document by name or ID
        documents = self.vector_store.get_documents_list()
        
        target_doc = None
        for doc in documents:
            if (doc_identifier.lower() in doc.title.lower() if doc.title else False) or \
               (doc_identifier.lower() in doc.file_name.lower() if doc.file_name else False) or \
               (doc_identifier == doc.document_id):
                target_doc = doc
                break
        
        if not target_doc:
            self.console.print(f"❌ Document not found: {doc_identifier}", style="red")
            self.console.print("💡 Use 'papers' command to see available documents", style="yellow")
            return
        
        self.console.print(f"🔍 Searching in document: [bold]{target_doc.title or target_doc.file_name}[/bold]")
        
        try:
            # Search within the specific document
            results = self.vector_store.search_within_document(
                search_query, target_doc.document_id, self.embedder, top_k=5
            )
            
            if not results:
                self.console.print("❌ No relevant content found in this document.", style="yellow")
                return
            
            # Display results
            self.console.print(f"\n📋 Found {len(results)} relevant chunks:")
            
            for i, (chunk, score) in enumerate(results, 1):
                self.console.print(f"\n[bold blue]Result {i}[/bold blue] (Score: {score:.3f})")
                self.console.print(f"[dim]Section: {chunk.section} | Chunk: {chunk.chunk_index}[/dim]")
                
                # Truncate long text
                text = chunk.text
                if len(text) > 300:
                    text = text[:300] + "..."
                
                self.console.print(f"[white]{text}[/white]")
                
        except Exception as e:
            self.console.print(f"❌ Document search failed: {e}", style="red")
    
    def display_error(self, message: str):
        """Display error message."""
        self.console.print(f"❌ {message}", style="red")
    
    def display_success(self, message: str):
        """Display success message."""
        self.console.print(f"✅ {message}", style="green")
    
    def display_info(self, message: str):
        """Display info message."""
        self.console.print(f"ℹ️  {message}", style="blue") 