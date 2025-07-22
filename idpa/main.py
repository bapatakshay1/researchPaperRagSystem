#!/usr/bin/env python3
"""
Intelligent Document Processing Agent (IDPA)
Main entry point for the system.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
from utils.config import Config
from extraction.enhanced_pdf_processor import PDFProcessor
from embedding.multimodal_embedder import Embedder as DocumentEmbedder
from vector_store.enhanced_vector_store import FAISSVectorStore as FAISSStore
from query_engine.query_processor import QueryProcessor
from interface.cli import CLIInterface


def main():
    """Main entry point for IDPA."""
    parser = argparse.ArgumentParser(description="Intelligent Document Processing Agent")
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload and process PDF documents')
    upload_parser.add_argument('path', help='Path to PDF file or directory containing PDFs')
    upload_parser.add_argument('--parallel', action='store_true', help='Process files in parallel')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract content from uploaded PDFs')
    extract_parser.add_argument('--force', action='store_true', help='Force re-extraction of all files')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings for extracted content')
    embed_parser.add_argument('--model', default='all-MiniLM-L6-v2', help='Embedding model to use')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the document collection')
    query_parser.add_argument('question', help='Question to ask about the documents')
    query_parser.add_argument('--top-k', type=int, default=5, help='Number of top results to retrieve')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive query session')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize logger and config
    logger = setup_logger()
    config = Config()
    
    # Initialize CLI interface
    cli = CLIInterface(config, logger)
    
    try:
        if args.command == 'upload':
            cli.handle_upload(args.path, parallel=args.parallel)
        elif args.command == 'extract':
            cli.handle_extract(force=args.force)
        elif args.command == 'embed':
            cli.handle_embed(model=args.model)
        elif args.command == 'query':
            cli.handle_query(args.question, top_k=args.top_k)
        elif args.command == 'interactive':
            cli.start_interactive_session()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 