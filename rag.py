import os
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    UnstructuredFileLoader
)
from langchain.schema import Document
import tempfile
import hashlib
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAG:
    def __init__(self):
        logger.debug("Initializing RAG system")
        try:
            self.embeddings = OpenAIEmbeddings()
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            self.vector_store = None
            self.documents = {}
            self.document_metadata = {}
            self.persist_directory = "chroma_db"
            self.max_document_size = 10 * 1024 * 1024  # 10MB limit
            self.max_chunks = 200
            os.makedirs(self.persist_directory, exist_ok=True)
            self._load_metadata()
            self._load_vector_store()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            raise

    def _load_vector_store(self):
        """Load existing vector store if available."""
        try:
            if os.path.exists(self.persist_directory):
                logger.debug("Loading existing vector store")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            self.vector_store = None

    def _load_metadata(self):
        """Load document metadata from disk."""
        metadata_path = os.path.join(self.persist_directory, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                logger.debug("Loading document metadata")
                with open(metadata_path, 'r') as f:
                    self.document_metadata = json.load(f)
                logger.info("Document metadata loaded successfully")
            except Exception as e:
                logger.error(f"Error loading metadata: {str(e)}")
                self.document_metadata = {}

    def _save_metadata(self):
        """Save document metadata to disk."""
        metadata_path = os.path.join(self.persist_directory, "metadata.json")
        try:
            logger.debug("Saving document metadata")
            with open(metadata_path, 'w') as f:
                json.dump(self.document_metadata, f)
            logger.info("Document metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")

    def _get_loader(self, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        loaders = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.doc': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader,
            '*': UnstructuredFileLoader  # Fallback for other file types
        }
        loader = loaders.get(ext, loaders['*'])
        logger.debug(f"Selected loader for {file_path}: {loader.__name__}")
        return loader

    def _generate_document_id(self, contents: bytes) -> str:
        """Generate a unique document ID based on file content."""
        try:
            # Use a more robust hashing method
            return hashlib.sha256(contents).hexdigest()
        except Exception as e:
            logger.error(f"Error generating document ID: {str(e)}")
            # Fallback to timestamp-based ID
            return f"doc_{int(datetime.now().timestamp())}"

    def _check_file_size(self, file_path: str) -> bool:
        try:
            size = os.path.getsize(file_path)
            logger.debug(f"File size: {size} bytes")
            return size <= self.max_document_size
        except Exception as e:
            logger.error(f"Error checking file size: {str(e)}")
            return False

    def _generate_reference_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate a reference ID in the format doc_id.chunk_index."""
        ref_id = f"{doc_id}.{chunk_index}"
        logger.debug(f"Generated reference ID: {ref_id}")
        return ref_id

    def process_document(self, contents: bytes, file_name: str) -> bool:
        """Process a document from its contents."""
        try:
            logger.debug(f"Processing document: {file_name}")
            
            # Generate document ID first
            doc_id = self._generate_document_id(contents)
            if doc_id in self.document_metadata:
                logger.info(f"Document {file_name} was already processed")
                return True

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
                temp_file.write(contents)
                temp_path = temp_file.name

            try:
                if not self._check_file_size(temp_path):
                    logger.warning(f"Document {file_name} is too large")
                    return False

                loader_class = self._get_loader(temp_path)
                loader = loader_class(temp_path)
                documents = loader.load()
                
                if not documents:
                    logger.warning(f"Document {file_name} is empty or could not be processed")
                    return False

                # Add metadata to each document
                for i, doc in enumerate(documents):
                    ref_id = self._generate_reference_id(doc_id, i)
                    doc.metadata.update({
                        'source': file_name,
                        'doc_id': doc_id,
                        'chunk_index': i,
                        'reference_id': ref_id,
                        'page': getattr(doc.metadata, 'page', None) or i + 1
                    })

                # Use smaller chunks for better context matching
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=100,
                    length_function=len,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                )
                
                texts = self.text_splitter.split_documents(documents)
                
                if len(texts) > self.max_chunks:
                    logger.warning(f"Document {file_name} has too many chunks")
                    return False

                if not self.vector_store:
                    logger.debug("Creating new vector store")
                    self.vector_store = Chroma.from_documents(
                        documents=texts,
                        embedding=self.embeddings,
                        persist_directory=self.persist_directory
                    )
                else:
                    logger.debug("Adding documents to existing vector store")
                    self.vector_store.add_documents(texts)

                # Store metadata
                self.document_metadata[doc_id] = {
                    'file_name': file_name,
                    'chunks': len(texts),
                    'processed_at': datetime.now().isoformat(),
                    'size': len(contents),
                    'ref_ids': [self._generate_reference_id(doc_id, i) for i in range(len(texts))]
                }
                self._save_metadata()
                logger.info(f"Document {file_name} processed successfully")
                return True

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {str(e)}")

        except Exception as e:
            logger.error(f"Error processing document {file_name}: {str(e)}")
            return False

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        try:
            # Simple keyword extraction
            words = text.lower().split()
            # Remove common words and short words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            keywords = [w for w in words if len(w) > 3 and w not in stop_words]
            # Take most frequent words
            from collections import Counter
            return [w for w, _ in Counter(keywords).most_common(10)]
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    def get_relevant_context(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Get relevant context for a query."""
        if not self.vector_store:
            logger.warning("No vector store available")
            return []
        
        try:
            logger.debug(f"Getting relevant context for query: {query}")
            
            # First check if we have any documents
            if not self.document_metadata:
                logger.warning("No documents available in the system")
                return []
            
            # Get relevant chunks
            docs = self.vector_store.similarity_search_with_score(query, k=k*2)
            
            results = []
            for doc, score in docs:
                if score < 0.9:  # More lenient threshold
                    results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': float(score)
                    })
            
            # Sort by relevance score and take top k
            results.sort(key=lambda x: x['score'])
            results = results[:k]
            
            if not results:
                logger.warning(f"No relevant chunks found for query: {query}")
            else:
                logger.info(f"Found {len(results)} relevant chunks")
            
            return results

        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return []

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents."""
        try:
            logger.debug("Listing documents")
            return [
                {
                    'name': metadata['file_name'],
                    'size': metadata['size'],
                    'chunks': metadata['chunks'],
                    'processed_at': metadata['processed_at'],
                    'ref_ids': metadata.get('ref_ids', [])
                }
                for metadata in self.document_metadata.values()
            ]
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []

    def get_reference_content(self, ref_id: str) -> Optional[Dict[str, Any]]:
        """Get content for a specific reference ID."""
        try:
            logger.debug(f"Getting content for reference: {ref_id}")
            doc_id, chunk_index = ref_id.split('.')
            if doc_id not in self.document_metadata:
                logger.warning(f"Document ID {doc_id} not found")
                return None

            results = self.vector_store.similarity_search(
                query="",
                k=1,
                filter={"doc_id": doc_id, "chunk_index": int(chunk_index)}
            )

            if not results:
                logger.warning(f"No content found for reference {ref_id}")
                return None

            doc = results[0]
            return {
                'content': doc.page_content,
                'metadata': doc.metadata
            }

        except Exception as e:
            logger.error(f"Error getting reference content: {str(e)}")
            return None

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks."""
        try:
            logger.debug(f"Deleting document: {doc_id}")
            if doc_id not in self.document_metadata:
                logger.warning(f"Document {doc_id} not found")
                return False

            del self.document_metadata[doc_id]
            self._rebuild_vector_store()
            self._save_metadata()
            logger.info(f"Document {doc_id} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    def _rebuild_vector_store(self):
        """Rebuild the vector store from remaining documents."""
        try:
            logger.debug("Rebuilding vector store")
            if os.path.exists(self.persist_directory):
                import shutil
                shutil.rmtree(self.persist_directory)
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vector_store = None
            self._load_vector_store()
            logger.info("Vector store rebuilt successfully")
        except Exception as e:
            logger.error(f"Error rebuilding vector store: {str(e)}")

    def cleanup(self):
        """Clean up resources."""
        try:
            logger.debug("Cleaning up resources")
            if os.path.exists(self.persist_directory):
                import shutil
                shutil.rmtree(self.persist_directory)
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vector_store = None
            self.document_metadata = {}
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}") 