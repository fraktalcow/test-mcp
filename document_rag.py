import os
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components import MemoryRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
import io
from pypdf import PdfReader

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize working directory
WORKING_DIR = os.getenv("WORKING_DIR", "working_dir")
os.makedirs(WORKING_DIR, exist_ok=True)

# Create document store
embedding_dim = 384
document_store = InMemoryDocumentStore(embedding_dim=embedding_dim)

# Initialize retriever and embedder
retriever = MemoryRetriever(document_store=document_store, top_k=3, retrieval_method="embedding")
embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

async def initialize_rag() -> Dict[str, Any]:
    """Initialize the RAG system (dummy for compatibility)"""
    try:
        return {"status": "success", "message": "RAG system initialized successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes"""
    try:
        logger.debug(f"Attempting to extract text from PDF of size {len(pdf_bytes)} bytes")
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            logger.debug(f"Extracted {len(page_text)} characters from page {i+1}")
            text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

async def process_document(content: str | bytes, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process and add document to document store"""
    try:
        logger.debug(f"Processing document with metadata: {metadata}")

        # Handle PDF content
        if isinstance(content, bytes) and metadata and metadata.get("filename", "").lower().endswith(".pdf"):
            logger.debug("Detected PDF file, extracting text...")
            content = extract_text_from_pdf(content)
            logger.debug(f"Extracted {len(content)} characters from PDF")

        # Handle text encoding
        elif isinstance(content, bytes):
            logger.debug("Detected bytes content, attempting to decode...")
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    content = content.decode('latin-1')
                except UnicodeDecodeError:
                    logger.error("Failed to decode content with both UTF-8 and Latin-1")
                    return {"status": "error", "message": "Could not decode document content"}

        # Clean and normalize text
        content = content.strip()
        if not content:
            logger.error("Empty document content after processing")
            return {"status": "error", "message": "Empty document content"}

        logger.debug(f"Creating document with {len(content)} characters")
        doc = {
            "content": content,
            "meta": metadata or {}
        }

        logger.debug("Writing document to store...")
        document_store.write_documents([doc])

        logger.debug("Updating embeddings...")
        docs = document_store.filter_documents()
        docs_with_embeddings = embedder.run(documents=docs)["documents"]
        document_store.write_documents(documents=docs_with_embeddings, policy="OVERWRITE")

        logger.debug("Document processing completed successfully")
        return {"status": "success", "message": "Document processed successfully"}
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Error processing document: {str(e)}"}

async def query_document(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Query document store"""
    try:
        logger.debug(f"Processing query: {query}")

        # Check if document store is empty
        if not document_store.filter_documents():
            logger.debug("Document store is empty, returning empty result")
            return {
                "status": "success",
                "documents": []
            }

        # Embed the query
        query_embedding = embedder.run(texts=[query])["embeddings"][0]
        # Retrieve documents
        results = retriever.run(query_embedding=query_embedding, top_k=top_k)
        documents = []
        for doc in results["documents"]:
            documents.append({
                "content": doc.content,
                "meta": doc.meta,
                "score": getattr(doc, "score", None)
            })
        logger.debug(f"Found {len(documents)} matching documents")
        return {
            "status": "success",
            "documents": documents
        }
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}
