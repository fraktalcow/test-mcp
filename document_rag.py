import os
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv
from lightrag_hku import LightRAG
from lightrag_hku.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag_hku.kg.shared_storage import initialize_pipeline_status

# Load environment variables
load_dotenv()

WORKING_DIR = "./documents"

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

async def initialize_rag() -> LightRAG:
    """Initialize and return a LightRAG instance"""
    if not os.getenv("OPENAI_API_KEY"):
        raise Exception("OPENAI_API_KEY environment variable is not set")
        
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def process_document(text: str) -> Dict[str, Any]:
    """Process a document and verify its processing"""
    try:
        rag = await initialize_rag()
        await rag.ainsert(text)
        return {
            "status": "success",
            "message": "Document processed successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        if 'rag' in locals():
            await rag.finalize_storages()

async def query_document(query: str) -> Dict[str, Any]:
    """Query the RAG system with the given query"""
    try:
        rag = await initialize_rag()
        response = await rag.aquery(
            query,
            mode="hybrid",
            stream=False,
            top_k=60,
            max_token_for_text_unit=4000
        )
        
        if not response:
            return {
                "response": "No relevant information found in the documents.",
                "status": "no_context"
            }
            
        return {
            "response": response,
            "status": "success"
        }
    except Exception as e:
        return {
            "response": str(e),
            "status": "error"
        }
    finally:
        if 'rag' in locals():
            await rag.finalize_storages() 