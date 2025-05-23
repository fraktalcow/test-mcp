import os
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, HTTPException
from lightrag.components import LightRAG
from lightrag.components import QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

app = FastAPI()
WORKING_DIR = "./documents"

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

@app.on_event("startup")
async def startup_event():
    if not os.getenv("OPENAI_API_KEY"):
        raise Exception("OPENAI_API_KEY environment variable is not set")
    app.state.rag = await initialize_rag()

@app.post("/upload")
async def upload_document(file: UploadFile):
    try:
        content = await file.read()
        text = content.decode('utf-8')
        
        # Insert document into RAG system
        await app.state.rag.ainsert(text)
        
        # Verify document was processed
        try:
            # Test query to verify document processing
            test_query = "What is this document about?"
            response = await app.state.rag.aquery(
                test_query,
                param=QueryParam(mode="hybrid")
            )
            if not response:
                raise Exception("Document processing verification failed")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Document uploaded but processing verification failed: {str(e)}"
            )
            
        return {
            "message": "Document uploaded and processed successfully",
            "status": "success",
            "verified": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_document(query: str, mode: str = "hybrid") -> Dict[str, Any]:
    try:
        if not app.state.rag:
            raise HTTPException(
                status_code=500,
                detail="RAG system not initialized"
            )
            
        if mode not in ["naive", "local", "global", "hybrid"]:
            raise HTTPException(status_code=400, detail="Invalid mode")
        
        # Get response from RAG
        response = await app.state.rag.aquery(
            query,
            param=QueryParam(mode=mode)
        )
        
        if not response:
            return {
                "response": "No relevant information found in the documents.",
                "status": "no_context"
            }
            
        return {
            "response": response,
            "status": "success",
            "mode": mode
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get the current status of the RAG system."""
    try:
        if not app.state.rag:
            return {
                "status": "not_initialized",
                "message": "RAG system not initialized"
            }
            
        # Check if any documents are processed
        try:
            test_query = "What documents are available?"
            response = await app.state.rag.aquery(
                test_query,
                param=QueryParam(mode="naive")
            )
            has_documents = bool(response)
        except:
            has_documents = False
            
        return {
            "status": "initialized",
            "has_documents": has_documents,
            "working_dir": WORKING_DIR
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'rag'):
        await app.state.rag.finalize_storages()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 