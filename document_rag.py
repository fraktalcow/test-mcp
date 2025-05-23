import os
import asyncio
from typing import Optional
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
        await app.state.rag.ainsert(text)
        return {"message": "Document uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_document(query: str, mode: str = "hybrid"):
    try:
        if mode not in ["naive", "local", "global", "hybrid"]:
            raise HTTPException(status_code=400, detail="Invalid mode")
        
        response = await app.state.rag.aquery(
            query,
            param=QueryParam(mode=mode)
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'rag'):
        await app.state.rag.finalize_storages()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 