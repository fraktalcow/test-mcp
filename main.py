from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import tempfile
from dotenv import load_dotenv
from mcp import MCP
from document_rag import initialize_rag, process_document, query_document
import json
import asyncio
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()



# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize MCP and LightRAG
try:
    logger.debug("Initializing MCP and LightRAG")
    mcp = MCP()
    app.state.rag = None  # Will be initialized in startup event
    app.state.has_documents = False  # Track if documents are uploaded
    logger.info("MCP initialized successfully")
except Exception as e:
    logger.error(f"Error initializing MCP: {str(e)}")
    raise

# Create temp directory for uploaded files
os.makedirs("temp", exist_ok=True)

class MessageRequest(BaseModel):
    message: str
    settings: dict
    useRag: bool = False

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.debug(f"New WebSocket connection. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.debug(f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)
        logger.debug(f"Broadcasted message to {len(self.active_connections)} connections")

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    if not os.getenv("OPENAI_API_KEY"):
        raise Exception("OPENAI_API_KEY environment variable is not set")
    app.state.rag = await initialize_rag()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming responses."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            use_rag = data.get("useRag", False)  # Get RAG flag from request

            logger.debug(f"WebSocket message received: {message[:100]}...")

            # Get context from RAG if documents are uploaded and RAG is requested
            context = None
            if use_rag and app.state.has_documents:
                try:
                    rag_response = await query_document(message)
                    if rag_response["status"] == "success" and rag_response.get("documents"):
                        context = [{"content": doc["content"], "metadata": doc["meta"]} for doc in rag_response["documents"]]
                        logger.debug(f"Retrieved RAG context with {len(context)} documents")
                except Exception as e:
                    logger.error(f"Error getting context from RAG: {str(e)}")

            # Process message with context
            await mcp.process_message_stream(message, context, websocket)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": str(e)
            })
        except:
            pass

@app.post("/send_message")
async def send_message(message: MessageRequest):
    """Send a message and get a response."""
    try:
        logger.debug(f"Processing message: {message.message}")

        # Get context from RAG if documents are uploaded
        context = None
        if message.useRag and app.state.has_documents:
            try:
                rag_response = await query_document(message.message)
                if rag_response["status"] == "success" and rag_response.get("documents"):
                    # Format context as list of dicts for MCP
                    context = [{"content": doc["content"], "metadata": doc["meta"]} for doc in rag_response["documents"]]
                    logger.debug(f"Retrieved RAG context with {len(context)} documents")
            except Exception as e:
                logger.error(f"Error getting context from RAG: {str(e)}")

        # Process message with context
        result = await mcp.process_message(message.message, context)

        if "error" in result:
            logger.error(f"Error processing message: {result['error']}")
            return JSONResponse(
                status_code=500,
                content={"error": result["error"], "content": ""}
            )

        # Ensure we always return a properly formatted response
        return JSONResponse(
            content={
                "content": result.get("content", ""),
                "error": None
            }
        )

    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "content": ""}
        )

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        logger.debug(f"Processing document upload: {file.filename}")

        # Validate file
        if not file.filename:
            logger.error("No filename provided")
            raise HTTPException(status_code=400, detail="No filename provided")

        # Check file extension
        allowed_extensions = {'.pdf', '.txt', '.docx', '.doc', '.md'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            logger.error(f"Unsupported file type: {file_ext}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
            )

        # Read file contents
        try:
            contents = await file.read()
            if not contents:
                logger.error("Empty file uploaded")
                raise HTTPException(status_code=400, detail="File is empty")
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            raise HTTPException(status_code=400, detail="Error reading file")

        # Process document with LightRAG
        try:
            metadata = {"filename": file.filename}
            result = await process_document(contents, metadata)

            if result["status"] == "error":
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing document: {result['message']}"
                )

            app.state.has_documents = True
            logger.info(f"Document {file.filename} processed successfully")
            return JSONResponse(content={
                "message": f"Document {file.filename} processed successfully",
                "status": "success"
            })
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing document: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload_document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
