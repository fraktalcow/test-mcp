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
from rag import RAG
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize RAG and MCP
try:
    logger.debug("Initializing RAG and MCP")
    rag = RAG()
    mcp = MCP()
    logger.info("RAG and MCP initialized successfully")
except Exception as e:
    logger.error(f"Error initializing RAG and MCP: {str(e)}")
    raise

# Create temp directory for uploaded files
os.makedirs("temp", exist_ok=True)

class MessageRequest(BaseModel):
    message: str
    settings: dict
    useRag: bool = False
    documents: List[str] = []

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming responses."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            logger.debug(f"WebSocket message received: {message[:100]}...")
            
            # Always try to get relevant context
            context = rag.get_relevant_context(message)
            logger.debug(f"Retrieved {len(context) if context else 0} relevant chunks")
            
            # Process message with context
            await mcp.process_message_stream(message, context, websocket)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except:
            pass

@app.post("/send_message")
async def send_message(message: MessageRequest):
    """Send a message and get a response."""
    try:
        logger.debug(f"Processing message: {message.message}")
        
        # Always try to get relevant context first
        context = rag.get_relevant_context(message.message)
        logger.debug(f"Retrieved {len(context) if context else 0} relevant chunks")
        
        # Process message with context if available
        result = await mcp.process_message(message.message, context)
        
        if "error" in result:
            logger.error(f"Error processing message: {result['error']}")
            return {"error": result["error"]}
        
        return result

    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}")
        return {"error": str(e)}

@app.get("/reference/{ref_id}")
async def get_reference(ref_id: str):
    """Get content for a specific reference ID."""
    try:
        logger.debug(f"Getting reference content for: {ref_id}")
        content = rag.get_reference_content(ref_id)
        if not content:
            logger.warning(f"Reference not found: {ref_id}")
            raise HTTPException(status_code=404, detail="Reference not found")
        return JSONResponse(content=content)
    except Exception as e:
        logger.error(f"Error getting reference content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        
        # Process document
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                temp_file.write(contents)
                temp_path = temp_file.name

            try:
                success = rag.process_document(contents, file.filename)
                if success:
                    logger.info(f"Document {file.filename} processed successfully")
                    return JSONResponse(content={
                        "message": f"Document {file.filename} processed successfully",
                        "status": "success"
                    })
                else:
                    # Check if document was already processed
                    doc_id = rag._generate_document_id(contents)
                    if doc_id in rag.document_metadata:
                        logger.info(f"Document {file.filename} was already processed")
                        return JSONResponse(content={
                            "message": f"Document {file.filename} was already processed and is available for querying",
                            "status": "already_processed"
                        })
                    else:
                        logger.warning(f"Failed to process document: {file.filename}")
                        raise HTTPException(
                            status_code=400, 
                            detail="Failed to process document. The file might be too large or in an unsupported format."
                        )
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing document: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during document upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during document upload: {str(e)}"
        )

@app.get("/documents")
async def get_documents():
    try:
        logger.debug("Getting document list")
        documents = rag.list_documents()
        return JSONResponse(content={"documents": documents})
    except Exception as e:
        logger.error(f"Error getting document list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    try:
        logger.debug(f"Deleting document: {doc_id}")
        success = rag.delete_document(doc_id)
        if not success:
            logger.warning(f"Document not found: {doc_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        logger.info(f"Document {doc_id} deleted successfully")
        return JSONResponse(content={"message": f"Document {doc_id} deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cleanup")
async def cleanup():
    try:
        logger.debug("Starting cleanup")
        rag.cleanup()
        logger.info("Cleanup completed successfully")
        return JSONResponse(content={"message": "Cleanup completed successfully"})
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000) 