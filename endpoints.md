# API Endpoints Documentation

## WebSocket Endpoints

### `/ws`
WebSocket endpoint for streaming responses with RAG integration.

**Request Format:**
```json
{
    "message": "string",
    "useRag": boolean
}
```

**Response Format:**
```json
{
    "type": "stream|error|references",
    "content": "string",
    "references": ["string"] // Optional, only for RAG responses
}
```

## HTTP Endpoints

### `POST /send_message`
Send a message and get a response with optional RAG integration.

**Request Body:**
```json
{
    "message": "string",
    "settings": {
        // Optional settings for the LLM
    },
    "useRag": boolean,
    "documents": ["string"] // Optional list of document references
}
```

**Response:**
```json
{
    "response": "string",
    "references": ["string"], // Optional, only for RAG responses
    "error": "string" // Only present if there was an error
}
```

### `POST /upload_document`
Upload and process a document for RAG.

**Request:**
- Content-Type: multipart/form-data
- File parameter name: file
- Supported file types: .pdf, .txt, .docx, .doc, .md

**Response:**
```json
{
    "message": "string",
    "status": "success|error"
}
```

### `GET /`
Serves the main web interface.

**Response:**
- HTML file (static/index.html)

## Environment Variables

The following environment variables are required:

- `OPENAI_API_KEY`: Your OpenAI API key
- `LOG_DIR`: (Optional) Directory for log files
- `LOG_MAX_BYTES`: (Optional) Maximum size of log files in bytes
- `LOG_BACKUP_COUNT`: (Optional) Number of log file backups to keep
- `VERBOSE_DEBUG`: (Optional) Enable verbose debug logging

## RAG Query Modes

When using RAG functionality, the following query modes are available:

- `naive`: Basic retrieval without additional processing
- `local`: Local context-based retrieval
- `global`: Global context-based retrieval
- `hybrid`: Combination of local and global retrieval 