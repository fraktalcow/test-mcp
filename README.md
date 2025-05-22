# AI Chat Interface

A modern web interface for interacting with various AI tools powered by OpenAI's GPT-4.

## Features

- Text Analysis
- Translation
- Text Summarization
- Text Classification
- Question Generation
- Keyword Extraction
- Code Generation
- Entity Extraction
- General Chat

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Application

Start the server:
```bash
python main.py
```

The application will be available at `http://localhost:8000`

## Usage

1. Open your browser and navigate to `http://localhost:8000`
2. Select a tool from the sidebar
3. Enter your message in the input field
4. Press Enter or click the send button

### Tool-specific Input Formats

- Translation: `text|target_language`
- Classification: `text|categories`
- Code Generation: `description|language`

## Development

The application uses:
- FastAPI for the backend
- HTMX for dynamic interactions
- Modern CSS for styling
- OpenAI's GPT-4 for AI capabilities 