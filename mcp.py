import os
from dotenv import load_dotenv
from typing import Dict, Any, Generator, Optional, List
from tools import OpenAITools
from openai import OpenAI
import json
import asyncio
from fastapi import WebSocket
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCP:
    def __init__(self):
        logger.debug("Initializing MCP system")
        try:
            load_dotenv()
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            self.client = OpenAI(api_key=self.api_key)
            self.model = os.getenv("OPENAI_MODEL", "gpt-4")
            self.tools = OpenAITools(api_key=self.api_key)
            self.temperature = 0.7
            self.max_tokens = 2000
            logger.info("MCP system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MCP system: {str(e)}")
            raise

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context for the prompt."""
        formatted = []
        for item in context:
            ref_id = item.get('metadata', {}).get('reference_id', 'N/A')
            source = item.get('metadata', {}).get('source', 'Unknown')
            page = item.get('metadata', {}).get('page', 'N/A')
            content = item.get('content', '')
            formatted.append(f"[{ref_id}] From {source} (Page {page}):\n{content}")
        return "\n\n".join(formatted)

    def _create_qa_prompt(self, question: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        """Create a prompt for Q&A."""
        base_prompt = """You are a helpful AI assistant. You can:
1. Answer general questions based on your knowledge
2. Use provided context to answer questions about specific documents

When using context:
- If the answer is in the context, cite sources using reference IDs in square brackets (e.g., [1.2])
- If the answer isn't in the context, provide a natural response based on your knowledge
- If no context is provided, answer based on your general knowledge
- Always maintain a natural conversation flow

Question: {question}"""

        if context and len(context) > 0:
            formatted_context = self._format_context(context)
            return f"{base_prompt}\n\nContext:\n{formatted_context}"
        return base_prompt.format(question=question)

    def _extract_references(self, text: str) -> List[str]:
        """Extract reference IDs from text."""
        pattern = r'\[([\d\.]+)\]'
        return re.findall(pattern, text)

    async def process_message_stream(self, message: str, context: Optional[List[Dict[str, Any]]] = None, websocket=None):
        """Process a message with streaming response."""
        try:
            logger.debug(f"Processing message stream: {message[:100]}...")
            prompt = self._create_qa_prompt(message, context)
            full_response = ""
            references = set()

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Provide clear, concise, and accurate responses. Maintain a natural conversation flow and cite sources when using document context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    if websocket:
                        await websocket.send_json({
                            "type": "stream",
                            "content": content
                        })

            # Extract references from the full response
            if context:
                refs = self._extract_references(full_response)
                references.update(refs)
                if websocket and references:
                    await websocket.send_json({
                        "type": "references",
                        "references": list(references)
                    })

            logger.info("Message stream processed successfully")

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            if websocket:
                await websocket.send_json({
                    "type": "error",
                    "content": error_msg
                })
            raise

    async def process_message(self, message: str, context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Process a message and return the response."""
        try:
            logger.debug(f"Processing message: {message[:100]}...")
            
            # Check if it's a tool command
            command = self.tools.extract_command(message)
            if command:
                logger.debug(f"Processing tool command: {command}")
                result = await self.tools.execute_command(command, message)
                return {"response": self.format_response(command, result)}

            prompt = self._create_qa_prompt(message, context)
            references = set()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Provide clear, concise, and accurate responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            result = {
                "response": response.choices[0].message.content
            }

            if context:
                refs = self._extract_references(result["response"])
                references.update(refs)
                result["references"] = list(references)

            logger.info("Message processed successfully")
            return result

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            return {
                "error": error_msg
            }

    def get_available_tools(self) -> Dict[str, str]:
        """Get list of available tools and their descriptions."""
        return self.tools.get_available_tools()

    async def process_with_tools(self, message: str, websocket: Optional[WebSocket] = None, context: Optional[List[Dict[str, str]]] = None) -> str:
        """Process message with available tools."""
        try:
            # Check if message contains a tool command
            command = self.tools.extract_command(message)
            if command:
                result = await self.tools.execute_command(command, message)
                response = self.format_response(command, result)
                
                if websocket:
                    await websocket.send_json({
                        "type": "tool_response",
                        "content": response
                    })
                return response
            
            # If no tool command, process as regular message
            if websocket:
                await self.process_message_stream(message, context, websocket)
                return ""
            else:
                result = self.process_message(message, context)
                return result["response"]

        except Exception as e:
            error_msg = f"Error processing message with tools: {str(e)}"
            if websocket:
                await websocket.send_json({
                    "type": "error",
                    "content": error_msg
                })
            return error_msg

    def format_response(self, command: str, result: Dict[str, Any]) -> str:
        """Format the response based on the command and result."""
        if "error" in result:
            return f"❌ **Error:** {result['error']}"

        if command == "analyze":
            return f"🔍 **Analysis:**\n{result['analysis']}"
        elif command == "translate":
            return f"🌐 **Translation:**\n{result['translation']}"
        elif command == "summarize":
            return f"📝 **Summary:**\n{result['summary']}"
        elif command == "classify":
            return f"🏷️ **Classification:**\n{result['classification']}"
        elif command == "questions":
            return "❓ **Questions:**\n" + "\n".join(f"• {q}" for q in result['questions'])
        elif command == "keywords":
            return "🔑 **Keywords:**\n" + ", ".join(result['keywords'])
        elif command == "code":
            return result['code']
        elif command == "entities":
            return "👥 **Entities:**\n" + "\n".join(f"• {e}" for e in result['entities'])
        else:
            return str(result) 