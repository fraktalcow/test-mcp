from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, CreateMessageResult
import asyncio
from tools import OpenAITools
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

class MCPServer:
    def __init__(self):
        load_dotenv()
        self.mcp = FastMCP("AgentMCP")
        self.tools = OpenAITools(api_key=os.getenv("APIKEY"))
        self._setup_tools()

    def _setup_tools(self):
        @self.mcp.tool()
        async def analyze_text(text: str, analysis_type: str = "general") -> Dict[str, Any]:
            return await self.tools.analyze_text(text, analysis_type)

        @self.mcp.tool()
        async def generate_code(description: str, language: str = "python") -> Dict[str, Any]:
            return await self.tools.generate_code(description, language)

        @self.mcp.tool()
        async def chat_with_context(message: str, context: List[Dict[str, str]] = None) -> Dict[str, Any]:
            return await self.tools.chat_with_context(message, context)

        @self.mcp.tool()
        async def extract_entities(text: str, entity_types: List[str] = None) -> Dict[str, Any]:
            return await self.tools.extract_entities(text, entity_types)


    async def start(self):
        await self.mcp.start()

async def main():
    server = MCPServer()
    await server.start()

if __name__ == "__main__":
    asyncio.run(main()) 