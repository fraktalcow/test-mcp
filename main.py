from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, CreateMessageResult
import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("APIKEY"))

# Initialize MCP server
mcp = FastMCP("AgentMCP")

@mcp.tool()
async def process_message(message: str) -> str:
    """Process a message using GPT-4"""
    try:
        response = await client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4"),
            messages=[{"role": "user", "content": message}],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing message: {str(e)}"

@mcp.resource("context://{key}")
async def get_context(key: str) -> str:
    """Get context information"""
    return f"Context for {key}"

@mcp.prompt()
async def create_prompt(template: str) -> str:
    """Create a prompt template"""
    return f"Template: {template}"

async def main():
    # Start the MCP server
    await mcp.start()

if __name__ == "__main__":
    asyncio.run(main()) 