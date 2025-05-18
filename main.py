import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tools import OpenAITools

# Load environment variables
load_dotenv()

# Initialize OpenAI client and tools
client = AsyncOpenAI(api_key=os.getenv("APIKEY"))
tools = OpenAITools(api_key=os.getenv("APIKEY"))

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

async def chat_loop():
    print("Welcome to the Terminal Chatbot!")
    print("Use /analyze <text>, /code <desc>, /entities <text> to invoke tools.")
    print("Type /exit to quit. Any other input is normal chat.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == '/exit':
            print("Goodbye!")
            break
        if user_input.startswith('/analyze '):
            args = user_input[len('/analyze '):]
            result = await tools.analyze_text(args)
            print(f"\nBot: Analysis Result:\n{result.get('analysis', result)}")
        elif user_input.startswith('/code '):
            args = user_input[len('/code '):]
            result = await tools.generate_code(args)
            print(f"\nBot: Generated Code:\n{result.get('code', result)}")
        elif user_input.startswith('/entities '):
            args = user_input[len('/entities '):]
            result = await tools.extract_entities(args)
            print(f"\nBot: Extracted Entities:\n{result.get('entities', result)}")
        else:
            response = await process_message(user_input)
            print(f"\nBot: {response}")

async def main():
    await chat_loop()

if __name__ == "__main__":
    asyncio.run(main()) 