from openai import AsyncOpenAI
import os
from typing import Dict, Any, List
import json
from datetime import datetime

class OpenAITools:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4"
        self.conversation_history = []

    async def analyze_text(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """Analyze text for various purposes (sentiment, key points, summary)"""
        try:
            prompt = f"Analyze this text for {analysis_type}:\n{text}"
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return {
                "analysis": response.choices[0].message.content,
                "type": analysis_type,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

    async def generate_code(self, description: str, language: str = "python") -> Dict[str, Any]:
        """Generate code based on description"""
        try:
            prompt = f"Generate {language} code for: {description}. Include comments and docstrings."
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            return {
                "code": response.choices[0].message.content,
                "language": language,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

    async def chat_with_context(self, message: str, context: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Chat with conversation history and context"""
        try:
            messages = []
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": message})
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            result = {
                "response": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }
            
            self.conversation_history.append({
                "user": message,
                "assistant": result["response"],
                "timestamp": result["timestamp"]
            })
            
            return result
        except Exception as e:
            return {"error": str(e)}

    async def extract_entities(self, text: str, entity_types: List[str] = None) -> Dict[str, Any]:
        """Extract named entities from text"""
        try:
            entity_types_str = ", ".join(entity_types) if entity_types else "all types"
            prompt = f"Extract {entity_types_str} entities from this text and return as JSON:\n{text}"
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            return {
                "entities": json.loads(response.choices[0].message.content),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}


    async def process_message(self, message: str, **kwargs) -> str:
        """Process a message using GPT-4"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": message}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000)
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing message: {str(e)}"

    async def get_context(self, key: str) -> str:
        """Get context information"""
        return f"Context for {key}"

    async def create_prompt(self, template: str) -> str:
        """Create a prompt template"""
        return f"Template: {template}" 