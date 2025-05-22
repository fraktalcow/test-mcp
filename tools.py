from typing import Dict, Any, List, Callable, Optional
import openai
from openai import OpenAI
from functools import wraps
import os
from dotenv import load_dotenv

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.descriptions: Dict[str, str] = {}

    def register(self, name: str, description: str):
        def decorator(func: Callable):
            self.tools[name] = func
            self.descriptions[name] = description
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

class OpenAITools:
    def __init__(self, api_key: str):
        load_dotenv()
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.registry = ToolRegistry()
        self.settings = {
            "maxTokens": 2000,
            "temperature": 0.7,
            "model": self.model
        }
        self._register_tools()

    def update_settings(self, settings: Dict[str, Any]):
        """Update the settings for the OpenAI client."""
        self.settings.update(settings)
        self.model = settings.get("model", self.model)

    def _create_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Create a completion with the given prompts."""
        response = self.client.chat.completions.create(
            model=self.settings["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.settings["temperature"],
            max_tokens=self.settings["maxTokens"]
        )
        return response.choices[0].message.content

    def _register_tools(self):
        """Register all available tools."""
        @self.registry.register("analyze", "Analyze text for sentiment, tone, and key points")
        def analyze_text(text: str) -> Dict[str, Any]:
            system_prompt = "Analyze the following text for sentiment, tone, and key points. Format your response with clear sections."
            return {"analysis": self._create_completion(system_prompt, text)}

        @self.registry.register("translate", "Translate text to target language")
        def translate_text(text: str, target_language: str = "English") -> Dict[str, str]:
            system_prompt = f"Translate the following text to {target_language}. Maintain the original formatting and style."
            return {"translation": self._create_completion(system_prompt, text)}

        @self.registry.register("summarize", "Generate a concise summary of text")
        def summarize_text(text: str) -> Dict[str, str]:
            system_prompt = "Provide a concise summary of the following text. Focus on key points and main ideas."
            return {"summary": self._create_completion(system_prompt, text)}

        @self.registry.register("classify", "Classify text into categories")
        def classify_text(text: str, categories: List[str] = None) -> Dict[str, str]:
            if not categories:
                categories = ["General", "Technical", "Business", "Creative"]
            categories_str = ", ".join(categories)
            system_prompt = f"Classify the following text into one of these categories: {categories_str}. Explain your reasoning."
            return {"classification": self._create_completion(system_prompt, text)}

        @self.registry.register("questions", "Generate questions about text")
        def generate_questions(text: str) -> Dict[str, List[str]]:
            system_prompt = "Generate 3-5 relevant questions about the following text. Make them thought-provoking and specific."
            questions = self._create_completion(system_prompt, text).split('\n')
            return {"questions": [q.strip() for q in questions if q.strip()]}

        @self.registry.register("keywords", "Extract key terms and phrases")
        def extract_keywords(text: str) -> Dict[str, List[str]]:
            system_prompt = "Extract key terms and phrases from the following text. Focus on important concepts and technical terms."
            keywords = self._create_completion(system_prompt, text).split(',')
            return {"keywords": [k.strip() for k in keywords if k.strip()]}

        @self.registry.register("code", "Generate code based on description")
        def generate_code(description: str, language: str = "Python") -> Dict[str, str]:
            system_prompt = f"""Generate {language} code based on the following description. 
            Include:
            1. Proper code formatting with markdown
            2. Comments explaining the code
            3. Error handling where appropriate
            4. Example usage if relevant"""
            return {"code": self._create_completion(system_prompt, description)}

        @self.registry.register("entities", "Extract named entities from text")
        def extract_entities(text: str) -> Dict[str, List[str]]:
            system_prompt = """Extract named entities (people, places, organizations) from the following text.
            Format each entity with its type (e.g., "Person: John Smith", "Organization: Acme Corp")."""
            entities = self._create_completion(system_prompt, text).split('\n')
            return {"entities": [e.strip() for e in entities if e.strip()]}

    def extract_command(self, text: str) -> Optional[str]:
        """Extract command from text if present."""
        if text.startswith('/'):
            command = text[1:].split()[0].lower()
            if command in self.registry.tools:
                return command
        return None

    async def execute_command(self, command: str, text: str) -> Dict[str, Any]:
        """Execute a command with the given text."""
        if command in self.registry.tools:
            return self.registry.tools[command](text)
        return {"error": f"Invalid command: {command}. Use /help to see available commands."}

    def get_available_tools(self) -> Dict[str, str]:
        """Get list of available tools and their descriptions."""
        return self.registry.descriptions 