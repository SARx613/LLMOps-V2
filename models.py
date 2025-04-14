from enum import Enum
from typing import List, Optional
from pydantic import BaseModel

class Provider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    groq = "groq"
    ollama = "ollama"

class Message(BaseModel):
    role: str 
    content: str

class ChatCompletionRequest(BaseModel):
    provider: Provider
    model: str
    messages: List[Message]

class ChatCompletionResponse(BaseModel):
    response: str
    model: str
    usage: Optional[dict] = None

class TextEmbeddingRequest(BaseModel):
    provider: Provider
    model: str
    input: str

class TextEmbeddingResponse(BaseModel):
    embedding: List[float]
    model: str
    dimensions: Optional[int] = None

class ModelInfo(BaseModel):
    provider: Provider
    model: str
    description: Optional[str] = None
    context_window: Optional[int] = None
    supports_chat: bool = True
    supports_embeddings: bool = False
