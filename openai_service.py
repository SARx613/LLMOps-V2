from abc import ABC
from core.models import *
from typing import List, Optional, Union, Dict, Any

class BaseLLMService(ABC):
    """
    Abstract base class for LLM service implementations.
    All provider-specific services must implement these methods.
    """
    
    provider: Provider
    # Get chat completion 
    @abstractmethod
    async def get_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        pass
    
    # Get embeddings 
    @abstractmethod
    async def get_embeddings(self, request: TextEmbeddingRequest) -> TextEmbeddingResponse:
        pass

    # List models 
    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        pass

    # Get model infos from provider 
    @abstractmethod
    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        pass

    # Check if the service is running right 
    @abstractmethod
    async def health_check(self) -> bool:
        pass
    
    # Convert a standardized request to the provider-specific format
    @abstractmethod
    def convert_request(self, request: Union[ChatCompletionRequest, TextEmbeddingRequest]) -> Dict[str, Any]:
        pass
    
    # Convert a standardized response request to the provider-specific format
    @abstractmethod
    def convert_response(self, response: Any, request_type: str) -> Union[ChatCompletionResponse, TextEmbeddingResponse]:
        pass

    # Count token 
    @abstractmethod
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        pass