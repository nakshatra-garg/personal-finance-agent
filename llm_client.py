"""Unified LLM client supporting multiple providers."""

from typing import List, Dict, Any, Optional
import os
from groq import Groq
from openai import OpenAI, AzureOpenAI
from config import LLMConfig


class LLMClient:
    """Unified interface for multiple LLM providers."""

    def __init__(self, provider: str, model: Optional[str] = None):
        """Initialize the LLM client with specified provider."""
        self.provider = provider
        self.model = model or LLMConfig.get_default_model(provider)
        self.client = self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        api_key = LLMConfig.get_api_key(self.provider)

        if not api_key:
            raise ValueError(f"API key not found for provider: {self.provider}")

        if self.provider == "groq":
            return Groq(api_key=api_key)

        elif self.provider == "openai":
            return OpenAI(api_key=api_key)

        elif self.provider == "azure":
            endpoint = LLMConfig.get_endpoint(self.provider)
            if not endpoint:
                raise ValueError("Azure OpenAI endpoint not configured")
            return AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=endpoint
            )

        elif self.provider == "perplexity":
            return OpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai"
            )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Any:
        """
        Generate chat completion using the configured provider.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Chat completion response
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": stream
            }

            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**kwargs)
            return response

        except Exception as e:
            raise Exception(f"Error calling {self.provider} API: {str(e)}")

    def get_response_text(self, response: Any) -> str:
        """Extract text from the API response."""
        if hasattr(response, "choices") and len(response.choices) > 0:
            return response.choices[0].message.content
        return ""

    def stream_response(self, response: Any):
        """Stream response chunks."""
        for chunk in response:
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content


def get_llm_client(provider: str, model: Optional[str] = None) -> LLMClient:
    """Factory function to create LLM client."""
    return LLMClient(provider, model)
