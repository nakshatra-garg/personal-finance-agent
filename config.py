"""Configuration module for LLM providers and app settings."""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMConfig:
    """Configuration for different LLM providers."""

    PROVIDERS = {
        "groq": {
            "name": "Groq",
            "api_key_env": "GROQ_API_KEY",
            "models": [
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ],
            "default_model": "llama-3.3-70b-versatile"
        },
        "openai": {
            "name": "OpenAI",
            "api_key_env": "OPENAI_API_KEY",
            "models": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo"
            ],
            "default_model": "gpt-4o-mini"
        },
        "azure": {
            "name": "Azure OpenAI",
            "api_key_env": "AZURE_OPENAI_API_KEY",
            "endpoint_env": "AZURE_OPENAI_ENDPOINT",
            "models": [
                "gpt-4o",
                "gpt-4",
                "gpt-35-turbo"
            ],
            "default_model": "gpt-4o"
        },
        "perplexity": {
            "name": "Perplexity",
            "api_key_env": "PERPLEXITY_API_KEY",
            "models": [
                "llama-3.1-sonar-large-128k-online",
                "llama-3.1-sonar-small-128k-online",
                "llama-3.1-sonar-large-128k-chat",
                "llama-3.1-sonar-small-128k-chat"
            ],
            "default_model": "llama-3.1-sonar-small-128k-chat"
        }
    }

    @staticmethod
    def get_api_key(provider: str, session_keys: Dict[str, str] = None) -> str:
        """Get API key for the specified provider from session state or env."""
        config = LLMConfig.PROVIDERS.get(provider)
        if not config:
            raise ValueError(f"Unknown provider: {provider}")

        # Check session state first, then fallback to environment
        if session_keys and provider in session_keys:
            return session_keys[provider]
        return os.getenv(config["api_key_env"], "")

    @staticmethod
    def get_endpoint(provider: str, session_endpoints: Dict[str, str] = None) -> str:
        """Get endpoint for providers that need it (Azure)."""
        config = LLMConfig.PROVIDERS.get(provider)
        if config and "endpoint_env" in config:
            # Check session state first
            if session_endpoints and f"{provider}_endpoint" in session_endpoints:
                return session_endpoints[f"{provider}_endpoint"]
            return os.getenv(config["endpoint_env"], "")
        return ""

    @staticmethod
    def get_available_providers(session_keys: Dict[str, str] = None) -> list:
        """Get list of providers with valid API keys from session or env."""
        available = []
        for provider, config in LLMConfig.PROVIDERS.items():
            # Check session state first
            if session_keys and provider in session_keys and session_keys[provider]:
                available.append(provider)
            elif os.getenv(config["api_key_env"]):
                available.append(provider)
        return available

    @staticmethod
    def get_models(provider: str) -> list:
        """Get available models for the specified provider."""
        config = LLMConfig.PROVIDERS.get(provider, {})
        return config.get("models", [])

    @staticmethod
    def get_default_model(provider: str) -> str:
        """Get default model for the specified provider."""
        config = LLMConfig.PROVIDERS.get(provider, {})
        return config.get("default_model", "")


class AppConfig:
    """General application configuration."""

    # Supported file types
    SUPPORTED_FILE_TYPES = ["pdf", "docx", "doc", "txt", "csv", "xlsx"]

    # Transaction categories
    EXPENSE_CATEGORIES = [
        "Housing", "Transportation", "Food & Dining", "Utilities",
        "Healthcare", "Entertainment", "Shopping", "Personal Care",
        "Education", "Insurance", "Debt Payments", "Savings",
        "Investments", "Travel", "Subscriptions", "Other"
    ]

    INCOME_CATEGORIES = [
        "Salary", "Business Income", "Investment Income",
        "Rental Income", "Freelance", "Gifts", "Other"
    ]

    # Default settings
    DEFAULT_CURRENCY = "USD"
    MAX_FILE_SIZE_MB = 10
