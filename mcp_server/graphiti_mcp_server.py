#!/usr/bin/env python3
"""
Graphiti MCP Server - Exposes Graphiti functionality through the Model Context Protocol (MCP)

Supports multiple LLM providers:
- OpenAI: Set OPENAI_API_KEY, use models like gpt-4.1-mini, gpt-4o
- Google Gemini: Set GOOGLE_API_KEY, use models like gemini-2.5-flash, gemini-2.5-pro
- Azure OpenAI: Set AZURE_OPENAI_ENDPOINT and related Azure configuration

For Gemini models:
- Set GOOGLE_API_KEY environment variable (get from https://aistudio.google.com/app/apikey)
- Optionally set GEMINI_THINKING_ENABLED=true for Gemini 2.5+ models
- Use model names like: gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-flash

Example usage:
  uv run graphiti_mcp_server.py --model gemini-2.5-flash --transport sse
"""

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypedDict, cast

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

# Import GeminiEmbedder with proper error handling
try:
    from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
    GEMINI_EMBEDDER_AVAILABLE = True
except ImportError:
    GEMINI_EMBEDDER_AVAILABLE = False
    GeminiEmbedder = None
    GeminiEmbedderConfig = None

# Import reranker clients with proper error handling
try:
    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
    GEMINI_RERANKER_AVAILABLE = True
except ImportError:
    GEMINI_RERANKER_AVAILABLE = False
    GeminiRerankerClient = None

try:
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    OPENAI_RERANKER_AVAILABLE = True
except ImportError:
    OPENAI_RERANKER_AVAILABLE = False
    OpenAIRerankerClient = None

from graphiti_core.cross_encoder.client import CrossEncoderClient


class NoOpCrossEncoderClient(CrossEncoderClient):
    """
    A no-op cross encoder client that returns passages in their original order.
    Used when no proper cross encoder is available.
    """
    
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """Return passages in original order with equal scores."""
        return [(passage, 1.0) for passage in passages]
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import RateLimitError, RefusalError, EmptyResponseError
from graphiti_core.llm_client.openai_client import OpenAIClient

# Import GeminiClient with proper error handling
try:
    from graphiti_core.llm_client.gemini_client import GeminiClient
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    GeminiClient = None
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

load_dotenv()


DEFAULT_LLM_MODEL = 'gpt-4.1-mini'
SMALL_LLM_MODEL = 'gpt-4.1-nano'
DEFAULT_EMBEDDER_MODEL = 'text-embedding-3-small'

# Semaphore limit for concurrent Graphiti operations.
# Decrease this if you're experiencing 429 rate limit errors from your LLM provider.
# Increase if you have high rate limits.
SEMAPHORE_LIMIT = int(os.getenv('SEMAPHORE_LIMIT', 10))


class Requirement(BaseModel):
    """A Requirement represents a specific need, feature, or functionality that a product or service must fulfill.

    Always ensure an edge is created between the requirement and the project it belongs to, and clearly indicate on the
    edge that the requirement is a requirement.

    Instructions for identifying and extracting requirements:
    1. Look for explicit statements of needs or necessities ("We need X", "X is required", "X must have Y")
    2. Identify functional specifications that describe what the system should do
    3. Pay attention to non-functional requirements like performance, security, or usability criteria
    4. Extract constraints or limitations that must be adhered to
    5. Focus on clear, specific, and measurable requirements rather than vague wishes
    6. Capture the priority or importance if mentioned ("critical", "high priority", etc.)
    7. Include any dependencies between requirements when explicitly stated
    8. Preserve the original intent and scope of the requirement
    9. Categorize requirements appropriately based on their domain or function
    """

    project_name: str = Field(
        ...,
        description='The name of the project to which the requirement belongs.',
    )
    description: str = Field(
        ...,
        description='Description of the requirement. Only use information mentioned in the context to write this description.',
    )


class Preference(BaseModel):
    """A Preference represents a user's expressed like, dislike, or preference for something.

    Instructions for identifying and extracting preferences:
    1. Look for explicit statements of preference such as "I like/love/enjoy/prefer X" or "I don't like/hate/dislike X"
    2. Pay attention to comparative statements ("I prefer X over Y")
    3. Consider the emotional tone when users mention certain topics
    4. Extract only preferences that are clearly expressed, not assumptions
    5. Categorize the preference appropriately based on its domain (food, music, brands, etc.)
    6. Include relevant qualifiers (e.g., "likes spicy food" rather than just "likes food")
    7. Only extract preferences directly stated by the user, not preferences of others they mention
    8. Provide a concise but specific description that captures the nature of the preference
    """

    category: str = Field(
        ...,
        description="The category of the preference. (e.g., 'Brands', 'Food', 'Music')",
    )
    description: str = Field(
        ...,
        description='Brief description of the preference. Only use information mentioned in the context to write this description.',
    )


class Procedure(BaseModel):
    """A Procedure informing the agent what actions to take or how to perform in certain scenarios. Procedures are typically composed of several steps.

    Instructions for identifying and extracting procedures:
    1. Look for sequential instructions or steps ("First do X, then do Y")
    2. Identify explicit directives or commands ("Always do X when Y happens")
    3. Pay attention to conditional statements ("If X occurs, then do Y")
    4. Extract procedures that have clear beginning and end points
    5. Focus on actionable instructions rather than general information
    6. Preserve the original sequence and dependencies between steps
    7. Include any specified conditions or triggers for the procedure
    8. Capture any stated purpose or goal of the procedure
    9. Summarize complex procedures while maintaining critical details
    """

    description: str = Field(
        ...,
        description='Brief description of the procedure. Only use information mentioned in the context to write this description.',
    )


ENTITY_TYPES: dict[str, BaseModel] = {
    'Requirement': Requirement,  # type: ignore
    'Preference': Preference,  # type: ignore
    'Procedure': Procedure,  # type: ignore
}


# Type definitions for API responses
class ErrorResponse(TypedDict):
    error: str


class SuccessResponse(TypedDict):
    message: str


class NodeResult(TypedDict):
    uuid: str
    name: str
    summary: str
    labels: list[str]
    group_id: str
    created_at: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(TypedDict):
    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(TypedDict):
    status: str
    message: str


def create_azure_credential_token_provider() -> Callable[[], str]:
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, 'https://cognitiveservices.azure.com/.default'
    )
    return token_provider


# Server configuration classes
# The configuration system has a hierarchy:
# - GraphitiConfig is the top-level configuration
#   - LLMConfig handles all OpenAI/LLM related settings
#   - EmbedderConfig manages embedding settings
#   - Neo4jConfig manages database connection details
#   - Various other settings like group_id and feature flags
# Configuration values are loaded from:
# 1. Default values in the class definitions
# 2. Environment variables (loaded via load_dotenv())
# 3. Command line arguments (which override environment variables)
class GraphitiLLMConfig(BaseModel):
    """Configuration for the LLM client.

    Centralizes all LLM-specific configuration parameters including API keys and model selection.
    """

    api_key: str | None = None
    model: str = DEFAULT_LLM_MODEL
    small_model: str = SMALL_LLM_MODEL
    temperature: float = 0.0
    azure_openai_endpoint: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_use_managed_identity: bool = False
    # Gemini-specific configuration
    google_api_key: str | None = None
    gemini_thinking_enabled: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiLLMConfig':
        """Create LLM configuration from environment variables."""
        # Get model from environment, or use default if not set or empty
        model_env = os.environ.get('MODEL_NAME', '')
        
        # Get API keys to determine default model
        google_api_key = os.environ.get('GOOGLE_API_KEY', None)
        openai_api_key = os.environ.get('OPENAI_API_KEY', None)
        azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', None)
        
        # Determine default model based on available API keys if no model specified
        if model_env.strip():
            model = model_env.strip()
        else:
            # Default model selection based on available API keys and priority
            # Requirements 1.4, 1.5, 2.4, 2.5: Proper default model handling
            if azure_endpoint:
                # Azure OpenAI takes highest priority when configured
                model = DEFAULT_LLM_MODEL
                logger.debug(f'No model specified, using default for Azure OpenAI: {model}')
            elif google_api_key:
                # Google API key present, default to Gemini (gemini-2.5-flash)
                # This satisfies requirement 1.4: default to "gemini-2.5-flash" when GOOGLE_API_KEY is present
                model = 'gemini-2.5-flash'
                if openai_api_key:
                    logger.info(f'Both GOOGLE_API_KEY and OPENAI_API_KEY detected, defaulting to Gemini: {model}')
                else:
                    logger.info(f'GOOGLE_API_KEY detected, defaulting to Gemini: {model}')
            elif openai_api_key:
                # Only OpenAI API key available, use OpenAI default (backward compatibility)
                # This satisfies requirement 1.5: existing OpenAI default behavior remains unchanged
                model = DEFAULT_LLM_MODEL
                logger.debug(f'OPENAI_API_KEY detected, using OpenAI default: {model}')
            else:
                # No API keys available, use OpenAI default for backward compatibility
                # This ensures existing behavior remains unchanged when no API keys are present
                model = DEFAULT_LLM_MODEL
                if not model_env:
                    logger.debug(f'No MODEL_NAME or API keys set, using default: {model}')
                else:
                    logger.warning(f'Empty MODEL_NAME environment variable, using default: {model}')

        # Get small_model from environment, or use default if not set or empty
        small_model_env = os.environ.get('SMALL_MODEL_NAME', '')
        small_model = small_model_env if small_model_env.strip() else SMALL_LLM_MODEL

        azure_openai_api_version = os.environ.get('AZURE_OPENAI_API_VERSION', None)
        azure_openai_deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', None)
        azure_openai_use_managed_identity = (
            os.environ.get('AZURE_OPENAI_USE_MANAGED_IDENTITY', 'false').lower() == 'true'
        )

        # Get Gemini-specific configuration
        gemini_thinking_enabled = (
            os.environ.get('GEMINI_THINKING_ENABLED', 'false').lower() == 'true'
        )

        if azure_endpoint is None:
            # Setup for OpenAI or Gemini API
            return cls(
                api_key=openai_api_key,
                model=model,
                small_model=small_model,
                temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
                google_api_key=google_api_key,
                gemini_thinking_enabled=gemini_thinking_enabled,
            )
        else:
            # Setup for Azure OpenAI API
            # Log if empty deployment name was provided
            if azure_openai_deployment_name is None:
                logger.error('AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set')
                raise ValueError('AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set')
            
            if not azure_openai_use_managed_identity:
                # api key
                api_key = openai_api_key
            else:
                # Managed identity
                api_key = None

            return cls(
                azure_openai_use_managed_identity=azure_openai_use_managed_identity,
                azure_openai_endpoint=azure_endpoint,
                api_key=api_key,
                azure_openai_api_version=azure_openai_api_version,
                azure_openai_deployment_name=azure_openai_deployment_name,
                model=model,
                small_model=small_model,
                temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
                google_api_key=google_api_key,
                gemini_thinking_enabled=gemini_thinking_enabled,
            )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiLLMConfig':
        """Create LLM configuration from CLI arguments, falling back to environment variables.
        
        CLI arguments properly override environment variables for Gemini configuration.
        Validates model names to ensure they are properly handled for all providers.
        """
        # Start with environment-based config
        config = cls.from_env()

        # CLI arguments override environment variables when provided
        if hasattr(args, 'model') and args.model:
            # Only use CLI model if it's not empty
            if args.model.strip():
                # Validate model name format and provide helpful feedback
                model_name = args.model.strip()
                config.model = model_name
                
                # Log model selection with provider detection
                try:
                    # Temporarily set model to detect provider
                    temp_config = cls(
                        model=model_name,
                        api_key=config.api_key,
                        google_api_key=config.google_api_key,
                        azure_openai_endpoint=config.azure_openai_endpoint
                    )
                    provider = temp_config._detect_provider()
                    logger.info(f'CLI model "{model_name}" detected as {provider} provider')
                    
                    # Validate provider-specific model naming conventions
                    if provider == 'gemini' and not model_name.startswith('gemini-'):
                        logger.warning(
                            f'Model "{model_name}" detected as Gemini provider but does not follow '
                            'Gemini naming convention (gemini-X.X-model). This may cause issues.'
                        )
                    elif provider == 'openai' and not model_name.startswith('gpt-'):
                        logger.warning(
                            f'Model "{model_name}" detected as OpenAI provider but does not follow '
                            'OpenAI naming convention (gpt-X). This may cause issues.'
                        )
                        
                except Exception as e:
                    logger.warning(f'Could not validate model "{model_name}": {e}')
                    
            else:
                # Log that empty model was provided and default is used
                logger.warning(f'Empty model name provided, using default: {DEFAULT_LLM_MODEL}')

        if hasattr(args, 'small_model') and args.small_model:
            if args.small_model.strip():
                small_model_name = args.small_model.strip()
                config.small_model = small_model_name
                logger.info(f'CLI small_model set to: {small_model_name}')
            else:
                logger.warning(f'Empty small_model name provided, using default: {SMALL_LLM_MODEL}')

        if hasattr(args, 'temperature') and args.temperature is not None:
            # Validate temperature range
            if 0.0 <= args.temperature <= 2.0:
                config.temperature = args.temperature
                logger.info(f'CLI temperature set to: {args.temperature}')
            else:
                logger.warning(
                    f'Temperature {args.temperature} is outside valid range (0.0-2.0). '
                    f'Using default: {config.temperature}'
                )

        # Validate final configuration for consistency
        config._validate_configuration_consistency()

        return config

    def _detect_provider(self) -> str:
        """Detect which LLM provider to use based on model name patterns and available API keys.
        
        Provider detection follows this priority order:
        1. Azure OpenAI (when azure_openai_endpoint is configured)
        2. Model name patterns (gemini-* → gemini, gpt-* → openai)
        3. API key-based fallback (google_api_key → gemini, api_key → openai)
        
        This method works with the updated DEFAULT_LLM_MODEL handling to ensure proper
        provider detection for both explicit model names and default model scenarios.
        
        Returns:
            Provider name: 'azure_openai', 'gemini', or 'openai'
        """
        # Azure OpenAI has highest priority when configured
        if self.azure_openai_endpoint is not None:
            return 'azure_openai'
        
        # Model name-based detection takes precedence over API key fallback
        if self.model and self.model.startswith('gemini-'):
            return 'gemini'
        elif self.model and self.model.startswith('gpt-'):
            # Handle the case where default OpenAI model is used but only Google API key is available
            # This can happen if someone manually sets MODEL_NAME to a gpt- model but only has GOOGLE_API_KEY
            if self.model == DEFAULT_LLM_MODEL and self.google_api_key and not self.api_key:
                logger.warning(
                    f'Default OpenAI model "{DEFAULT_LLM_MODEL}" specified but only GOOGLE_API_KEY is available. '
                    'Consider setting MODEL_NAME to a Gemini model (e.g., gemini-2.5-flash) or providing OPENAI_API_KEY.'
                )
                return 'gemini'
            return 'openai'
        
        # API key-based fallback for models without specific patterns
        # This handles cases where model names don't follow standard patterns
        # Requirements 2.4, 2.5: Fallback scenarios with various API key combinations
        if self.google_api_key:
            return 'gemini'
        elif self.api_key:
            return 'openai'
        
        raise ValueError(
            'No valid LLM provider configuration found. '
            'Please set OPENAI_API_KEY, GOOGLE_API_KEY, or configure Azure OpenAI.'
        )

    def _validate_configuration_consistency(self) -> None:
        """Validate configuration consistency across all providers.
        
        Ensures that model names are properly handled for all providers and that
        required API keys are available for the detected provider.
        """
        try:
            provider = self._detect_provider()
            
            # Validate provider-specific requirements
            if provider == 'gemini':
                if not self.google_api_key:
                    logger.warning(
                        f'Model "{self.model}" requires GOOGLE_API_KEY but it is not set. '
                        'The server may fail to start. Get your API key from: https://aistudio.google.com/app/apikey'
                    )
                elif not GEMINI_AVAILABLE:
                    logger.warning(
                        f'Model "{self.model}" requires google-genai dependency but it is not available. '
                        'Install with: pip install graphiti-core[google-genai]'
                    )
                    
            elif provider == 'openai':
                if not self.api_key:
                    logger.warning(
                        f'Model "{self.model}" requires OPENAI_API_KEY but it is not set. '
                        'The server may fail to start.'
                    )
                    
            elif provider == 'azure_openai':
                if not self.azure_openai_use_managed_identity and not self.api_key:
                    logger.warning(
                        'Azure OpenAI configuration detected but OPENAI_API_KEY is not set '
                        'and managed identity is not enabled. The server may fail to start.'
                    )
                    
            # Log successful configuration validation
            logger.info(f'Configuration validated for {provider} provider with model: {self.model}')
            
        except ValueError as e:
            # Log configuration issues but don't fail - let the actual client creation handle errors
            logger.warning(f'Configuration validation warning: {e}')

    def _validate_gemini_config(self) -> None:
        """Validate Gemini-specific configuration.
        
        Performs comprehensive validation of Gemini configuration including:
        - Dependency availability check
        - API key validation
        - Model compatibility checks
        - Thinking configuration validation
        
        Raises:
            ImportError: When google-genai dependency is missing
            ValueError: When required configuration is missing or invalid
        """
        # Check if google-genai dependency is available
        if not GEMINI_AVAILABLE:
            raise ImportError(
                'google-genai is required for Gemini models. '
                'Install the required dependency with: pip install graphiti-core[google-genai] '
                'or pip install google-genai'
            )
        
        # Validate API key presence and format
        if not self.google_api_key:
            raise ValueError(
                'GOOGLE_API_KEY environment variable must be set when using Gemini models. '
                'Get your API key from: https://aistudio.google.com/app/apikey'
            )
        
        # Basic API key format validation (Google API keys typically start with 'AI')
        if not self.google_api_key.strip():
            raise ValueError(
                'GOOGLE_API_KEY cannot be empty. '
                'Please provide a valid Google API key from: https://aistudio.google.com/app/apikey'
            )
        
        # Validate model name format for Gemini
        if self.model and not self.model.startswith('gemini-'):
            logger.warning(
                f'Model name "{self.model}" does not follow Gemini naming convention. '
                'Expected format: gemini-X.X-model (e.g., gemini-2.5-flash)'
            )
        
        # Validate thinking configuration compatibility
        if self.gemini_thinking_enabled and self.model:
            if not self.model.startswith('gemini-2.5'):
                logger.warning(
                    f'Thinking configuration is enabled but model "{self.model}" may not support it. '
                    'Thinking is only supported on Gemini 2.5+ models. '
                    'The configuration will be ignored for unsupported models.'
                )
        
        # Log successful validation
        logger.info(f'Gemini configuration validated successfully for model: {self.model}')

    def _create_gemini_client(self) -> LLMClient:
        """Create and configure Gemini client with proper configuration mapping and error handling.
        
        Returns:
            GeminiClient instance configured with the current settings
            
        Raises:
            ImportError: When google-genai dependency is missing or incompatible
            ValueError: When configuration is invalid
            Exception: When client creation fails for other reasons
        """
        try:
            # Validate configuration first
            self._validate_gemini_config()
            
            # Create LLMConfig with mapped parameters
            llm_config = LLMConfig(
                api_key=self.google_api_key,
                model=self.model,
                small_model=self.small_model,
                temperature=self.temperature,
            )
            
            # Handle thinking configuration for Gemini 2.5+ models
            thinking_config = None
            if self.gemini_thinking_enabled and self.model:
                # Check if model supports thinking (Gemini 2.5+ models)
                if self.model.startswith('gemini-2.5'):
                    try:
                        # Import thinking config from google.genai
                        from google.genai import types
                        thinking_config = types.ThinkingConfig()
                        logger.info(f'Thinking configuration enabled for model: {self.model}')
                    except ImportError as import_err:
                        logger.warning(
                            'google-genai package does not support thinking configuration. '
                            'Please update to a newer version: pip install --upgrade google-genai'
                        )
                        # Continue without thinking config rather than failing
                    except AttributeError as attr_err:
                        logger.warning(
                            f'ThinkingConfig not available in current google-genai version: {attr_err}. '
                            'Please update to a newer version: pip install --upgrade google-genai'
                        )
                    except Exception as e:
                        logger.warning(
                            f'Failed to create thinking configuration: {e}. '
                            'Continuing without thinking configuration.'
                        )
                else:
                    logger.warning(
                        f'Thinking configuration requested but model {self.model} does not support it. '
                        'Only Gemini 2.5+ models support thinking. Continuing without thinking configuration.'
                    )
            
            # Create and return GeminiClient with comprehensive error handling
            try:
                # Try creating client with thinking_config first
                if thinking_config is not None:
                    try:
                        client = GeminiClient(
                            config=llm_config,
                            thinking_config=thinking_config,
                        )
                        logger.info(f'Successfully created Gemini client for model: {self.model} with thinking enabled')
                        return client
                    except TypeError as e:
                        if 'thinking_config' in str(e):
                            logger.warning(
                                f'Current google-genai version does not support thinking_config. '
                                f'Creating client without thinking configuration. '
                                f'To enable thinking, update google-genai: pip install --upgrade google-genai'
                            )
                            # Fall through to create client without thinking_config
                        else:
                            raise
                
                # Create client without thinking_config
                client = GeminiClient(
                    config=llm_config,
                )
                logger.info(f'Successfully created Gemini client for model: {self.model}')
                return client
                
            except Exception as client_err:
                # Handle specific Gemini client creation errors
                error_msg = str(client_err).lower()
                
                if 'api key' in error_msg or 'authentication' in error_msg:
                    raise ValueError(
                        f'Invalid Google API key. Please check your GOOGLE_API_KEY environment variable. '
                        f'Get a valid API key from: https://aistudio.google.com/app/apikey. '
                        f'Original error: {client_err}'
                    ) from client_err
                elif 'model' in error_msg and 'not found' in error_msg:
                    raise ValueError(
                        f'Gemini model "{self.model}" not found or not accessible. '
                        f'Please check the model name and your API key permissions. '
                        f'Available models include: gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-flash. '
                        f'Original error: {client_err}'
                    ) from client_err
                elif 'quota' in error_msg or 'rate limit' in error_msg:
                    raise ValueError(
                        f'Google API quota exceeded or rate limit hit. '
                        f'Please check your API usage and billing settings. '
                        f'Original error: {client_err}'
                    ) from client_err
                else:
                    # Re-raise with additional context
                    raise ValueError(
                        f'Failed to create Gemini client: {client_err}. '
                        f'Please check your GOOGLE_API_KEY and model configuration.'
                    ) from client_err
                    
        except (ImportError, ValueError):
            # Re-raise validation and import errors as-is
            raise
        except Exception as e:
            # Handle any other unexpected errors
            raise ValueError(
                f'Unexpected error while creating Gemini client: {e}. '
                f'Please check your configuration and try again.'
            ) from e

    def create_client(self) -> LLMClient:
        """Create an LLM client based on this configuration.

        Uses provider detection logic to determine which client to create based on
        model name patterns and available API keys. Supports Azure OpenAI, Gemini,
        and OpenAI providers with proper error handling.

        Returns:
            LLMClient instance

        Raises:
            ValueError: When no valid provider configuration is found or required API keys are missing
            ImportError: When required dependencies are missing (e.g., google-genai for Gemini)
        """
        try:
            # Detect provider based on configuration
            provider = self._detect_provider()
            
            if provider == 'azure_openai':
                # Azure OpenAI API setup - highest priority when configured
                if self.azure_openai_use_managed_identity:
                    # Use managed identity for authentication
                    token_provider = create_azure_credential_token_provider()
                    return AzureOpenAILLMClient(
                        azure_client=AsyncAzureOpenAI(
                            azure_endpoint=self.azure_openai_endpoint,
                            azure_deployment=self.azure_openai_deployment_name,
                            api_version=self.azure_openai_api_version,
                            azure_ad_token_provider=token_provider,
                        ),
                        config=LLMConfig(
                            api_key=self.api_key,
                            model=self.model,
                            small_model=self.small_model,
                            temperature=self.temperature,
                        ),
                    )
                elif self.api_key:
                    # Use API key for authentication
                    return AzureOpenAILLMClient(
                        azure_client=AsyncAzureOpenAI(
                            azure_endpoint=self.azure_openai_endpoint,
                            azure_deployment=self.azure_openai_deployment_name,
                            api_version=self.azure_openai_api_version,
                            api_key=self.api_key,
                        ),
                        config=LLMConfig(
                            api_key=self.api_key,
                            model=self.model,
                            small_model=self.small_model,
                            temperature=self.temperature,
                        ),
                    )
                else:
                    raise ValueError('OPENAI_API_KEY must be set when using Azure OpenAI API')
            
            elif provider == 'gemini':
                # Create Gemini client with proper error handling
                return self._create_gemini_client()
            
            elif provider == 'openai':
                # Standard OpenAI API setup
                if not self.api_key:
                    raise ValueError('OPENAI_API_KEY must be set when using OpenAI API')

                llm_client_config = LLMConfig(
                    api_key=self.api_key, 
                    model=self.model, 
                    small_model=self.small_model,
                    temperature=self.temperature
                )

                return OpenAIClient(config=llm_client_config)
            
            else:
                raise ValueError(f'Unknown provider: {provider}')
                
        except ImportError as e:
            # Handle missing dependencies with clear error messages
            error_msg = str(e).lower()
            if 'google-genai' in error_msg or 'gemini' in error_msg:
                raise ImportError(
                    f'Failed to create Gemini client due to missing dependency: {e}. '
                    'Install the required dependency with one of the following commands:\n'
                    '  pip install graphiti-core[google-genai]\n'
                    '  pip install google-genai\n'
                    'Then restart the MCP server.'
                ) from e
            else:
                # Re-raise other import errors with context
                provider = 'unknown'
                try:
                    provider = self._detect_provider()
                except Exception:
                    pass
                raise ImportError(
                    f'Failed to import required dependencies for {provider} provider: {e}'
                ) from e
        except ValueError as e:
            # Re-raise ValueError with provider context if not already included
            error_msg = str(e)
            if 'provider' not in error_msg.lower():
                try:
                    provider = self._detect_provider()
                    raise ValueError(f'Configuration error for {provider} provider: {e}') from e
                except Exception:
                    pass
            raise
        except Exception as e:
            # Provide context for other configuration errors
            provider = 'unknown'
            try:
                provider = self._detect_provider()
            except Exception:
                pass
            
            # Check for common error patterns and provide helpful messages
            error_msg = str(e).lower()
            if 'safety' in error_msg or 'blocked' in error_msg:
                raise ValueError(
                    f'Content safety error from {provider} provider: {e}. '
                    'This may be due to content being blocked by safety filters. '
                    'Try rephrasing your request or check the provider\'s content policy.'
                ) from e
            elif 'rate limit' in error_msg or 'quota' in error_msg or '429' in error_msg:
                raise ValueError(
                    f'Rate limit or quota exceeded for {provider} provider: {e}. '
                    'Please check your API usage limits and billing settings.'
                ) from e
            elif 'network' in error_msg or 'connection' in error_msg or 'timeout' in error_msg:
                raise ValueError(
                    f'Network connectivity error with {provider} provider: {e}. '
                    'Please check your internet connection and try again.'
                ) from e
            else:
                raise ValueError(
                    f'Failed to create LLM client for {provider} provider: {e}. '
                    'Please check your configuration and API credentials.'
                ) from e


class GraphitiEmbedderConfig(BaseModel):
    """Configuration for the embedder client.

    Centralizes all embedding-related configuration parameters.
    """

    model: str = DEFAULT_EMBEDDER_MODEL
    api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_use_managed_identity: bool = False
    # Gemini-specific configuration
    google_api_key: str | None = None

    @classmethod
    def from_env(cls) -> 'GraphitiEmbedderConfig':
        """Create embedder configuration from environment variables."""

        # Get model from environment, or use default if not set or empty
        model_env = os.environ.get('EMBEDDER_MODEL_NAME', '')
        google_api_key = os.environ.get('GOOGLE_API_KEY', None)
        openai_api_key = os.environ.get('OPENAI_API_KEY', None)
        
        # Determine default embedder model based on available API keys
        if model_env.strip():
            model = model_env.strip()
        else:
            # If Google API key is available, default to Gemini embeddings
            if google_api_key:
                model = 'text-embedding-004'  # Gemini embedding model
                logger.info('GOOGLE_API_KEY detected, defaulting to Gemini embeddings: text-embedding-004')
            else:
                model = DEFAULT_EMBEDDER_MODEL

        azure_openai_endpoint = os.environ.get('AZURE_OPENAI_EMBEDDING_ENDPOINT', None)
        azure_openai_api_version = os.environ.get('AZURE_OPENAI_EMBEDDING_API_VERSION', None)
        azure_openai_deployment_name = os.environ.get(
            'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', None
        )
        azure_openai_use_managed_identity = (
            os.environ.get('AZURE_OPENAI_USE_MANAGED_IDENTITY', 'false').lower() == 'true'
        )
        if azure_openai_endpoint is not None:
            # Setup for Azure OpenAI API
            # Log if empty deployment name was provided
            azure_openai_deployment_name = os.environ.get(
                'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', None
            )
            if azure_openai_deployment_name is None:
                logger.error('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable not set')

                raise ValueError(
                    'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable not set'
                )

            if not azure_openai_use_managed_identity:
                # api key
                api_key = os.environ.get('AZURE_OPENAI_EMBEDDING_API_KEY', None) or os.environ.get(
                    'OPENAI_API_KEY', None
                )
            else:
                # Managed identity
                api_key = None

            return cls(
                azure_openai_use_managed_identity=azure_openai_use_managed_identity,
                azure_openai_endpoint=azure_openai_endpoint,
                api_key=api_key,
                azure_openai_api_version=azure_openai_api_version,
                azure_openai_deployment_name=azure_openai_deployment_name,
            )
        else:
            return cls(
                model=model,
                api_key=openai_api_key,
                google_api_key=google_api_key,
            )

    def create_client(self) -> EmbedderClient | None:
        if self.azure_openai_endpoint is not None:
            # Azure OpenAI API setup
            if self.azure_openai_use_managed_identity:
                # Use managed identity for authentication
                token_provider = create_azure_credential_token_provider()
                return AzureOpenAIEmbedderClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        azure_ad_token_provider=token_provider,
                    ),
                    model=self.model,
                )
            elif self.api_key:
                # Use API key for authentication
                return AzureOpenAIEmbedderClient(
                    azure_client=AsyncAzureOpenAI(
                        azure_endpoint=self.azure_openai_endpoint,
                        azure_deployment=self.azure_openai_deployment_name,
                        api_version=self.azure_openai_api_version,
                        api_key=self.api_key,
                    ),
                    model=self.model,
                )
            else:
                logger.error('OPENAI_API_KEY must be set when using Azure OpenAI API')
                return None
        else:
            # Determine embedder type based on model name and available API keys
            if self.model.startswith('text-embedding-') and self.google_api_key:
                # Use Gemini embeddings
                if not GEMINI_EMBEDDER_AVAILABLE:
                    logger.error('google-genai is required for Gemini embeddings but not available')
                    return None
                
                if not self.google_api_key:
                    logger.error('GOOGLE_API_KEY must be set when using Gemini embeddings')
                    return None
                
                embedder_config = GeminiEmbedderConfig(
                    api_key=self.google_api_key,
                    embedding_model=self.model
                )
                logger.info(f'Using Gemini embeddings with model: {self.model}')
                return GeminiEmbedder(config=embedder_config)
            
            elif self.api_key:
                # Use OpenAI embeddings
                embedder_config = OpenAIEmbedderConfig(api_key=self.api_key, embedding_model=self.model)
                return OpenAIEmbedder(config=embedder_config)
            
            else:
                logger.warning('No valid embedder configuration found. Either OPENAI_API_KEY or GOOGLE_API_KEY must be set.')
                return None


class Neo4jConfig(BaseModel):
    """Configuration for Neo4j database connection."""

    uri: str = 'bolt://localhost:7687'
    user: str = 'neo4j'
    password: str = 'password'

    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Create Neo4j configuration from environment variables."""
        return cls(
            uri=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
            user=os.environ.get('NEO4J_USER', 'neo4j'),
            password=os.environ.get('NEO4J_PASSWORD', 'password'),
        )


class GraphitiConfig(BaseModel):
    """Configuration for Graphiti client.

    Centralizes all configuration parameters for the Graphiti client.
    """

    llm: GraphitiLLMConfig = Field(default_factory=GraphitiLLMConfig)
    embedder: GraphitiEmbedderConfig = Field(default_factory=GraphitiEmbedderConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    group_id: str | None = None
    use_custom_entities: bool = False
    destroy_graph: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiConfig':
        """Create a configuration instance from environment variables."""
        return cls(
            llm=GraphitiLLMConfig.from_env(),
            embedder=GraphitiEmbedderConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiConfig':
        """Create configuration from CLI arguments, falling back to environment variables."""
        # Start with environment configuration
        config = cls.from_env()

        # Apply CLI overrides
        if args.group_id:
            config.group_id = args.group_id
        else:
            config.group_id = 'default'

        config.use_custom_entities = args.use_custom_entities
        config.destroy_graph = args.destroy_graph

        # Update LLM config using CLI args
        config.llm = GraphitiLLMConfig.from_cli_and_env(args)

        return config


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    transport: str = 'sse'  # Default to SSE transport

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> 'MCPConfig':
        """Create MCP configuration from CLI arguments."""
        return cls(transport=args.transport)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def handle_gemini_runtime_error(error: Exception, operation: str = 'LLM operation') -> str:
    """Handle Gemini-specific runtime errors and return user-friendly error messages.
    
    Args:
        error: The exception that occurred
        operation: Description of the operation that failed
        
    Returns:
        User-friendly error message with guidance
    """
    error_msg = str(error).lower()
    
    # Handle Gemini safety filter errors
    if 'safety' in error_msg or 'blocked' in error_msg:
        return (
            f'{operation} failed due to Gemini safety filters. '
            'The content was blocked for safety reasons. '
            'Try rephrasing your request to avoid potentially harmful content, '
            'or check Google\'s AI safety policies for more information.'
        )
    
    # Handle rate limiting errors
    if any(term in error_msg for term in ['rate limit', 'quota', 'resource_exhausted', '429']):
        return (
            f'{operation} failed due to rate limiting or quota exceeded. '
            'Please wait a moment and try again, or check your Google API usage limits '
            'and billing settings at https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com'
        )
    
    # Handle authentication errors
    if any(term in error_msg for term in ['api key', 'authentication', 'unauthorized', '401', '403']):
        return (
            f'{operation} failed due to authentication error. '
            'Please check your GOOGLE_API_KEY environment variable and ensure it\'s valid. '
            'Get a new API key from: https://aistudio.google.com/app/apikey'
        )
    
    # Handle model not found errors
    if 'model' in error_msg and any(term in error_msg for term in ['not found', 'invalid', 'unsupported']):
        return (
            f'{operation} failed because the Gemini model is not found or unsupported. '
            'Please check your model name and ensure it\'s a valid Gemini model '
            '(e.g., gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-flash).'
        )
    
    # Handle network connectivity errors
    if any(term in error_msg for term in ['network', 'connection', 'timeout', 'unreachable']):
        return (
            f'{operation} failed due to network connectivity issues. '
            'Please check your internet connection and try again. '
            'If the problem persists, Google\'s Gemini API may be temporarily unavailable.'
        )
    
    # Handle token limit errors
    if any(term in error_msg for term in ['token', 'length', 'too long', 'context']):
        return (
            f'{operation} failed because the input is too long for the Gemini model. '
            'Try reducing the size of your input or breaking it into smaller chunks.'
        )
    
    # Handle thinking configuration errors
    if 'thinking' in error_msg:
        return (
            f'{operation} failed due to thinking configuration error. '
            'Thinking is only supported on Gemini 2.5+ models. '
            'Either use a supported model or disable thinking by setting GEMINI_THINKING_ENABLED=false.'
        )
    
    # Generic error handling for other Gemini errors
    return (
        f'{operation} failed with Gemini error: {error}. '
        'Please check your configuration and try again. '
        'If the problem persists, consult the Gemini API documentation.'
    )

# Create global config instance - will be properly initialized later
config = GraphitiConfig()

# MCP server instructions
GRAPHITI_MCP_INSTRUCTIONS = """
Graphiti is a memory service for AI agents built on a knowledge graph. Graphiti performs well
with dynamic data such as user interactions, changing enterprise data, and external information.

Graphiti transforms information into a richly connected knowledge network, allowing you to 
capture relationships between concepts, entities, and information. The system organizes data as episodes 
(content snippets), nodes (entities), and facts (relationships between entities), creating a dynamic, 
queryable memory store that evolves with new information. Graphiti supports multiple data formats, including 
structured JSON data, enabling seamless integration with existing data pipelines and systems.

Facts contain temporal metadata, allowing you to track the time of creation and whether a fact is invalid 
(superseded by new information).

Key capabilities:
1. Add episodes (text, messages, or JSON) to the knowledge graph with the add_memory tool
2. Search for nodes (entities) in the graph using natural language queries with search_nodes
3. Find relevant facts (relationships between entities) with search_facts
4. Retrieve specific entity edges or episodes by UUID
5. Manage the knowledge graph with tools like delete_episode, delete_entity_edge, and clear_graph

The server connects to a database for persistent storage and uses language models for certain operations. 
Each piece of information is organized by group_id, allowing you to maintain separate knowledge domains.

When adding information, provide descriptive names and detailed content to improve search quality. 
When searching, use specific queries and consider filtering by group_id for more relevant results.

For optimal performance, ensure the database is properly configured and accessible, and valid 
API keys are provided for any language model operations.
"""

# MCP server instance
mcp = FastMCP(
    'Graphiti Agent Memory',
    instructions=GRAPHITI_MCP_INSTRUCTIONS,
)

# Initialize Graphiti client
graphiti_client: Graphiti | None = None


async def initialize_graphiti():
    """Initialize the Graphiti client with the configured settings."""
    global graphiti_client, config

    try:
        # Create LLM client if possible
        llm_client = config.llm.create_client()
        if not llm_client and config.use_custom_entities:
            # If custom entities are enabled, we must have an LLM client
            raise ValueError('OPENAI_API_KEY must be set when custom entities are enabled')

        # Validate Neo4j configuration
        if not config.neo4j.uri or not config.neo4j.user or not config.neo4j.password:
            raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

        embedder_client = config.embedder.create_client()

        # Create cross encoder (reranker) client
        cross_encoder_client = None
        if llm_client and config.llm._detect_provider() == 'gemini':
            # Use Gemini reranker when using Gemini models
            logger.debug(f'GEMINI_RERANKER_AVAILABLE: {GEMINI_RERANKER_AVAILABLE}')
            logger.debug(f'google_api_key available: {bool(config.llm.google_api_key)}')
            
            if GEMINI_RERANKER_AVAILABLE and config.llm.google_api_key:
                try:
                    # Create LLMConfig for the reranker
                    reranker_config = LLMConfig(
                        api_key=config.llm.google_api_key,
                        model='gemini-2.5-flash-lite-preview-06-17',  # Default reranker model
                        temperature=0.0
                    )
                    cross_encoder_client = GeminiRerankerClient(config=reranker_config)
                    logger.info('Using Gemini reranker for cross encoding')
                except Exception as e:
                    logger.warning(f'Failed to create Gemini reranker, will try OpenAI fallback: {e}')
            else:
                if not GEMINI_RERANKER_AVAILABLE:
                    logger.warning('Gemini reranker not available (import failed)')
                if not config.llm.google_api_key:
                    logger.warning('Google API key not available for reranker')
                logger.warning('Will try OpenAI reranker fallback')
        
        # Fallback to OpenAI reranker if no cross encoder client was created and OpenAI API key is available
        if cross_encoder_client is None and OPENAI_RERANKER_AVAILABLE and config.llm.api_key:
            try:
                # Create OpenAI reranker as fallback
                openai_reranker_config = LLMConfig(
                    api_key=config.llm.api_key,
                    model=config.llm.small_model,  # Use small model for reranking
                    temperature=0.0
                )
                cross_encoder_client = OpenAIRerankerClient(config=openai_reranker_config)
                logger.info('Using OpenAI reranker as fallback')
            except Exception as e:
                logger.warning(f'Failed to create OpenAI reranker fallback: {e}')
        
        # If no reranker could be created, use a no-op client
        if cross_encoder_client is None:
            cross_encoder_client = NoOpCrossEncoderClient()
            logger.warning('Using no-op cross encoder (reranker) client - search ranking may be less accurate')
        
        # Initialize Graphiti client
        graphiti_client = Graphiti(
            uri=config.neo4j.uri,
            user=config.neo4j.user,
            password=config.neo4j.password,
            llm_client=llm_client,
            embedder=embedder_client,
            cross_encoder=cross_encoder_client,
            max_coroutines=SEMAPHORE_LIMIT,
        )

        # Destroy graph if requested
        if config.destroy_graph:
            logger.info('Destroying graph...')
            await clear_data(graphiti_client.driver)

        # Initialize the graph database with Graphiti's indices
        await graphiti_client.build_indices_and_constraints()
        logger.info('Graphiti client initialized successfully')

        # Log configuration details for transparency
        if llm_client:
            try:
                provider = config.llm._detect_provider()
                logger.info(f'Using {provider} provider with model: {config.llm.model}')
                logger.info(f'Using temperature: {config.llm.temperature}')
                
                # Log Gemini-specific configuration
                if provider == 'gemini':
                    logger.info(f'Gemini thinking enabled: {config.llm.gemini_thinking_enabled}')
                    if config.llm.gemini_thinking_enabled and config.llm.model:
                        if config.llm.model.startswith('gemini-2.5'):
                            logger.info('Thinking configuration will be applied to supported model')
                        else:
                            logger.warning('Thinking configuration requested but model may not support it')
            except Exception:
                logger.info(f'Using LLM model: {config.llm.model}')
                logger.info(f'Using temperature: {config.llm.temperature}')
        else:
            logger.info('No LLM client configured - entity extraction will be limited')

        logger.info(f'Using group_id: {config.group_id}')
        logger.info(
            f'Custom entity extraction: {"enabled" if config.use_custom_entities else "disabled"}'
        )
        logger.info(f'Using concurrency limit: {SEMAPHORE_LIMIT}')

    except Exception as e:
        # Enhanced error handling with specific guidance for different error types
        error_msg = str(e).lower()
        
        if 'google-genai' in error_msg or 'gemini' in error_msg:
            logger.error(
                f'Failed to initialize Graphiti due to Gemini configuration error: {e}\n'
                'Please ensure:\n'
                '  1. google-genai is installed: pip install graphiti-core[google-genai]\n'
                '  2. GOOGLE_API_KEY is set with a valid API key from https://aistudio.google.com/app/apikey\n'
                '  3. Your model name follows Gemini naming convention (e.g., gemini-2.5-flash)'
            )
        elif 'api key' in error_msg or 'authentication' in error_msg:
            logger.error(
                f'Failed to initialize Graphiti due to API key error: {e}\n'
                'Please check your API key configuration:\n'
                '  - For OpenAI: Set OPENAI_API_KEY\n'
                '  - For Gemini: Set GOOGLE_API_KEY\n'
                '  - For Azure OpenAI: Set OPENAI_API_KEY and Azure configuration'
            )
        elif 'neo4j' in error_msg or 'database' in error_msg:
            logger.error(
                f'Failed to initialize Graphiti due to database error: {e}\n'
                'Please ensure:\n'
                '  1. Neo4j is running and accessible\n'
                '  2. NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD are correctly set\n'
                '  3. Database credentials are valid'
            )
        else:
            logger.error(f'Failed to initialize Graphiti: {e}')
        
        raise


def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into a readable result.

    Since EntityEdge is a Pydantic BaseModel, we can use its built-in serialization capabilities.

    Args:
        edge: The EntityEdge to format

    Returns:
        A dictionary representation of the edge with serialized dates and excluded embeddings
    """
    result = edge.model_dump(
        mode='json',
        exclude={
            'fact_embedding',
        },
    )
    result.get('attributes', {}).pop('fact_embedding', None)
    return result


# Dictionary to store queues for each group_id
# Each queue is a list of tasks to be processed sequentially
episode_queues: dict[str, asyncio.Queue] = {}
# Dictionary to track if a worker is running for each group_id
queue_workers: dict[str, bool] = {}


async def process_episode_queue(group_id: str):
    """Process episodes for a specific group_id sequentially.

    This function runs as a long-lived task that processes episodes
    from the queue one at a time.
    """
    global queue_workers

    logger.info(f'Starting episode queue worker for group_id: {group_id}')
    queue_workers[group_id] = True

    try:
        while True:
            # Get the next episode processing function from the queue
            # This will wait if the queue is empty
            process_func = await episode_queues[group_id].get()

            try:
                # Process the episode
                await process_func()
            except Exception as e:
                logger.error(f'Error processing queued episode for group_id {group_id}: {str(e)}')
            finally:
                # Mark the task as done regardless of success/failure
                episode_queues[group_id].task_done()
    except asyncio.CancelledError:
        logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
    except Exception as e:
        logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
    finally:
        queue_workers[group_id] = False
        logger.info(f'Stopped episode queue worker for group_id: {group_id}')


@mcp.tool()
async def add_memory(
    name: str,
    episode_body: str,
    group_id: str | None = None,
    source: str = 'text',
    source_description: str = '',
    uuid: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Add an episode to memory. This is the primary way to add information to the graph.

    This function returns immediately and processes the episode addition in the background.
    Episodes for the same group_id are processed sequentially to avoid race conditions.

    Args:
        name (str): Name of the episode
        episode_body (str): The content of the episode to persist to memory. When source='json', this must be a
                           properly escaped JSON string, not a raw Python dictionary. The JSON data will be
                           automatically processed to extract entities and relationships.
        group_id (str, optional): A unique ID for this graph. If not provided, uses the default group_id from CLI
                                 or a generated one.
        source (str, optional): Source type, must be one of:
                               - 'text': For plain text content (default)
                               - 'json': For structured data
                               - 'message': For conversation-style content
        source_description (str, optional): Description of the source
        uuid (str, optional): Optional UUID for the episode

    Examples:
        # Adding plain text content
        add_memory(
            name="Company News",
            episode_body="Acme Corp announced a new product line today.",
            source="text",
            source_description="news article",
            group_id="some_arbitrary_string"
        )

        # Adding structured JSON data
        # NOTE: episode_body must be a properly escaped JSON string. Note the triple backslashes
        add_memory(
            name="Customer Profile",
            episode_body="{\\\"company\\\": {\\\"name\\\": \\\"Acme Technologies\\\"}, \\\"products\\\": [{\\\"id\\\": \\\"P001\\\", \\\"name\\\": \\\"CloudSync\\\"}, {\\\"id\\\": \\\"P002\\\", \\\"name\\\": \\\"DataMiner\\\"}]}",
            source="json",
            source_description="CRM data"
        )

        # Adding message-style content
        add_memory(
            name="Customer Conversation",
            episode_body="user: What's your return policy?\nassistant: You can return items within 30 days.",
            source="message",
            source_description="chat transcript",
            group_id="some_arbitrary_string"
        )

    Notes:
        When using source='json':
        - The JSON must be a properly escaped string, not a raw Python dictionary
        - The JSON will be automatically processed to extract entities and relationships
        - Complex nested structures are supported (arrays, nested objects, mixed data types), but keep nesting to a minimum
        - Entities will be created from appropriate JSON properties
        - Relationships between entities will be established based on the JSON structure
    """
    global graphiti_client, episode_queues, queue_workers

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Map string source to EpisodeType enum
        source_type = EpisodeType.text
        if source.lower() == 'message':
            source_type = EpisodeType.message
        elif source.lower() == 'json':
            source_type = EpisodeType.json

        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id if group_id is not None else config.group_id

        # Cast group_id to str to satisfy type checker
        # The Graphiti client expects a str for group_id, not Optional[str]
        group_id_str = str(effective_group_id) if effective_group_id is not None else ''

        # We've already checked that graphiti_client is not None above
        # This assert statement helps type checkers understand that graphiti_client is defined
        assert graphiti_client is not None, 'graphiti_client should not be None here'

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Define the episode processing function
        async def process_episode():
            try:
                logger.info(f"Processing queued episode '{name}' for group_id: {group_id_str}")
                # Use all entity types if use_custom_entities is enabled, otherwise use empty dict
                entity_types = ENTITY_TYPES if config.use_custom_entities else {}

                await client.add_episode(
                    name=name,
                    episode_body=episode_body,
                    source=source_type,
                    source_description=source_description,
                    group_id=group_id_str,  # Using the string version of group_id
                    uuid=uuid,
                    reference_time=datetime.now(timezone.utc),
                    entity_types=entity_types,
                )
                logger.info(f"Episode '{name}' added successfully")

                logger.info(f"Episode '{name}' processed successfully")
            except Exception as e:
                # Enhanced error handling with Gemini-specific error messages
                if isinstance(e, (RateLimitError, RefusalError, EmptyResponseError)):
                    # Handle known LLM client errors
                    if isinstance(e, RateLimitError):
                        error_msg = handle_gemini_runtime_error(e, f"Episode processing for '{name}'")
                    elif isinstance(e, RefusalError):
                        error_msg = f"Episode processing for '{name}' was refused by the LLM: {e}"
                    else:  # EmptyResponseError
                        error_msg = f"Episode processing for '{name}' received empty response from LLM: {e}"
                else:
                    # Check if this might be a Gemini-specific error based on the current LLM configuration
                    try:
                        current_provider = config.llm._detect_provider() if config.llm else 'unknown'
                        if current_provider == 'gemini':
                            error_msg = handle_gemini_runtime_error(e, f"Episode processing for '{name}'")
                        else:
                            error_msg = str(e)
                    except Exception:
                        # Fallback to generic error message if provider detection fails
                        error_msg = str(e)
                
                logger.error(
                    f"Error processing episode '{name}' for group_id {group_id_str}: {error_msg}"
                )

        # Initialize queue for this group_id if it doesn't exist
        if group_id_str not in episode_queues:
            episode_queues[group_id_str] = asyncio.Queue()

        # Add the episode processing function to the queue
        await episode_queues[group_id_str].put(process_episode)

        # Start a worker for this queue if one isn't already running
        if not queue_workers.get(group_id_str, False):
            asyncio.create_task(process_episode_queue(group_id_str))

        # Return immediately with a success message
        return SuccessResponse(
            message=f"Episode '{name}' queued for processing (position: {episode_queues[group_id_str].qsize()})"
        )
    except Exception as e:
        # Enhanced error handling for configuration and setup errors
        try:
            current_provider = config.llm._detect_provider() if config.llm else 'unknown'
            if current_provider == 'gemini':
                error_msg = handle_gemini_runtime_error(e, 'Episode queuing')
            else:
                error_msg = str(e)
        except Exception:
            # Fallback to generic error message if provider detection fails
            error_msg = str(e)
        
        logger.error(f'Error queuing episode task: {error_msg}')
        return ErrorResponse(error=f'Error queuing episode task: {error_msg}')


@mcp.tool()
async def search_memory_nodes(
    query: str,
    group_ids: list[str] | None = None,
    max_nodes: int = 10,
    center_node_uuid: str | None = None,
    entity: str = '',  # cursor seems to break with None
) -> NodeSearchResponse | ErrorResponse:
    """Search the graph memory for relevant node summaries.
    These contain a summary of all of a node's relationships with other nodes.

    Note: entity is a single entity type to filter results (permitted: "Preference", "Procedure").

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_nodes: Maximum number of nodes to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
        entity: Optional single entity type to filter results (permitted: "Preference", "Procedure")
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids if group_ids is not None else [config.group_id] if config.group_id else []
        )

        # Configure the search
        if center_node_uuid is not None:
            search_config = NODE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
        else:
            search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        search_config.limit = max_nodes

        filters = SearchFilters()
        if entity != '':
            filters.node_labels = [entity]

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Perform the search using the _search method
        search_results = await client._search(
            query=query,
            config=search_config,
            group_ids=effective_group_ids,
            center_node_uuid=center_node_uuid,
            search_filter=filters,
        )

        if not search_results.nodes:
            return NodeSearchResponse(message='No relevant nodes found', nodes=[])

        # Format the node results
        formatted_nodes: list[NodeResult] = [
            {
                'uuid': node.uuid,
                'name': node.name,
                'summary': node.summary if hasattr(node, 'summary') else '',
                'labels': node.labels if hasattr(node, 'labels') else [],
                'group_id': node.group_id,
                'created_at': node.created_at.isoformat(),
                'attributes': node.attributes if hasattr(node, 'attributes') else {},
            }
            for node in search_results.nodes
        ]

        return NodeSearchResponse(message='Nodes retrieved successfully', nodes=formatted_nodes)
    except Exception as e:
        # Enhanced error handling with Gemini-specific error messages
        if isinstance(e, (RateLimitError, RefusalError, EmptyResponseError)):
            # Handle known LLM client errors
            if isinstance(e, RateLimitError):
                error_msg = handle_gemini_runtime_error(e, 'Node search')
            elif isinstance(e, RefusalError):
                error_msg = f"Node search was refused by the LLM: {e}"
            else:  # EmptyResponseError
                error_msg = f"Node search received empty response from LLM: {e}"
        else:
            # Check if this might be a Gemini-specific error
            try:
                current_provider = config.llm._detect_provider() if config.llm else 'unknown'
                if current_provider == 'gemini':
                    error_msg = handle_gemini_runtime_error(e, 'Node search')
                else:
                    error_msg = str(e)
            except Exception:
                error_msg = str(e)
        
        logger.error(f'Error searching nodes: {error_msg}')
        return ErrorResponse(error=f'Error searching nodes: {error_msg}')


@mcp.tool()
async def search_memory_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
    center_node_uuid: str | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Search the graph memory for relevant facts.

    Args:
        query: The search query
        group_ids: Optional list of group IDs to filter results
        max_facts: Maximum number of facts to return (default: 10)
        center_node_uuid: Optional UUID of a node to center the search around
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Validate max_facts parameter
        if max_facts <= 0:
            return ErrorResponse(error='max_facts must be a positive integer')

        # Use the provided group_ids or fall back to the default from config if none provided
        effective_group_ids = (
            group_ids if group_ids is not None else [config.group_id] if config.group_id else []
        )

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        relevant_edges = await client.search(
            group_ids=effective_group_ids,
            query=query,
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
        )

        if not relevant_edges:
            return FactSearchResponse(message='No relevant facts found', facts=[])

        facts = [format_fact_result(edge) for edge in relevant_edges]
        return FactSearchResponse(message='Facts retrieved successfully', facts=facts)
    except Exception as e:
        # Enhanced error handling with Gemini-specific error messages
        if isinstance(e, (RateLimitError, RefusalError, EmptyResponseError)):
            # Handle known LLM client errors
            if isinstance(e, RateLimitError):
                error_msg = handle_gemini_runtime_error(e, 'Fact search')
            elif isinstance(e, RefusalError):
                error_msg = f"Fact search was refused by the LLM: {e}"
            else:  # EmptyResponseError
                error_msg = f"Fact search received empty response from LLM: {e}"
        else:
            # Check if this might be a Gemini-specific error
            try:
                current_provider = config.llm._detect_provider() if config.llm else 'unknown'
                if current_provider == 'gemini':
                    error_msg = handle_gemini_runtime_error(e, 'Fact search')
                else:
                    error_msg = str(e)
            except Exception:
                error_msg = str(e)
        
        logger.error(f'Error searching facts: {error_msg}')
        return ErrorResponse(error=f'Error searching facts: {error_msg}')


@mcp.tool()
async def delete_entity_edge(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an entity edge from the graph memory.

    Args:
        uuid: UUID of the entity edge to delete
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge by UUID
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)
        # Delete the edge using its delete method
        await entity_edge.delete(client.driver)
        return SuccessResponse(message=f'Entity edge with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error deleting entity edge: {error_msg}')


@mcp.tool()
async def delete_episode(uuid: str) -> SuccessResponse | ErrorResponse:
    """Delete an episode from the graph memory.

    Args:
        uuid: UUID of the episode to delete
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the episodic node by UUID - EpisodicNode is already imported at the top
        episodic_node = await EpisodicNode.get_by_uuid(client.driver, uuid)
        # Delete the node using its delete method
        await episodic_node.delete(client.driver)
        return SuccessResponse(message=f'Episode with UUID {uuid} deleted successfully')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error deleting episode: {error_msg}')
        return ErrorResponse(error=f'Error deleting episode: {error_msg}')


@mcp.tool()
async def get_entity_edge(uuid: str) -> dict[str, Any] | ErrorResponse:
    """Get an entity edge from the graph memory by its UUID.

    Args:
        uuid: UUID of the entity edge to retrieve
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Get the entity edge directly using the EntityEdge class method
        entity_edge = await EntityEdge.get_by_uuid(client.driver, uuid)

        # Use the format_fact_result function to serialize the edge
        # Return the Python dict directly - MCP will handle serialization
        return format_fact_result(entity_edge)
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting entity edge: {error_msg}')
        return ErrorResponse(error=f'Error getting entity edge: {error_msg}')


@mcp.tool()
async def get_episodes(
    group_id: str | None = None, last_n: int = 10
) -> list[dict[str, Any]] | EpisodeSearchResponse | ErrorResponse:
    """Get the most recent memory episodes for a specific group.

    Args:
        group_id: ID of the group to retrieve episodes from. If not provided, uses the default group_id.
        last_n: Number of most recent episodes to retrieve (default: 10)
    """
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # Use the provided group_id or fall back to the default from config
        effective_group_id = group_id if group_id is not None else config.group_id

        if not isinstance(effective_group_id, str):
            return ErrorResponse(error='Group ID must be a string')

        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        episodes = await client.retrieve_episodes(
            group_ids=[effective_group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
        )

        if not episodes:
            return EpisodeSearchResponse(
                message=f'No episodes found for group {effective_group_id}', episodes=[]
            )

        # Use Pydantic's model_dump method for EpisodicNode serialization
        formatted_episodes = [
            # Use mode='json' to handle datetime serialization
            episode.model_dump(mode='json')
            for episode in episodes
        ]

        # Return the Python list directly - MCP will handle serialization
        return formatted_episodes
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error getting episodes: {error_msg}')
        return ErrorResponse(error=f'Error getting episodes: {error_msg}')


@mcp.tool()
async def clear_graph() -> SuccessResponse | ErrorResponse:
    """Clear all data from the graph memory and rebuild indices."""
    global graphiti_client

    if graphiti_client is None:
        return ErrorResponse(error='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # clear_data is already imported at the top
        await clear_data(client.driver)
        await client.build_indices_and_constraints()
        return SuccessResponse(message='Graph cleared successfully and indices rebuilt')
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error clearing graph: {error_msg}')
        return ErrorResponse(error=f'Error clearing graph: {error_msg}')


@mcp.resource('http://graphiti/status')
async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and Neo4j connection."""
    global graphiti_client

    if graphiti_client is None:
        return StatusResponse(status='error', message='Graphiti client not initialized')

    try:
        # We've already checked that graphiti_client is not None above
        assert graphiti_client is not None

        # Use cast to help the type checker understand that graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        # Test database connection
        await client.driver.client.verify_connectivity()  # type: ignore

        return StatusResponse(
            status='ok', message='Graphiti MCP server is running and connected to Neo4j'
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error checking Neo4j connection: {error_msg}')
        return StatusResponse(
            status='error',
            message=f'Graphiti MCP server is running but Neo4j connection failed: {error_msg}',
        )


async def initialize_server() -> MCPConfig:
    """Parse CLI arguments and initialize the Graphiti server configuration."""
    global config

    parser = argparse.ArgumentParser(
        description='Run the Graphiti MCP server with optional LLM client'
    )
    parser.add_argument(
        '--group-id',
        help='Namespace for the graph. This is an arbitrary string used to organize related data. '
        'If not provided, a random UUID will be generated.',
    )
    parser.add_argument(
        '--transport',
        choices=['sse', 'stdio'],
        default='sse',
        help='Transport to use for communication with the client. (default: sse)',
    )
    parser.add_argument(
        '--model', 
        help=f'Model name to use with the LLM client. '
             f'Supports OpenAI models (gpt-4.1-mini, gpt-4o), '
             f'Gemini models (gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-flash), '
             f'and Azure OpenAI models. Requires corresponding API key: '
             f'OPENAI_API_KEY for OpenAI/gpt models, GOOGLE_API_KEY for Gemini models. '
             f'(default: {DEFAULT_LLM_MODEL})'
    )
    parser.add_argument(
        '--small-model',
        help=f'Small model name to use with the LLM client. (default: {SMALL_LLM_MODEL})',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        help='Temperature setting for the LLM (0.0-2.0). Lower values make output more deterministic. (default: 0.7)',
    )
    parser.add_argument('--destroy-graph', action='store_true', help='Destroy all Graphiti graphs')
    parser.add_argument(
        '--use-custom-entities',
        action='store_true',
        help='Enable entity extraction using the predefined ENTITY_TYPES',
    )
    parser.add_argument(
        '--host',
        default=os.environ.get('MCP_SERVER_HOST'),
        help='Host to bind the MCP server to (default: MCP_SERVER_HOST environment variable)',
    )

    args = parser.parse_args()

    # Build configuration from CLI arguments and environment variables
    config = GraphitiConfig.from_cli_and_env(args)

    # Log the group ID configuration
    if args.group_id:
        logger.info(f'Using provided group_id: {config.group_id}')
    else:
        logger.info(f'Generated random group_id: {config.group_id}')

    # Log entity extraction configuration
    if config.use_custom_entities:
        logger.info('Entity extraction enabled using predefined ENTITY_TYPES')
    else:
        logger.info('Entity extraction disabled (no custom entities will be used)')

    # Initialize Graphiti
    await initialize_graphiti()

    if args.host:
        logger.info(f'Setting MCP server host to: {args.host}')
        # Set MCP server host from CLI or env
        mcp.settings.host = args.host

    # Return MCP configuration
    return MCPConfig.from_cli(args)


async def run_mcp_server():
    """Run the MCP server in the current event loop."""
    # Initialize the server
    mcp_config = await initialize_server()

    # Run the server with stdio transport for MCP in the same event loop
    logger.info(f'Starting MCP server with transport: {mcp_config.transport}')
    if mcp_config.transport == 'stdio':
        await mcp.run_stdio_async()
    elif mcp_config.transport == 'sse':
        logger.info(
            f'Running MCP server with SSE transport on {mcp.settings.host}:{mcp.settings.port}'
        )
        await mcp.run_sse_async()


def main():
    """Main function to run the Graphiti MCP server."""
    try:
        # Run everything in a single event loop
        asyncio.run(run_mcp_server())
    except Exception as e:
        logger.error(f'Error initializing Graphiti MCP server: {str(e)}')
        raise


if __name__ == '__main__':
    main()
