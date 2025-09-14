"""
Pydantic AI agents for LLM interactions.

This module provides agent factory functions and configuration for
different LLM providers with native structured output support.
"""

import logging
from typing import Type, Union, Optional, Dict, Any, List
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput

from .schemas import (
    CategorySelection,
    BatchClassificationResponse,
    QueryCategorySelection,
    SummarizationResponse,
    FinalAnswer,
    ContextualHelperGeneration,
    CategoryCreation,
    CategoryLimitResponse,
    ModelConfiguration,
    supports_native_output,
    CLASSIFICATION_SCHEMAS,
    SUMMARIZATION_SCHEMAS,
    ANSWER_SCHEMAS,
    NATIVE_OUTPUT_SUPPORTED_MODELS,
)
from ..exceptions import ConfigurationError, LLMError


logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Factory class for creating Pydantic AI agents with proper configuration.

    Handles native output support detection, fallback mechanisms, and
    agent configuration for different LLM providers and use cases.
    """

    def __init__(self, default_config: Optional[ModelConfiguration] = None):
        """
        Initialize the agent factory.

        Args:
            default_config: Default configuration for all agents
        """
        self.default_config = default_config or ModelConfiguration(
            model_name="openai:gpt-4", temperature=0.0, timeout=30, retry_attempts=3
        )
        self._agent_cache: Dict[str, Agent] = {}

    def create_classification_agent(
        self,
        model_name: Optional[str] = None,
        config: Optional[ModelConfiguration] = None,
    ) -> Agent:
        """
        Create an agent for single category classification.

        Args:
            model_name: Model to use (overrides config)
            config: Model configuration

        Returns:
            Configured Pydantic AI agent
        """
        effective_config = self._get_effective_config(model_name, config)
        cache_key = f"classification_{effective_config.model_name}_{effective_config.temperature}"

        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        agent = self._create_agent_with_schema(
            schema=CategorySelection,
            model_name=effective_config.model_name,
            config=effective_config,
            agent_name="CategorySelection",
            description="Select the most relevant category for content classification",
        )

        self._agent_cache[cache_key] = agent
        return agent

    def create_batch_classification_agent(
        self,
        model_name: Optional[str] = None,
        config: Optional[ModelConfiguration] = None,
    ) -> Agent:
        """
        Create an agent for batch category classification.

        Args:
            model_name: Model to use (overrides config)
            config: Model configuration

        Returns:
            Configured Pydantic AI agent
        """
        effective_config = self._get_effective_config(model_name, config)
        cache_key = f"batch_classification_{effective_config.model_name}_{effective_config.temperature}"

        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        agent = self._create_agent_with_schema(
            schema=BatchClassificationResponse,
            model_name=effective_config.model_name,
            config=effective_config,
            agent_name="BatchClassification",
            description="Classify multiple chunks into categories in a single request",
        )

        self._agent_cache[cache_key] = agent
        return agent

    def create_query_classification_agent(
        self,
        model_name: Optional[str] = None,
        config: Optional[ModelConfiguration] = None,
    ) -> Agent:
        """
        Create an agent for query-based category selection.

        Args:
            model_name: Model to use (overrides config)
            config: Model configuration

        Returns:
            Configured Pydantic AI agent
        """
        effective_config = self._get_effective_config(model_name, config)
        cache_key = f"query_classification_{effective_config.model_name}_{effective_config.temperature}"

        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        agent = self._create_agent_with_schema(
            schema=QueryCategorySelection,
            model_name=effective_config.model_name,
            config=effective_config,
            agent_name="QueryCategorySelection",
            description="Select relevant categories based on natural language queries",
        )

        self._agent_cache[cache_key] = agent
        return agent

    def create_summarization_agent(
        self,
        model_name: Optional[str] = None,
        config: Optional[ModelConfiguration] = None,
    ) -> Agent:
        """
        Create an agent for content summarization.

        Args:
            model_name: Model to use (overrides config)
            config: Model configuration

        Returns:
            Configured Pydantic AI agent
        """
        effective_config = self._get_effective_config(model_name, config)
        cache_key = f"summarization_{effective_config.model_name}_{effective_config.temperature}"

        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        agent = self._create_agent_with_schema(
            schema=SummarizationResponse,
            model_name=effective_config.model_name,
            config=effective_config,
            agent_name="ContentSummarization",
            description="Summarize content chunks to fit within token budgets",
        )

        self._agent_cache[cache_key] = agent
        return agent

    def create_final_answer_agent(
        self,
        model_name: Optional[str] = None,
        config: Optional[ModelConfiguration] = None,
    ) -> Agent:
        """
        Create an agent for generating final answers to queries.

        Args:
            model_name: Model to use (overrides config)
            config: Model configuration

        Returns:
            Configured Pydantic AI agent
        """
        effective_config = self._get_effective_config(model_name, config)
        cache_key = (
            f"final_answer_{effective_config.model_name}_{effective_config.temperature}"
        )

        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        agent = self._create_agent_with_schema(
            schema=FinalAnswer,
            model_name=effective_config.model_name,
            config=effective_config,
            agent_name="FinalAnswer",
            description="Generate final answers to user queries based on retrieved content",
        )

        self._agent_cache[cache_key] = agent
        return agent

    def create_contextual_helper_agent(
        self,
        model_name: Optional[str] = None,
        config: Optional[ModelConfiguration] = None,
    ) -> Agent:
        """
        Create an agent for generating contextual helpers.

        Args:
            model_name: Model to use (overrides config)
            config: Model configuration

        Returns:
            Configured Pydantic AI agent
        """
        effective_config = self._get_effective_config(model_name, config)
        cache_key = f"contextual_helper_{effective_config.model_name}_{effective_config.temperature}"

        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        agent = self._create_agent_with_schema(
            schema=ContextualHelperGeneration,
            model_name=effective_config.model_name,
            config=effective_config,
            agent_name="ContextualHelper",
            description="Generate contextual helpers for document classification and retrieval",
        )

        self._agent_cache[cache_key] = agent
        return agent

    def create_category_creation_agent(
        self,
        model_name: Optional[str] = None,
        config: Optional[ModelConfiguration] = None,
    ) -> Agent:
        """
        Create an agent for category creation decisions.

        Args:
            model_name: Model to use (overrides config)
            config: Model configuration

        Returns:
            Configured Pydantic AI agent
        """
        effective_config = self._get_effective_config(model_name, config)
        cache_key = f"category_creation_{effective_config.model_name}_{effective_config.temperature}"

        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        agent = self._create_agent_with_schema(
            schema=CategoryCreation,
            model_name=effective_config.model_name,
            config=effective_config,
            agent_name="CategoryCreation",
            description="Decide on new category creation with justification",
        )

        self._agent_cache[cache_key] = agent
        return agent

    def create_category_limit_agent(
        self,
        model_name: Optional[str] = None,
        config: Optional[ModelConfiguration] = None,
    ) -> Agent:
        """
        Create an agent for handling category limit scenarios.

        Args:
            model_name: Model to use (overrides config)
            config: Model configuration

        Returns:
            Configured Pydantic AI agent
        """
        effective_config = self._get_effective_config(model_name, config)
        cache_key = f"category_limit_{effective_config.model_name}_{effective_config.temperature}"

        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key]

        agent = self._create_agent_with_schema(
            schema=CategoryLimitResponse,
            model_name=effective_config.model_name,
            config=effective_config,
            agent_name="CategoryLimit",
            description="Select existing categories when creation limits are reached",
        )

        self._agent_cache[cache_key] = agent
        return agent

    def _create_agent_with_schema(
        self,
        schema: Type[BaseModel],
        model_name: str,
        config: ModelConfiguration,
        agent_name: str,
        description: str,
    ) -> Agent:
        """
        Create an agent with the specified schema and configuration.

        Args:
            schema: Pydantic schema for the agent output
            model_name: Model name to use
            config: Model configuration
            agent_name: Name for the agent (used in native output)
            description: Description for the agent

        Returns:
            Configured Pydantic AI agent
        """
        try:
            # Check if model supports native output
            if supports_native_output(model_name):
                logger.debug(f"Using native output for model {model_name}")
                output_type = NativeOutput(
                    schema, name=agent_name, description=description
                )
            else:
                logger.debug(
                    f"Using standard schema enforcement for model {model_name}"
                )
                output_type = schema

            # Create agent with configuration
            agent = Agent(
                model=model_name,
                output_type=output_type,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                retries=config.retry_attempts,
            )

            logger.info(f"Created {agent_name} agent for model {model_name}")
            return agent

        except Exception as e:
            logger.error(
                f"Failed to create agent {agent_name} for model {model_name}: {e}"
            )
            raise ConfigurationError(
                f"Failed to create {agent_name} agent: {str(e)}",
                parameter="model_configuration",
            )

    def _get_effective_config(
        self, model_name: Optional[str], config: Optional[ModelConfiguration]
    ) -> ModelConfiguration:
        """
        Get the effective configuration by merging provided config with defaults.

        Args:
            model_name: Optional model name override
            config: Optional configuration

        Returns:
            Effective configuration to use
        """
        if config is None:
            config = self.default_config

        if model_name is not None:
            # Create new config with overridden model name
            config = ModelConfiguration(
                model_name=model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                retry_attempts=config.retry_attempts,
            )

        return config

    def clear_cache(self):
        """Clear the agent cache."""
        self._agent_cache.clear()
        logger.info("Agent cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the agent cache.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_agents": len(self._agent_cache),
            "cache_keys": list(self._agent_cache.keys()),
        }


# Global agent factory instance
_default_factory: Optional[AgentFactory] = None


def get_agent_factory(config: Optional[ModelConfiguration] = None) -> AgentFactory:
    """
    Get the global agent factory instance.

    Args:
        config: Optional configuration for the factory

    Returns:
        AgentFactory instance
    """
    global _default_factory

    if _default_factory is None or config is not None:
        _default_factory = AgentFactory(config)

    return _default_factory


# Convenience functions for creating agents
def create_classification_agent(
    model_name: Optional[str] = None, config: Optional[ModelConfiguration] = None
) -> Agent:
    """Create a classification agent using the global factory."""
    return get_agent_factory().create_classification_agent(model_name, config)


def create_batch_classification_agent(
    model_name: Optional[str] = None, config: Optional[ModelConfiguration] = None
) -> Agent:
    """Create a batch classification agent using the global factory."""
    return get_agent_factory().create_batch_classification_agent(model_name, config)


def create_query_classification_agent(
    model_name: Optional[str] = None, config: Optional[ModelConfiguration] = None
) -> Agent:
    """Create a query classification agent using the global factory."""
    return get_agent_factory().create_query_classification_agent(model_name, config)


def create_summarization_agent(
    model_name: Optional[str] = None, config: Optional[ModelConfiguration] = None
) -> Agent:
    """Create a summarization agent using the global factory."""
    return get_agent_factory().create_summarization_agent(model_name, config)


def create_final_answer_agent(
    model_name: Optional[str] = None, config: Optional[ModelConfiguration] = None
) -> Agent:
    """Create a final answer agent using the global factory."""
    return get_agent_factory().create_final_answer_agent(model_name, config)


def create_contextual_helper_agent(
    model_name: Optional[str] = None, config: Optional[ModelConfiguration] = None
) -> Agent:
    """Create a contextual helper agent using the global factory."""
    return get_agent_factory().create_contextual_helper_agent(model_name, config)


def create_category_creation_agent(
    model_name: Optional[str] = None, config: Optional[ModelConfiguration] = None
) -> Agent:
    """Create a category creation agent using the global factory."""
    return get_agent_factory().create_category_creation_agent(model_name, config)


def create_category_limit_agent(
    model_name: Optional[str] = None, config: Optional[ModelConfiguration] = None
) -> Agent:
    """Create a category limit agent using the global factory."""
    return get_agent_factory().create_category_limit_agent(model_name, config)


# Model validation functions
def validate_model_name(model_name: str) -> bool:
    """
    Validate that a model name is properly formatted.

    Args:
        model_name: Model name to validate

    Returns:
        True if valid, False otherwise
    """
    if not model_name or not isinstance(model_name, str):
        return False

    # Check for provider:model format
    if ":" not in model_name:
        return False

    provider, model = model_name.split(":", 1)
    return bool(provider.strip() and model.strip())


def get_supported_providers() -> List[str]:
    """
    Get list of supported LLM providers.

    Returns:
        List of provider names
    """
    return ["openai", "anthropic", "grok", "gemini", "ollama", "azure"]


def get_native_output_providers() -> List[str]:
    """
    Get list of providers that support native structured output.

    Returns:
        List of provider names that support native output
    """
    return list(NATIVE_OUTPUT_SUPPORTED_MODELS)
