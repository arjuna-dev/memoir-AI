"""
Tests for LLM agents.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from memoir_ai.exceptions import ConfigurationError
from memoir_ai.llm.agents import (
    AgentFactory,
    create_batch_classification_agent,
    create_category_creation_agent,
    create_category_limit_agent,
    create_classification_agent,
    create_contextual_helper_agent,
    create_final_answer_agent,
    create_query_classification_agent,
    create_summarization_agent,
    get_agent_factory,
    get_native_output_providers,
    get_supported_providers,
    validate_model_name,
)
from memoir_ai.llm.schemas import (
    BatchClassificationResponse,
    CategoryCreation,
    CategoryLimitResponse,
    CategorySelection,
    ContextualHelperGeneration,
    FinalAnswer,
    ModelConfiguration,
    QueryCategorySelection,
    SummarizationResponse,
)


class TestAgentFactory:
    """Test AgentFactory functionality."""

    def test_agent_factory_initialization_defaults(self) -> None:
        """Test AgentFactory initialization with defaults."""
        factory = AgentFactory()

        assert factory.default_config.model_name == "openai:gpt-4"
        assert factory.default_config.temperature == 0.0
        assert factory.default_config.timeout == 30
        assert factory.default_config.retry_attempts == 3

    def test_agent_factory_initialization_custom(self) -> None:
        """Test AgentFactory initialization with custom config."""
        config = ModelConfiguration(
            model_name="anthropic:claude-3",
            temperature=0.5,
            timeout=60,
            retry_attempts=5,
        )

        factory = AgentFactory(config)

        assert factory.default_config.model_name == "anthropic:claude-3"
        assert factory.default_config.temperature == 0.5
        assert factory.default_config.timeout == 60
        assert factory.default_config.retry_attempts == 5

    @patch("memoir_ai.llm.agents.Agent")
    def test_create_classification_agent(self, mock_agent_class) -> None:
        """Test creating classification agent."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_classification_agent()

        assert agent == mock_agent
        mock_agent_class.assert_called_once()

        # Check that the agent was cached
        agent2 = factory.create_classification_agent()
        assert agent2 == mock_agent
        # Should not create a new agent (still only one call)
        assert mock_agent_class.call_count == 1

    @patch("memoir_ai.llm.agents.Agent")
    def test_create_batch_classification_agent(self, mock_agent_class) -> None:
        """Test creating batch classification agent."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_batch_classification_agent()

        assert agent == mock_agent
        mock_agent_class.assert_called_once()

    @patch("memoir_ai.llm.agents.Agent")
    def test_create_query_classification_agent(self, mock_agent_class) -> None:
        """Test creating query classification agent."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_query_classification_agent()

        assert agent == mock_agent
        mock_agent_class.assert_called_once()

    @patch("memoir_ai.llm.agents.Agent")
    def test_create_summarization_agent(self, mock_agent_class) -> None:
        """Test creating summarization agent."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_summarization_agent()

        assert agent == mock_agent
        mock_agent_class.assert_called_once()

    @patch("memoir_ai.llm.agents.Agent")
    def test_create_final_answer_agent(self, mock_agent_class) -> None:
        """Test creating final answer agent."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_final_answer_agent()

        assert agent == mock_agent
        mock_agent_class.assert_called_once()

    @patch("memoir_ai.llm.agents.Agent")
    def test_create_contextual_helper_agent(self, mock_agent_class) -> None:
        """Test creating contextual helper agent."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_contextual_helper_agent()

        assert agent == mock_agent
        mock_agent_class.assert_called_once()

    @patch("memoir_ai.llm.agents.Agent")
    def test_create_category_creation_agent(self, mock_agent_class) -> None:
        """Test creating category creation agent."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_category_creation_agent()

        assert agent == mock_agent
        mock_agent_class.assert_called_once()

    @patch("memoir_ai.llm.agents.Agent")
    def test_create_category_limit_agent(self, mock_agent_class) -> None:
        """Test creating category limit agent."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_category_limit_agent()

        assert agent == mock_agent
        mock_agent_class.assert_called_once()

    @patch("memoir_ai.llm.agents.Agent")
    def test_agent_with_custom_model(self, mock_agent_class) -> None:
        """Test creating agent with custom model."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_classification_agent(model_name="anthropic:claude-3")

        assert agent == mock_agent
        mock_agent_class.assert_called_once()

        # Check that the model name was passed correctly
        call_args = mock_agent_class.call_args
        assert call_args[1]["model"] == "anthropic:claude-3"

    @patch("memoir_ai.llm.agents.Agent")
    def test_agent_with_custom_config(self, mock_agent_class) -> None:
        """Test creating agent with custom configuration."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        config = ModelConfiguration(
            model_name="grok:grok-1",
            temperature=0.7,
            max_tokens=2000,
            timeout=45,
            retry_attempts=2,
        )

        factory = AgentFactory()
        agent = factory.create_classification_agent(config=config)

        assert agent == mock_agent
        mock_agent_class.assert_called_once()

        # Check that the configuration was passed correctly
        call_args = mock_agent_class.call_args
        assert call_args[1]["model"] == "grok:grok-1"
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["max_tokens"] == 2000
        assert call_args[1]["timeout"] == 45
        assert call_args[1]["retries"] == 2

    @patch("memoir_ai.llm.agents.Agent")
    @patch("memoir_ai.llm.agents.supports_native_output")
    @patch("memoir_ai.llm.agents.NativeOutput")
    def test_native_output_support(
        self, mock_native_output, mock_supports, mock_agent_class
    ) -> None:
        """Test native output support detection and usage."""
        mock_supports.return_value = True
        mock_native_output_instance = Mock()
        mock_native_output.return_value = mock_native_output_instance
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_classification_agent(model_name="openai:gpt-4")

        assert agent == mock_agent
        mock_supports.assert_called_once_with("openai:gpt-4")
        mock_native_output.assert_called_once()

        # Check that NativeOutput was used
        call_args = mock_agent_class.call_args
        assert call_args[1]["output_type"] == mock_native_output_instance

    @patch("memoir_ai.llm.agents.Agent")
    @patch("memoir_ai.llm.agents.supports_native_output")
    def test_fallback_to_standard_schema(self, mock_supports, mock_agent_class) -> None:
        """Test fallback to standard schema when native output not supported."""
        mock_supports.return_value = False
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        factory = AgentFactory()
        agent = factory.create_classification_agent(model_name="anthropic:claude-3")

        assert agent == mock_agent
        mock_supports.assert_called_once_with("anthropic:claude-3")

        # Check that standard schema was used (CategorySelection)
        call_args = mock_agent_class.call_args
        assert call_args[1]["output_type"] == CategorySelection

    @patch("memoir_ai.llm.agents.Agent")
    def test_agent_creation_error(self, mock_agent_class) -> None:
        """Test agent creation error handling."""
        mock_agent_class.side_effect = Exception("Agent creation failed")

        factory = AgentFactory()

        with pytest.raises(ConfigurationError) as exc_info:
            factory.create_classification_agent()

        assert "Failed to create CategorySelection agent" in str(exc_info.value)

    def test_cache_management(self) -> None:
        """Test agent cache management."""
        factory = AgentFactory()

        # Initially empty cache
        cache_info = factory.get_cache_info()
        assert cache_info["cached_agents"] == 0
        assert cache_info["cache_keys"] == []

        # Create some agents (mocked)
        with patch("memoir_ai.llm.agents.Agent") as mock_agent_class:
            mock_agent_class.return_value = Mock()

            factory.create_classification_agent()
            factory.create_batch_classification_agent()

            cache_info = factory.get_cache_info()
            assert cache_info["cached_agents"] == 2
            assert len(cache_info["cache_keys"]) == 2

            # Clear cache
            factory.clear_cache()

            cache_info = factory.get_cache_info()
            assert cache_info["cached_agents"] == 0
            assert cache_info["cache_keys"] == []

    def test_effective_config_model_override(self) -> None:
        """Test effective config with model name override."""
        config = ModelConfiguration(model_name="base:model", temperature=0.5)

        factory = AgentFactory(config)
        effective_config = factory._get_effective_config("override:model", None)

        assert effective_config.model_name == "override:model"
        assert effective_config.temperature == 0.5  # Should keep other settings

    def test_effective_config_config_override(self) -> None:
        """Test effective config with config override."""
        default_config = ModelConfiguration(model_name="default:model")
        override_config = ModelConfiguration(
            model_name="override:model", temperature=0.8
        )

        factory = AgentFactory(default_config)
        effective_config = factory._get_effective_config(None, override_config)

        assert effective_config.model_name == "override:model"
        assert effective_config.temperature == 0.8

    def test_effective_config_both_overrides(self) -> None:
        """Test effective config with both model and config overrides."""
        default_config = ModelConfiguration(model_name="default:model")
        override_config = ModelConfiguration(model_name="config:model", temperature=0.8)

        factory = AgentFactory(default_config)
        effective_config = factory._get_effective_config("final:model", override_config)

        # Model name override should take precedence
        assert effective_config.model_name == "final:model"
        assert effective_config.temperature == 0.8


class TestGlobalFunctions:
    """Test global convenience functions."""

    @patch("memoir_ai.llm.agents.get_agent_factory")
    def test_create_classification_agent_global(self, mock_get_factory) -> None:
        """Test global create_classification_agent function."""
        mock_factory = Mock()
        mock_agent = Mock()
        mock_factory.create_classification_agent.return_value = mock_agent
        mock_get_factory.return_value = mock_factory

        agent = create_classification_agent("test:model")

        assert agent == mock_agent
        mock_factory.create_classification_agent.assert_called_once_with(
            "test:model", None
        )

    @patch("memoir_ai.llm.agents.get_agent_factory")
    def test_create_batch_classification_agent_global(self, mock_get_factory) -> None:
        """Test global create_batch_classification_agent function."""
        mock_factory = Mock()
        mock_agent = Mock()
        mock_factory.create_batch_classification_agent.return_value = mock_agent
        mock_get_factory.return_value = mock_factory

        config = ModelConfiguration(model_name="test:model")
        agent = create_batch_classification_agent(config=config)

        assert agent == mock_agent
        mock_factory.create_batch_classification_agent.assert_called_once_with(
            None, config
        )

    def test_get_agent_factory_singleton(self) -> None:
        """Test that get_agent_factory returns singleton."""
        factory1 = get_agent_factory()
        factory2 = get_agent_factory()

        assert factory1 is factory2

    def test_get_agent_factory_with_config(self) -> None:
        """Test get_agent_factory with custom config."""
        config = ModelConfiguration(model_name="custom:model")
        factory = get_agent_factory(config)

        assert factory.default_config.model_name == "custom:model"


class TestValidationFunctions:
    """Test validation utility functions."""

    def test_validate_model_name_valid(self) -> None:
        """Test valid model name validation."""
        assert validate_model_name("openai:gpt-4") is True
        assert validate_model_name("anthropic:claude-3") is True
        assert validate_model_name("provider:model-name") is True

    def test_validate_model_name_invalid(self) -> None:
        """Test invalid model name validation."""
        assert validate_model_name("") is False
        assert validate_model_name("no-colon") is False
        assert validate_model_name(":no-provider") is False
        assert validate_model_name("no-model:") is False
        assert validate_model_name(None) is False
        assert validate_model_name(123) is False

    def test_get_supported_providers(self) -> None:
        """Test getting supported providers."""
        providers = get_supported_providers()

        assert isinstance(providers, list)
        assert "openai" in providers
        assert "anthropic" in providers
        assert "grok" in providers
        assert "gemini" in providers
        assert "ollama" in providers
        assert "azure" in providers

    def test_get_native_output_providers(self) -> None:
        """Test getting native output providers."""
        providers = get_native_output_providers()

        assert isinstance(providers, list)
        assert "openai" in providers
        assert "grok" in providers
        assert "gemini" in providers
        assert "anthropic" not in providers  # Doesn't support native output


class TestAgentFactoryIntegration:
    """Integration tests for AgentFactory."""

    @patch("memoir_ai.llm.agents.Agent")
    def test_different_agents_different_schemas(self, mock_agent_class) -> None:
        """Test that different agent types use different schemas."""
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent

        factory = AgentFactory()

        # Create different types of agents
        classification_agent = factory.create_classification_agent()
        batch_agent = factory.create_batch_classification_agent()
        query_agent = factory.create_query_classification_agent()
        summary_agent = factory.create_summarization_agent()
        answer_agent = factory.create_final_answer_agent()

        # Should have created 5 different agents
        assert mock_agent_class.call_count == 5

        # Check that different schemas were used
        calls = mock_agent_class.call_args_list

        # For non-native output models, should use the schema directly
        with patch("memoir_ai.llm.agents.supports_native_output", return_value=False):
            factory.clear_cache()  # Clear cache to force recreation

            factory.create_classification_agent()
            factory.create_batch_classification_agent()
            factory.create_summarization_agent()
            factory.create_final_answer_agent()

            # Get the new calls (last 4)
            new_calls = mock_agent_class.call_args_list[-4:]

            assert new_calls[0][1]["output_type"] == CategorySelection
            assert new_calls[1][1]["output_type"] == BatchClassificationResponse
            assert new_calls[2][1]["output_type"] == SummarizationResponse
            assert new_calls[3][1]["output_type"] == FinalAnswer

    @patch("memoir_ai.llm.agents.Agent")
    def test_cache_key_uniqueness(self, mock_agent_class) -> None:
        """Test that cache keys are unique for different configurations."""
        mock_agent_class.return_value = Mock()

        factory = AgentFactory()

        # Create agents with different models
        agent1 = factory.create_classification_agent("openai:gpt-4")
        agent2 = factory.create_classification_agent("anthropic:claude-3")

        # Create agents with different temperatures
        config1 = ModelConfiguration(model_name="openai:gpt-4", temperature=0.0)
        config2 = ModelConfiguration(model_name="openai:gpt-4", temperature=0.5)

        agent3 = factory.create_classification_agent(config=config1)
        agent4 = factory.create_classification_agent(config=config2)

        # Should have created 3 unique agents (agent1 and agent3 should be the same)
        cache_info = factory.get_cache_info()
        assert cache_info["cached_agents"] == 3

        # agent1 and agent3 should be the same (same cache key)
        assert agent1 is agent3
