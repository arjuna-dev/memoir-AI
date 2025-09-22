"""
Tests for configuration management and validation.
"""

import os

import pytest

from memoir_ai.config import MemoirAIConfig
from memoir_ai.exceptions import ConfigurationError


class TestMemoirAIConfig:
    """Test configuration validation."""

    def test_valid_config(self) -> None:
        """Test that valid configuration passes validation."""
        config = MemoirAIConfig(
            database_url="sqlite:///test.db",
            llm_provider="openai",
            model_name="gpt-4",
            hierarchy_depth=3,
            chunk_min_tokens=300,
            chunk_max_tokens=500,
            batch_size=5,
            max_token_budget=4000,
        )
        assert config.database_url == "sqlite:///test.db"
        assert config.hierarchy_depth == 3

    def test_token_budget_validation(self) -> None:
        """Test token budget validation."""
        with pytest.raises(ConfigurationError) as exc_info:
            MemoirAIConfig(
                database_url="sqlite:///test.db",
                chunk_min_tokens=300,
                max_token_budget=350,  # Too small
            )
        assert "max_token_budget" in str(exc_info.value)
        assert "401" in str(exc_info.value)  # 300 + 100 + 1

    def test_chunk_size_validation(self) -> None:
        """Test chunk size validation."""
        with pytest.raises(ConfigurationError) as exc_info:
            MemoirAIConfig(
                database_url="sqlite:///test.db",
                chunk_min_tokens=500,
                chunk_max_tokens=400,  # min > max
            )
        assert "chunk_min_tokens" in str(exc_info.value)

    def test_hierarchy_depth_validation(self) -> None:
        """Test hierarchy depth validation."""
        with pytest.raises(ConfigurationError) as exc_info:
            MemoirAIConfig(
                database_url="sqlite:///test.db", hierarchy_depth=101  # Too large
            )
        assert "hierarchy_depth" in str(exc_info.value)

        with pytest.raises(ConfigurationError) as exc_info:
            MemoirAIConfig(
                database_url="sqlite:///test.db", hierarchy_depth=0  # Too small
            )
        assert "hierarchy_depth" in str(exc_info.value)

    def test_batch_size_validation(self) -> None:
        """Test batch size validation."""
        with pytest.raises(ConfigurationError) as exc_info:
            MemoirAIConfig(database_url="sqlite:///test.db", batch_size=0)  # Too small
        assert "batch_size" in str(exc_info.value)

        with pytest.raises(ConfigurationError) as exc_info:
            MemoirAIConfig(database_url="sqlite:///test.db", batch_size=51)  # Too large
        assert "batch_size" in str(exc_info.value)

    def test_category_limits_validation(self) -> None:
        """Test category limits validation."""
        # Test negative global limit
        with pytest.raises(ConfigurationError) as exc_info:
            MemoirAIConfig(
                database_url="sqlite:///test.db", max_categories_per_level=-1
            )
        assert "max_categories_per_level" in str(exc_info.value)

        # Test per-level limits
        with pytest.raises(ConfigurationError) as exc_info:
            MemoirAIConfig(
                database_url="sqlite:///test.db",
                hierarchy_depth=3,
                max_categories_per_level={1: 10, 2: 0, 3: 5},  # Level 2 has 0 limit
            )
        assert "positive integer" in str(exc_info.value)

        # Test level exceeds hierarchy depth
        with pytest.raises(ConfigurationError) as exc_info:
            MemoirAIConfig(
                database_url="sqlite:///test.db",
                hierarchy_depth=3,
                max_categories_per_level={1: 10, 4: 5},  # Level 4 > depth 3
            )
        assert "exceeds hierarchy_depth" in str(exc_info.value)

    def test_database_url_validation(self) -> None:
        """Test database URL validation."""
        with pytest.raises(ConfigurationError) as exc_info:
            MemoirAIConfig(database_url="")
        assert "database_url cannot be empty" in str(exc_info.value)

        with pytest.raises(ConfigurationError) as exc_info:
            MemoirAIConfig(database_url="invalid://url")
        assert "scheme not supported" in str(exc_info.value)

    def test_environment_variables(self, monkeypatch) -> None:
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("MEMOIR_DATABASE_URL", "postgresql://test")
        monkeypatch.setenv("MEMOIR_HIERARCHY_DEPTH", "5")
        monkeypatch.setenv("MEMOIR_BATCH_SIZE", "10")

        config = MemoirAIConfig(database_url="sqlite:///default.db")

        assert config.database_url == "postgresql://test"
        assert config.hierarchy_depth == 5
        assert config.batch_size == 10

    def test_get_category_limit(self) -> None:
        """Test category limit retrieval."""
        # Test global limit
        config = MemoirAIConfig(
            database_url="sqlite:///test.db", max_categories_per_level=50
        )
        assert config.get_category_limit(1) == 50
        assert config.get_category_limit(2) == 50

        # Test per-level limits
        config = MemoirAIConfig(
            database_url="sqlite:///test.db", max_categories_per_level={1: 10, 2: 20}
        )
        assert config.get_category_limit(1) == 10
        assert config.get_category_limit(2) == 20
        assert config.get_category_limit(3) == 128  # Default fallback
