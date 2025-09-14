"""
Core MemoirAI class and main API interface.
"""

from typing import Optional, Dict, List, Union
from enum import Enum

from .models import (
    IngestionResult,
    QueryResult,
    CategoryTree,
    QueryStrategy,
    PromptLimitingStrategy,
)
from .config import MemoirAIConfig
from .exceptions import ConfigurationError
from .database import DatabaseManager, MigrationManager
from .text_processing import TextChunker


class MemoirAI:
    """
    Main MemoirAI class providing LLM-powered hierarchical text storage and retrieval.
    """

    def __init__(
        self,
        database_url: str,
        llm_provider: str = "openai",
        hierarchy_depth: int = 3,
        chunk_min_tokens: int = 300,
        chunk_max_tokens: int = 500,
        model_name: str = "gpt-4",
        batch_size: int = 5,
        max_categories_per_level: Union[int, Dict[int, int]] = 128,
        auto_source_identification: bool = True,
        **kwargs,
    ):
        """Initialize memoirAI with configuration parameters."""
        # Configuration will be implemented in config module
        self.config = MemoirAIConfig(
            database_url=database_url,
            llm_provider=llm_provider,
            hierarchy_depth=hierarchy_depth,
            chunk_min_tokens=chunk_min_tokens,
            chunk_max_tokens=chunk_max_tokens,
            model_name=model_name,
            batch_size=batch_size,
            max_categories_per_level=max_categories_per_level,
            auto_source_identification=auto_source_identification,
            **kwargs,
        )

        # Initialize database manager with migration system
        self._db_manager = DatabaseManager(self.config)
        self._migration_manager = None

        # Initialize database with migration system
        self._initialize_database()

        # Initialize text processing components
        self._text_chunker = TextChunker(
            min_tokens=self.config.chunk_min_tokens,
            max_tokens=self.config.chunk_max_tokens,
            model_name=self.config.model_name,
        )

        # Other components will be initialized in later tasks
        self._classifier = None
        self._query_engine = None

    def _initialize_database(self):
        """Initialize database with migration system."""
        try:
            self._migration_manager = MigrationManager(self.config)
            init_results = self._migration_manager.initialize_database(
                create_tables=True
            )

            # Log initialization results (in production, use proper logging)
            if init_results.get("tables_created"):
                print(f"✅ Database initialized: {self.config.database_url}")

        except Exception as e:
            raise ConfigurationError(
                f"Failed to initialize database: {str(e)}",
                parameter="database_url",
                suggested_fix="Check database URL and permissions",
            )

    async def ingest_text(
        self,
        content: str,
        source_id: str,
        contextual_helper: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> IngestionResult:
        """Ingest and categorize text content with batch processing."""
        # Implementation will be added in later tasks
        raise NotImplementedError("Ingestion not yet implemented")

    async def query(
        self,
        query_text: str,
        strategy: QueryStrategy = QueryStrategy.ONE_SHOT,
        strategy_params: Optional[Dict] = None,
        prompt_limiting_strategy: PromptLimitingStrategy = PromptLimitingStrategy.PRUNE,
        max_token_budget: int = 4000,
        use_rankings: bool = True,
        limit: int = 10,
    ) -> QueryResult:
        """Query stored content using natural language with configurable strategies."""
        # Implementation will be added in later tasks
        raise NotImplementedError("Query not yet implemented")

    def get_category_tree(self) -> CategoryTree:
        """Retrieve the complete category hierarchy."""
        # Implementation will be added in later tasks
        raise NotImplementedError("Category tree retrieval not yet implemented")

    async def regenerate_contextual_helper(self, source_id: str) -> str:
        """Regenerate contextual helper for a source."""
        # Implementation will be added in later tasks
        raise NotImplementedError("Contextual helper regeneration not yet implemented")

    def get_database_info(self) -> Dict:
        """Get database and migration information."""
        if not self._migration_manager:
            return {"error": "Migration manager not initialized"}

        try:
            validation = self._migration_manager.validate_database_schema()
            table_info = self._db_manager.get_table_info()

            return {
                "database_url": self.config.database_url,
                "current_revision": validation.get("current_revision"),
                "schema_valid": validation.get("is_valid"),
                "tables_exist": validation.get("tables_exist"),
                "pending_migrations": validation.get("pending_migrations", []),
                "table_info": table_info,
            }
        except Exception as e:
            return {"error": str(e)}

    def chunk_text(
        self,
        content: str,
        source_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> List:
        """
        Chunk text using the configured text chunker.

        Args:
            content: Text content to chunk
            source_id: Optional source identifier
            metadata: Optional metadata to attach to chunks

        Returns:
            List of TextChunk objects
        """
        return self._text_chunker.chunk_text(
            content=content, source_id=source_id, metadata=metadata
        )

    def get_chunking_stats(self, chunks: List) -> Dict:
        """Get statistics about chunking results."""
        return self._text_chunker.get_chunking_stats(chunks)
