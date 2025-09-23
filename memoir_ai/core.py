"""
Core MemoirAI class providing the main public API.

This module provides the primary interface for the MemoirAI library,
integrating all components for text ingestion and retrieval.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from types import TracebackType
from typing import Any, Dict, List, Optional, Union

from sqlalchemy.orm import Session

from .classification.category_manager import CategoryManager
from .database.engine import DatabaseManager
from .database.models import Category, Chunk, ContextualHelper
from .text_processing.chunker import TextChunk, TextChunker

# Import the implemented iterative classification workflow.
try:  # pragma: no cover - defensive import guard
    from .classification.iterative_classifier import IterativeClassificationWorkflow
except Exception:  # Fallback if something goes wrong unexpectedly
    IterativeClassificationWorkflow = None  # type: ignore
from .aggregation.result_aggregator import PromptLimitingStrategy, ResultAggregator
from .config import MemoirAIConfig
from .exceptions import (
    ClassificationError,
    ConfigurationError,
    DatabaseError,
    MemoirAIError,
    ValidationError,
)
from .query.query_processor import QueryProcessor, process_natural_language_query
from .query.query_strategy_engine import QueryStrategy

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of text ingestion operation."""

    # Success metrics
    success: bool
    chunks_processed: int
    chunks_stored: int
    categories_created: int

    # Processing details
    source_id: str
    processing_time_ms: int

    # Content information
    total_tokens: int
    chunk_details: List[Dict[str, Any]] = field(default_factory=list)

    # Error information
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CategoryTree:
    """Hierarchical representation of the category tree."""

    categories: List[Dict[str, Any]]
    total_categories: int
    max_depth: int
    categories_by_level: Dict[int, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "categories": self.categories,
            "total_categories": self.total_categories,
            "max_depth": self.max_depth,
            "categories_by_level": self.categories_by_level,
        }


class MemoirAI:
    """
    Main MemoirAI class providing text ingestion and retrieval capabilities.

    Features:
    - Text chunking and classification
    - Hierarchical category management
    - Natural language querying with multiple strategies
    - Token budget management and pruning
    - Transaction management for data consistency
    - Comprehensive error handling and validation
    """

    def __init__(
        self,
        database_url: str,
        llm_provider: str = "openai",
        hierarchy_depth: int = 3,
        chunk_min_tokens: int = 300,
        chunk_max_tokens: int = 500,
        model_name: str = "gpt-4o-mini",
        batch_size: int = 5,
        max_categories_per_level: Union[int, Dict[int, int]] = 128,
        auto_source_identification: bool = True,
        max_token_budget: int = 40000,
        prompt_limiting_strategy: PromptLimitingStrategy = PromptLimitingStrategy.PRUNE,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MemoirAI with configuration parameters.

        Args:
            database_url: Database connection URL
            llm_provider: LLM provider (e.g., "openai", "anthropic")
            hierarchy_depth: Maximum category hierarchy depth (1-100)
            chunk_min_tokens: Minimum tokens per chunk
            chunk_max_tokens: Maximum tokens per chunk
            model_name: LLM model name
            batch_size: Batch size for processing (1-50)
            max_categories_per_level: Category limits per level
            auto_source_identification: Whether to auto-identify sources
            max_token_budget: Maximum token budget for queries
            prompt_limiting_strategy: Strategy for handling token limits
            **kwargs: Additional configuration options
        """
        # Store configuration
        self.database_url = database_url
        self.llm_provider = llm_provider
        self.hierarchy_depth = hierarchy_depth
        self.chunk_min_tokens = chunk_min_tokens
        self.chunk_max_tokens = chunk_max_tokens
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_categories_per_level = max_categories_per_level
        self.auto_source_identification = auto_source_identification
        self.max_token_budget = max_token_budget
        self.prompt_limiting_strategy = prompt_limiting_strategy

        # Validate configuration
        self._validate_configuration()

        # Initialize database configuration (uses comprehensive validation)
        self.config = MemoirAIConfig(
            database_url=database_url,
            llm_provider=self.llm_provider,
            model_name=self.model_name,
            hierarchy_depth=self.hierarchy_depth,
            max_categories_per_level=self.max_categories_per_level,
            chunk_min_tokens=self.chunk_min_tokens,
            chunk_max_tokens=self.chunk_max_tokens,
            batch_size=self.batch_size,
            auto_source_identification=self.auto_source_identification,
            max_token_budget=self.max_token_budget,
        )
        self.db_manager = DatabaseManager(self.config)

        # Initialize components
        self._initialize_components()

        logger.info(
            f"MemoirAI initialized with model {model_name}, hierarchy depth {hierarchy_depth}"
        )

    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        errors = []

        # Validate hierarchy depth
        if not 1 <= self.hierarchy_depth <= 100:
            errors.append("hierarchy_depth must be between 1 and 100 inclusive")

        # Validate batch size
        if not 1 <= self.batch_size <= 50:
            errors.append("batch_size must be between 1 and 50 inclusive")

        # Validate token thresholds
        if self.chunk_min_tokens >= self.chunk_max_tokens:
            errors.append("chunk_min_tokens must be less than chunk_max_tokens")

        # Validate token budget
        min_required_budget = self.chunk_min_tokens + 100  # Reasonable overhead
        if self.max_token_budget <= min_required_budget:
            errors.append(
                f"max_token_budget ({self.max_token_budget}) must be greater than "
                f"chunk_min_tokens ({self.chunk_min_tokens}) plus overhead (100 tokens). "
                f"Minimum required: {min_required_budget}"
            )

        # Validate category limits
        if isinstance(self.max_categories_per_level, int):
            if self.max_categories_per_level <= 0:
                errors.append("max_categories_per_level must be a positive integer")
        elif isinstance(self.max_categories_per_level, dict):
            for level, limit in self.max_categories_per_level.items():
                if not isinstance(level, int) or not 1 <= level <= self.hierarchy_depth:
                    errors.append(
                        f"Category limit level {level} must be between 1 and {self.hierarchy_depth}"
                    )
                if not isinstance(limit, int) or limit <= 0:
                    errors.append(
                        f"Category limit for level {level} must be a positive integer"
                    )
        else:
            errors.append("max_categories_per_level must be an integer or dictionary")

        # Validate database URL
        if not self.database_url or not isinstance(self.database_url, str):
            errors.append("database_url must be a non-empty string")

        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(
                f"- {error}" for error in errors
            )
            raise ConfigurationError(error_message)

    def _initialize_components(self) -> None:
        """Initialize all internal components."""
        try:
            # Get database session
            with self.db_manager.get_session() as session:
                # Initialize text chunker
                self.text_chunker = TextChunker(
                    min_tokens=self.chunk_min_tokens,
                    max_tokens=self.chunk_max_tokens,
                    model_name=self.model_name,
                )

                # Initialize category manager
                self.category_manager = CategoryManager(
                    db_session=session,
                    hierarchy_depth=self.hierarchy_depth,
                    category_limits=self.max_categories_per_level,
                )

                # Initialize iterative classification workflow (requirement 5.2)
                self.iterative_classifier: Optional[IterativeClassificationWorkflow]
                if IterativeClassificationWorkflow is not None:
                    self.iterative_classifier = IterativeClassificationWorkflow(
                        session,
                        self.category_manager,
                        model_name=self.model_name,
                        batch_size=self.batch_size,
                    )
                else:  # pragma: no cover - unexpected
                    self.iterative_classifier = None

                # Initialize query processor
                self.query_processor = QueryProcessor(
                    category_manager=self.category_manager,
                    session=session,
                    model_name=self.model_name,
                )

                # Initialize result aggregator
                from .aggregation.result_aggregator import create_result_aggregator

                self.result_aggregator = create_result_aggregator(
                    max_token_budget=self.max_token_budget,
                    strategy=self.prompt_limiting_strategy,
                    model_name=self.model_name,
                )

        except Exception as e:
            logger.error(f"Failed to initialize MemoirAI components: {e}")
            raise ConfigurationError(f"Component initialization failed: {str(e)}")

    async def ingest_text(
        self,
        content: str,
        source_id: str,
        contextual_helper: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestionResult:
        """
        Ingest and categorize text content with batch processing.

        Args:
            content: Text content to ingest
            source_id: Unique identifier for the source
            contextual_helper: Optional contextual information
            metadata: Optional metadata dictionary

        Returns:
            IngestionResult with processing details and metrics
        """
        start_time = datetime.now()

        logger.info(
            f"Starting ingestion for source '{source_id}', content length: {len(content)}"
        )

        try:
            # Validate inputs
            if not content or not content.strip():
                raise ValidationError(
                    "Content cannot be empty", field="content", value=content
                )

            if not source_id or not source_id.strip():
                raise ValidationError(
                    "Source ID cannot be empty", field="source_id", value=source_id
                )

            # Start database transaction
            with self.db_manager.get_session() as session:
                try:
                    # Step 1: Chunk the text
                    chunks = self.text_chunker.chunk_text(content, source_id=source_id)

                    if not chunks:
                        logger.warning(f"No chunks generated for source '{source_id}'")
                        return IngestionResult(
                            success=True,
                            chunks_processed=0,
                            chunks_stored=0,
                            categories_created=0,
                            source_id=source_id,
                            processing_time_ms=0,
                            total_tokens=0,
                            warnings=["No chunks generated from content"],
                        )

                    logger.info(
                        f"Generated {len(chunks)} chunks for source '{source_id}'"
                    )

                    # Step 2: Classify chunks
                    if not self.iterative_classifier:
                        raise ClassificationError(
                            "Iterative classification workflow is not initialized",
                            retry_count=0,
                        )

                    classification_results = (
                        await self.iterative_classifier.classify_chunks(
                            chunks=chunks,
                            contextual_helper=contextual_helper
                            or f"Content from source: {source_id}",
                            source_id=source_id,
                        )
                    )

                    logger.info(f"Classified {len(classification_results)} chunks")

                    # Step 3: Store chunks and classifications
                    chunks_stored = 0
                    categories_created = 0
                    chunk_details = []
                    total_tokens = 0

                    for i, result in enumerate(classification_results):
                        chunk = result.chunk
                        if (
                            result.success
                            and result.category_path
                            and result.final_category
                        ):
                            leaf_category = result.final_category

                            # Create chunk record mirroring storage in workflow (idempotent within txn)
                            chunk_record = Chunk(
                                content=chunk.content,
                                token_count=chunk.token_count,
                                source_id=source_id,
                                category_id=leaf_category.id,
                                created_at=datetime.now(),
                            )
                            session.add(chunk_record)
                            chunks_stored += 1
                            total_tokens += chunk.token_count

                            chunk_details.append(
                                {
                                    "chunk_index": i,
                                    "token_count": chunk.token_count,
                                    "category_path": " > ".join(
                                        cat.name for cat in result.category_path
                                    ),
                                    "category_id": leaf_category.id,
                                }
                            )
                        else:
                            logger.warning(
                                f"Failed to classify chunk {i}: {result.error}"
                            )

                    # Store contextual helper if provided
                    if contextual_helper:
                        helper_record = ContextualHelper(
                            source_id=source_id,
                            helper_text=contextual_helper,
                            created_at=datetime.now(),
                        )
                        session.add(helper_record)

                    # Commit transaction
                    session.commit()

                    # Calculate processing time
                    end_time = datetime.now()
                    processing_time_ms = int(
                        (end_time - start_time).total_seconds() * 1000
                    )

                    logger.info(
                        f"Ingestion completed for '{source_id}': {chunks_stored}/{len(chunks)} chunks stored, "
                        f"{processing_time_ms}ms"
                    )

                    return IngestionResult(
                        success=True,
                        chunks_processed=len(chunks),
                        chunks_stored=chunks_stored,
                        categories_created=categories_created,
                        source_id=source_id,
                        processing_time_ms=processing_time_ms,
                        total_tokens=total_tokens,
                        chunk_details=chunk_details,
                    )

                except Exception as e:
                    # Rollback transaction on error
                    session.rollback()
                    logger.error(
                        f"Database transaction failed for source '{source_id}': {e}"
                    )
                    raise DatabaseError(
                        f"Ingestion transaction failed: {str(e)}",
                        operation="ingest_text",
                        table="chunks",
                    )

        except Exception as e:
            # Calculate processing time even for errors
            end_time = datetime.now()
            processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

            logger.error(f"Ingestion failed for source '{source_id}': {e}")

            return IngestionResult(
                success=False,
                chunks_processed=0,
                chunks_stored=0,
                categories_created=0,
                source_id=source_id,
                processing_time_ms=processing_time_ms,
                total_tokens=0,
                error_message=str(e),
            )

    async def query(
        self,
        query_text: str,
        strategy: QueryStrategy = QueryStrategy.ONE_SHOT,
        strategy_params: Optional[Dict[str, Any]] = None,
        prompt_limiting_strategy: Optional[PromptLimitingStrategy] = None,
        max_token_budget: Optional[int] = None,
        use_rankings: Optional[bool] = None,
        limit: int = 10,
        contextual_helper: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query stored content using natural language with configurable strategies.

        Args:
            query_text: Natural language query
            strategy: Query traversal strategy
            strategy_params: Parameters for the strategy
            prompt_limiting_strategy: Strategy for handling token limits
            max_token_budget: Maximum token budget (overrides default)
            use_rankings: Whether to use ranking-based pruning
            limit: Maximum number of results to return
            contextual_helper: Optional contextual information

        Returns:
            Dictionary with query results and metadata
        """
        start_time = datetime.now()

        logger.info(f"Processing query: '{query_text[:50]}...' using {strategy.value}")

        try:
            # Validate inputs
            if not query_text or not query_text.strip():
                raise ValidationError(
                    "Query text cannot be empty", field="query_text", value=query_text
                )

            # Use instance defaults if not provided
            budget = max_token_budget or self.max_token_budget
            limiting_strategy = (
                prompt_limiting_strategy or self.prompt_limiting_strategy
            )

            # Get database session
            with self.db_manager.get_session() as session:
                # Process query using query processor
                query_result = await self.query_processor.process_query(
                    query_text=query_text,
                    strategy=strategy,
                    strategy_params=strategy_params or {},
                    contextual_helper=contextual_helper,
                    chunk_limit_per_path=limit,
                )

                # Apply result aggregation with budget management
                aggregation_result = await self.result_aggregator.aggregate_results(
                    query_result=query_result,
                    query_text=query_text,
                    contextual_helper=contextual_helper,
                )

                # Calculate total processing time
                end_time = datetime.now()
                total_latency_ms = int((end_time - start_time).total_seconds() * 1000)

                # Construct response
                response: Dict[str, Any] = {
                    "success": True,
                    "query": query_text,
                    "strategy": strategy.value,
                    "chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "text_content": chunk.text_content,
                            "category_path": chunk.category_path,
                            "ranked_relevance": chunk.ranked_relevance,
                            "created_at": chunk.created_at.isoformat(),
                            "source_id": chunk.source_id,
                        }
                        for chunk in aggregation_result.final_chunks
                    ],
                    "metadata": {
                        "total_chunks": len(aggregation_result.final_chunks),
                        "total_latency_ms": total_latency_ms,
                        "within_budget": aggregation_result.within_budget,
                        "strategy_used": aggregation_result.strategy_used.value,
                        "token_estimate": {
                            "total_tokens": aggregation_result.token_estimate.total_tokens,
                            "fixed_prompt_tokens": aggregation_result.token_estimate.fixed_prompt_tokens,
                            "chunks_tokens": aggregation_result.token_estimate.chunks_total_tokens,
                        },
                        "llm_responses": len(query_result.responses),
                        "successful_paths": query_result.successful_paths,
                        "failed_paths": query_result.failed_paths,
                    },
                }

                # Add pruning information if applicable
                if aggregation_result.pruning_result:
                    response["metadata"]["pruning"] = {
                        "chunks_dropped": aggregation_result.pruning_result.dropped_count,
                        "pruning_ratio": aggregation_result.pruning_result.pruning_ratio,
                        "dropped_paths": aggregation_result.dropped_paths,
                    }

                # Add warnings and errors
                if aggregation_result.warnings:
                    response["warnings"] = aggregation_result.warnings

                if aggregation_result.error_message:
                    response["error"] = aggregation_result.error_message

                logger.info(
                    f"Query completed: {len(aggregation_result.final_chunks)} chunks, "
                    f"{total_latency_ms}ms"
                )

                return response

        except Exception as e:
            logger.error(f"Query failed: {e}")

            # Calculate processing time even for errors
            end_time = datetime.now()
            error_latency_ms = int((end_time - start_time).total_seconds() * 1000)

            return {
                "success": False,
                "query": query_text,
                "error": str(e),
                "chunks": [],
                "metadata": {
                    "total_chunks": 0,
                    "total_latency_ms": error_latency_ms,
                    "error_type": type(e).__name__,
                },
            }

    def get_category_tree(self) -> CategoryTree:
        """
        Retrieve the complete category hierarchy.

        Returns:
            CategoryTree with hierarchical category structure
        """
        logger.info("Retrieving category tree")

        try:
            with self.db_manager.get_session() as session:
                # Get category statistics
                stats = self.category_manager.get_category_stats()

                # Get all categories organized by level
                categories_by_level: Dict[int, List[Dict[str, Any]]] = {}
                for level in range(1, self.hierarchy_depth + 1):
                    level_categories = self.category_manager.get_existing_categories(
                        level
                    )
                    if level_categories:
                        categories_by_level[level] = [
                            {
                                "id": cat.id,
                                "name": cat.name,
                                "level": cat.level,
                                "parent_id": cat.parent_id,
                                "created_at": (
                                    cat.created_at.isoformat()
                                    if cat.created_at
                                    else None
                                ),
                            }
                            for cat in level_categories
                        ]

                # Build hierarchical structure
                def build_hierarchy(
                    parent_id: Optional[int] = None, level: int = 1
                ) -> List[Dict[str, Any]]:
                    if level > self.hierarchy_depth or level not in categories_by_level:
                        return []

                    result: List[Dict[str, Any]] = []
                    for cat in categories_by_level[level]:
                        if cat["parent_id"] == parent_id:
                            cat_with_children = cat.copy()
                            cat_with_children["children"] = build_hierarchy(
                                cat["id"], level + 1
                            )
                            result.append(cat_with_children)

                    return result

                hierarchical_categories = build_hierarchy()

                return CategoryTree(
                    categories=hierarchical_categories,
                    total_categories=stats.total_categories,
                    max_depth=stats.max_depth,
                    categories_by_level=stats.categories_by_level,
                )

        except Exception as e:
            logger.error(f"Failed to retrieve category tree: {e}")
            raise DatabaseError(
                f"Failed to retrieve category tree: {str(e)}",
                operation="get_category_tree",
                table="categories",
            )

    async def regenerate_contextual_helper(self, source_id: str) -> str:
        """
        Regenerate contextual helper for a source.

        Args:
            source_id: Source identifier

        Returns:
            Generated contextual helper text
        """
        logger.info(f"Regenerating contextual helper for source '{source_id}'")

        try:
            with self.db_manager.get_session() as session:
                # Get existing chunks for the source
                chunks = session.query(Chunk).filter(Chunk.source_id == source_id).all()

                if not chunks:
                    raise ValidationError(
                        f"No chunks found for source '{source_id}'",
                        field="source_id",
                        value=source_id,
                    )

                # Combine chunk content for analysis
                combined_content = "\n\n".join(
                    chunk.content for chunk in chunks[:10]
                )  # Limit for performance

                # Generate contextual helper using LLM
                # This is a simplified implementation - in practice, you'd use a more sophisticated approach
                contextual_helper = f"Content analysis for source {source_id}: Contains {len(chunks)} chunks covering various topics."

                # Update or create contextual helper record
                existing_helper = (
                    session.query(ContextualHelper)
                    .filter(ContextualHelper.source_id == source_id)
                    .first()
                )

                if existing_helper:
                    existing_helper.helper_text = contextual_helper
                    existing_helper.created_at = datetime.now()
                else:
                    helper_record = ContextualHelper(
                        source_id=source_id,
                        helper_text=contextual_helper,
                        created_at=datetime.now(),
                    )
                    session.add(helper_record)

                session.commit()

                logger.info(f"Regenerated contextual helper for source '{source_id}'")
                return contextual_helper

        except Exception as e:
            logger.error(
                f"Failed to regenerate contextual helper for '{source_id}': {e}"
            )
            raise ClassificationError(
                f"Failed to regenerate contextual helper: {str(e)}",
                model=self.model_name,
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the MemoirAI instance.

        Returns:
            Dictionary with system statistics
        """
        try:
            with self.db_manager.get_session() as session:
                # Get database info
                db_info = self.db_manager.get_table_info()

                # Get category statistics
                category_stats = self.category_manager.get_category_stats()

                return {
                    "configuration": {
                        "hierarchy_depth": self.hierarchy_depth,
                        "chunk_min_tokens": self.chunk_min_tokens,
                        "chunk_max_tokens": self.chunk_max_tokens,
                        "model_name": self.model_name,
                        "batch_size": self.batch_size,
                        "max_token_budget": self.max_token_budget,
                        "prompt_limiting_strategy": self.prompt_limiting_strategy.value,
                    },
                    "database": db_info,
                    "categories": {
                        "total_categories": category_stats.total_categories,
                        "max_depth": category_stats.max_depth,
                        "categories_by_level": category_stats.categories_by_level,
                        "leaf_categories": category_stats.leaf_categories,
                    },
                    "components": {
                        "text_chunker": "initialized",
                        "category_manager": "initialized",
                        "iterative_classifier": "initialized",
                        "query_processor": "initialized",
                        "result_aggregator": "initialized",
                    },
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                "error": str(e),
                "configuration": {
                    "hierarchy_depth": self.hierarchy_depth,
                    "model_name": self.model_name,
                },
            }

    def close(self) -> None:
        """Close database connections and cleanup resources."""
        try:
            if hasattr(self, "db_manager") and self.db_manager:
                self.db_manager.close()
            logger.info("MemoirAI instance closed successfully")
        except Exception as e:
            logger.error(f"Error closing MemoirAI instance: {e}")

    def __enter__(self) -> "MemoirAI":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Context manager exit."""
        self.close()


# Utility functions for easy initialization
def create_memoir_ai(
    database_url: str,
    model_name: str = "gpt-4o-mini",
    hierarchy_depth: int = 3,
    **kwargs: Any,
) -> MemoirAI:
    """
    Create a MemoirAI instance with simplified configuration.

    Args:
        database_url: Database connection URL
        model_name: LLM model name
        hierarchy_depth: Category hierarchy depth
        **kwargs: Additional configuration options

    Returns:
        Configured MemoirAI instance
    """
    return MemoirAI(
        database_url=database_url,
        model_name=model_name,
        hierarchy_depth=hierarchy_depth,
        **kwargs,
    )
