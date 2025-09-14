"""
Complete query processing pipeline for MemoirAI.

This module integrates query strategy execution with chunk retrieval
to provide a complete natural language query processing system.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from sqlalchemy.orm import Session

from .query_strategy_engine import (
    QueryStrategyEngine,
    QueryStrategy,
    QueryExecutionResult,
)
from .chunk_retrieval import ChunkRetriever, ResultConstructor, QueryResult
from ..classification.category_manager import CategoryManager
from ..exceptions import ValidationError, ClassificationError

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Complete query processing pipeline.

    Integrates query strategy execution with chunk retrieval to provide
    end-to-end natural language query processing.

    Features:
    - Strategy-based category traversal
    - SQL-based chunk retrieval
    - Comprehensive result construction
    - Error handling and fallback logic
    - Performance tracking
    """

    def __init__(
        self,
        category_manager: CategoryManager,
        session: Session,
        model_name: str = "openai:gpt-4",
        default_chunk_limit: Optional[int] = 100,
    ):
        """
        Initialize query processor.

        Args:
            category_manager: Category hierarchy manager
            session: Database session
            model_name: LLM model for query classification
            default_chunk_limit: Default limit for chunk retrieval
        """
        self.category_manager = category_manager
        self.session = session
        self.model_name = model_name
        self.default_chunk_limit = default_chunk_limit

        # Initialize components
        self.strategy_engine = QueryStrategyEngine(
            category_manager=category_manager, model_name=model_name
        )

        self.chunk_retriever = ChunkRetriever(
            session=session, default_limit=default_chunk_limit
        )

        self.result_constructor = ResultConstructor()

    async def process_query(
        self,
        query_text: str,
        strategy: QueryStrategy = QueryStrategy.ONE_SHOT,
        strategy_params: Optional[Dict[str, Any]] = None,
        contextual_helper: Optional[str] = None,
        chunk_limit_per_path: Optional[int] = None,
        offset: int = 0,
    ) -> QueryResult:
        """
        Process a complete natural language query.

        Args:
            query_text: Natural language query
            strategy: Query traversal strategy
            strategy_params: Parameters for the strategy
            contextual_helper: Additional context for classification
            chunk_limit_per_path: Maximum chunks per category path
            offset: Offset for pagination

        Returns:
            Complete query result with chunks and metadata
        """
        start_time = datetime.now()

        logger.info(f"Processing query: '{query_text[:50]}...' using {strategy.value}")

        try:
            # Step 1: Execute query strategy to get category paths
            strategy_result = await self.strategy_engine.execute_strategy(
                query_text=query_text,
                strategy=strategy,
                strategy_params=strategy_params or {},
                contextual_helper=contextual_helper,
            )

            logger.info(
                f"Strategy execution completed: {len(strategy_result.category_paths)} paths, "
                f"{len(strategy_result.llm_responses)} LLM calls"
            )

            # Step 2: Retrieve chunks for each category path
            path_results = self.chunk_retriever.retrieve_chunks_for_paths(
                category_paths=strategy_result.category_paths,
                limit_per_path=chunk_limit_per_path or self.default_chunk_limit,
                offset=offset,
            )

            logger.info(
                f"Chunk retrieval completed: {sum(len(r.chunks) for r in path_results)} total chunks"
            )

            # Step 3: Construct comprehensive result
            query_result = self.result_constructor.construct_query_result(
                path_results=path_results,
                llm_responses=strategy_result.llm_responses,
                query_text=query_text,
                strategy_used=strategy.value,
                start_time=start_time,
            )

            # Step 4: Validate result
            validation_errors = self.result_constructor.validate_query_result(
                query_result
            )
            if validation_errors:
                logger.warning(f"Query result validation errors: {validation_errors}")

            logger.info(
                f"Query processing completed: {query_result.total_chunks} chunks, "
                f"{query_result.total_latency_ms}ms total latency"
            )

            return query_result

        except Exception as e:
            logger.error(f"Error processing query '{query_text}': {e}")

            # Create error result
            end_time = datetime.now()
            error_latency = int((end_time - start_time).total_seconds() * 1000)

            error_result = QueryResult(
                chunks=[],
                responses=[],
                total_latency_ms=error_latency,
                total_chunks=0,
                successful_paths=0,
                failed_paths=1,
                query_text=query_text,
                strategy_used=strategy.value if strategy else None,
                dropped_paths=[f"Error: {str(e)}"],
            )

            return error_result

    def get_query_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about query processing.

        Returns:
            Dictionary with query processing statistics
        """
        # This would track statistics across multiple queries
        # For now, return basic information
        return {
            "model_name": self.model_name,
            "default_chunk_limit": self.default_chunk_limit,
            "hierarchy_depth": self.category_manager.hierarchy_depth,
            "available_strategies": [strategy.value for strategy in QueryStrategy],
        }

    def validate_query_parameters(
        self,
        query_text: str,
        strategy: QueryStrategy,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Validate query parameters before processing.

        Args:
            query_text: Query text to validate
            strategy: Strategy to validate
            strategy_params: Strategy parameters to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate query text
        if not query_text or not query_text.strip():
            errors.append("Query text cannot be empty")

        if len(query_text.strip()) > 1000:
            errors.append("Query text too long (max 1000 characters)")

        # Validate strategy
        if not isinstance(strategy, QueryStrategy):
            errors.append(f"Invalid strategy type: {type(strategy)}")

        # Validate strategy parameters
        if strategy_params:
            try:
                from .query_strategy_engine import validate_strategy_params

                validate_strategy_params(strategy, strategy_params)
            except ValidationError as e:
                errors.append(f"Invalid strategy parameters: {str(e)}")

        return errors


# Utility functions
def create_query_processor(
    category_manager: CategoryManager,
    session: Session,
    model_name: str = "openai:gpt-4",
    **kwargs,
) -> QueryProcessor:
    """
    Create a query processor with default configuration.

    Args:
        category_manager: Category hierarchy manager
        session: Database session
        model_name: LLM model name
        **kwargs: Additional configuration options

    Returns:
        Configured QueryProcessor
    """
    return QueryProcessor(
        category_manager=category_manager,
        session=session,
        model_name=model_name,
        **kwargs,
    )


async def process_natural_language_query(
    query_text: str,
    category_manager: CategoryManager,
    session: Session,
    strategy: QueryStrategy = QueryStrategy.ONE_SHOT,
    **kwargs,
) -> QueryResult:
    """
    Convenience function for processing a natural language query.

    Args:
        query_text: Natural language query
        category_manager: Category hierarchy manager
        session: Database session
        strategy: Query strategy to use
        **kwargs: Additional parameters

    Returns:
        Complete query result
    """
    processor = create_query_processor(
        category_manager=category_manager, session=session
    )

    return await processor.process_query(
        query_text=query_text, strategy=strategy, **kwargs
    )
