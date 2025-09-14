"""
Chunk retrieval and result construction for MemoirAI.

This module provides SQL-based chunk retrieval with deterministic ordering,
fallback logic, and comprehensive result object construction.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, asc, text
from pydantic import BaseModel

from ..database.models import Category, Chunk
from ..exceptions import ValidationError, DatabaseError
from .query_strategy_engine import CategoryPath, LLMCallResponse

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    """Represents a retrieved chunk with metadata."""

    chunk_id: int
    text_content: str
    category_path: str
    category_id_path: str
    ranked_relevance: int
    created_at: datetime
    source_id: Optional[str] = None
    token_count: Optional[int] = None


@dataclass
class PathRetrievalResult:
    """Result of retrieving chunks for a single category path."""

    category_path: CategoryPath
    chunks: List[ChunkResult]
    chunk_count: int
    retrieval_latency_ms: int
    error: Optional[str] = None
    success: bool = True


@dataclass
class QueryResult:
    """Complete query result with all metadata."""

    # Core results
    chunks: List[ChunkResult]
    responses: List[LLMCallResponse]

    # Metadata
    total_latency_ms: int
    total_chunks: int
    successful_paths: int
    failed_paths: int

    # Optional fields
    dropped_paths: Optional[List[str]] = None
    path_results: List[PathRetrievalResult] = field(default_factory=list)

    # Query context
    query_text: Optional[str] = None
    strategy_used: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class ChunkRetriever:
    """
    Handles SQL-based chunk retrieval with deterministic ordering.

    Features:
    - Deterministic ordering by created_at and chunk_id
    - Fallback logic for empty categories
    - Pagination support
    - Per-path limits
    - Comprehensive error handling
    - Performance tracking
    """

    def __init__(
        self,
        session: Session,
        default_limit: Optional[int] = None,
        enable_pagination: bool = True,
    ):
        """
        Initialize chunk retriever.

        Args:
            session: Database session
            default_limit: Default limit for chunk retrieval
            enable_pagination: Whether to support pagination
        """
        self.session = session
        self.default_limit = default_limit
        self.enable_pagination = enable_pagination

    def retrieve_chunks_for_paths(
        self,
        category_paths: List[CategoryPath],
        limit_per_path: Optional[int] = None,
        offset: int = 0,
    ) -> List[PathRetrievalResult]:
        """
        Retrieve chunks for multiple category paths.

        Args:
            category_paths: List of category paths to retrieve chunks for
            limit_per_path: Maximum chunks per path
            offset: Offset for pagination

        Returns:
            List of path retrieval results
        """
        if not category_paths:
            return []

        results = []

        for category_path in category_paths:
            start_time = datetime.now()

            try:
                chunks = self._retrieve_chunks_for_single_path(
                    category_path=category_path,
                    limit=limit_per_path or self.default_limit,
                    offset=offset,
                )

                # Calculate latency
                end_time = datetime.now()
                latency_ms = int((end_time - start_time).total_seconds() * 1000)

                # Create result
                result = PathRetrievalResult(
                    category_path=category_path,
                    chunks=chunks,
                    chunk_count=len(chunks),
                    retrieval_latency_ms=latency_ms,
                    success=True,
                )

                # Log empty results
                if not chunks:
                    logger.warning(
                        f"No chunks found for path: {category_path.path_string}"
                    )
                    result.error = "No chunks found in leaf category"

                results.append(result)

            except Exception as e:
                # Calculate latency even for errors
                end_time = datetime.now()
                latency_ms = int((end_time - start_time).total_seconds() * 1000)

                logger.error(
                    f"Error retrieving chunks for path {category_path.path_string}: {e}"
                )

                # Create error result
                error_result = PathRetrievalResult(
                    category_path=category_path,
                    chunks=[],
                    chunk_count=0,
                    retrieval_latency_ms=latency_ms,
                    success=False,
                    error=str(e),
                )

                results.append(error_result)

        return results

    def _retrieve_chunks_for_single_path(
        self,
        category_path: CategoryPath,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[ChunkResult]:
        """
        Retrieve chunks for a single category path.

        Args:
            category_path: Category path to retrieve chunks for
            limit: Maximum number of chunks to retrieve
            offset: Offset for pagination

        Returns:
            List of chunk results
        """
        if not category_path.path:
            return []

        # Get the leaf category
        leaf_category = category_path.leaf_category
        if not leaf_category:
            return []

        # Build the query using the enhanced view
        query = text(
            """
            SELECT 
                ch.id as chunk_id,
                ch.content as text_content,
                ch.token_count,
                ch.source_id,
                ch.created_at,
                cp.path as category_path,
                cp.id_path as category_id_path,
                cp.id as category_id
            FROM chunks ch
            JOIN (
                WITH RECURSIVE category_path AS (
                    SELECT id, name, level, parent_id, name as path, CAST(id AS TEXT) as id_path
                    FROM categories
                    WHERE parent_id IS NULL
                    
                    UNION ALL
                    
                    SELECT c.id, c.name, c.level, c.parent_id,
                           cp.path || ' → ' || c.name as path,
                           cp.id_path || '/' || CAST(c.id AS TEXT) as id_path
                    FROM categories c
                    JOIN category_path cp ON c.parent_id = cp.id
                )
                SELECT * FROM category_path
            ) cp ON ch.category_id = cp.id
            WHERE ch.category_id = :category_id
            ORDER BY ch.created_at ASC, ch.id ASC
        """
        )

        # Add limit and offset if specified
        if limit is not None:
            query = text(str(query) + f" LIMIT {limit}")
        if offset > 0:
            query = text(str(query) + f" OFFSET {offset}")

        try:
            # Execute query
            result = self.session.execute(query, {"category_id": leaf_category.id})

            # Convert to ChunkResult objects
            chunks = []
            for row in result:
                chunk_result = ChunkResult(
                    chunk_id=row.chunk_id,
                    text_content=row.text_content,
                    category_path=row.category_path,
                    category_id_path=row.category_id_path,
                    ranked_relevance=category_path.ranked_relevance,
                    created_at=row.created_at,
                    source_id=row.source_id,
                    token_count=row.token_count,
                )
                chunks.append(chunk_result)

            logger.debug(
                f"Retrieved {len(chunks)} chunks for category {leaf_category.name}"
            )

            return chunks

        except Exception as e:
            logger.error(f"SQL error retrieving chunks: {e}")
            raise DatabaseError(
                f"Failed to retrieve chunks for category {leaf_category.name}: {str(e)}",
                operation="select",
                table="chunks",
            )

    def get_chunk_count_for_path(self, category_path: CategoryPath) -> int:
        """
        Get the total number of chunks for a category path.

        Args:
            category_path: Category path to count chunks for

        Returns:
            Number of chunks in the leaf category
        """
        if not category_path.path:
            return 0

        leaf_category = category_path.leaf_category
        if not leaf_category:
            return 0

        try:
            count = (
                self.session.query(Chunk)
                .filter(Chunk.category_id == leaf_category.id)
                .count()
            )

            return count

        except Exception as e:
            logger.error(
                f"Error counting chunks for path {category_path.path_string}: {e}"
            )
            return 0


class ResultConstructor:
    """
    Constructs comprehensive query result objects with all required metadata.

    Features:
    - Aggregates chunks from multiple paths
    - Tracks LLM call responses
    - Calculates total latencies
    - Handles dropped paths
    - Provides comprehensive metadata
    """

    def __init__(self):
        """Initialize result constructor."""
        pass

    def construct_query_result(
        self,
        path_results: List[PathRetrievalResult],
        llm_responses: List[LLMCallResponse],
        query_text: Optional[str] = None,
        strategy_used: Optional[str] = None,
        dropped_paths: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
    ) -> QueryResult:
        """
        Construct a comprehensive query result.

        Args:
            path_results: Results from path retrieval
            llm_responses: LLM call responses
            query_text: Original query text
            strategy_used: Strategy that was used
            dropped_paths: Paths that were dropped
            start_time: Query start time for total latency calculation

        Returns:
            Complete query result object
        """
        # Aggregate all chunks
        all_chunks = []
        successful_paths = 0
        failed_paths = 0

        for path_result in path_results:
            all_chunks.extend(path_result.chunks)
            if path_result.success:
                successful_paths += 1
            else:
                failed_paths += 1

        # Calculate total latencies
        retrieval_latency = sum(result.retrieval_latency_ms for result in path_results)
        llm_latency = sum(response.latency_ms for response in llm_responses)
        total_latency = retrieval_latency + llm_latency

        # If start_time provided, calculate actual total latency
        if start_time:
            actual_total = int((datetime.now() - start_time).total_seconds() * 1000)
            total_latency = max(total_latency, actual_total)

        # Sort chunks deterministically (by created_at, then chunk_id)
        all_chunks.sort(key=lambda c: (c.created_at, c.chunk_id))

        # Construct result
        result = QueryResult(
            chunks=all_chunks,
            responses=llm_responses,
            total_latency_ms=total_latency,
            total_chunks=len(all_chunks),
            successful_paths=successful_paths,
            failed_paths=failed_paths,
            dropped_paths=dropped_paths,
            path_results=path_results,
            query_text=query_text,
            strategy_used=strategy_used,
        )

        logger.info(
            f"Constructed query result: {len(all_chunks)} chunks, "
            f"{successful_paths} successful paths, {failed_paths} failed paths, "
            f"{total_latency}ms total latency"
        )

        return result

    def validate_query_result(self, result: QueryResult) -> List[str]:
        """
        Validate a query result for completeness and consistency.

        Args:
            result: Query result to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check basic consistency
        if result.total_chunks != len(result.chunks):
            errors.append(
                f"Total chunks mismatch: {result.total_chunks} != {len(result.chunks)}"
            )

        if result.successful_paths + result.failed_paths != len(result.path_results):
            errors.append(
                f"Path count mismatch: {result.successful_paths + result.failed_paths} "
                f"!= {len(result.path_results)}"
            )

        # Check chunk ordering
        if len(result.chunks) > 1:
            for i in range(1, len(result.chunks)):
                prev_chunk = result.chunks[i - 1]
                curr_chunk = result.chunks[i]

                if prev_chunk.created_at > curr_chunk.created_at or (
                    prev_chunk.created_at == curr_chunk.created_at
                    and prev_chunk.chunk_id > curr_chunk.chunk_id
                ):
                    errors.append(
                        f"Chunks not properly ordered at index {i}: "
                        f"{prev_chunk.chunk_id} -> {curr_chunk.chunk_id}"
                    )
                    break

        # Check for required fields
        for i, chunk in enumerate(result.chunks):
            if not chunk.text_content:
                errors.append(f"Chunk {i} missing text_content")
            if not chunk.category_path:
                errors.append(f"Chunk {i} missing category_path")

        return errors


# Utility functions
def create_chunk_retriever(
    session: Session, default_limit: Optional[int] = 100, **kwargs
) -> ChunkRetriever:
    """
    Create a chunk retriever with default configuration.

    Args:
        session: Database session
        default_limit: Default limit for chunk retrieval
        **kwargs: Additional configuration options

    Returns:
        Configured ChunkRetriever
    """
    return ChunkRetriever(session=session, default_limit=default_limit, **kwargs)


def create_result_constructor(**kwargs) -> ResultConstructor:
    """
    Create a result constructor.

    Args:
        **kwargs: Configuration options (for future extensibility)

    Returns:
        Configured ResultConstructor
    """
    return ResultConstructor(**kwargs)
