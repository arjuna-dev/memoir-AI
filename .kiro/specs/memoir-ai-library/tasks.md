# Implementation Plan

- [ ] 1. Set up project structure and core dependencies

  - Create Python package structure with poetry/setuptools configuration
  - Add core dependencies: SQLAlchemy, Pydantic AI, liteLLM, asyncio support
  - Set up development dependencies: pytest, black, mypy, pre-commit hooks
  - Create basic configuration management for database and LLM settings
  - _Requirements: 1.1, 1.2, 1.4_

- [ ] 2.1 Create SQLAlchemy database models

  - Implement Category model with hierarchy constraints and validation
  - Implement Chunk model with token counting and source tracking
  - Implement ContextualHelper model for source context storage
  - Implement CategoryLimits model for configurable category limits per level
  - Add all required indexes and constraints as specified in design
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9_

- [ ] 2.2 Create database migration and initialization system

  - Implement Alembic migration setup for schema versioning
  - Create database initialization logic that auto-creates tables and indexes
  - Add database connection management with retry logic and error handling
  - Write unit tests for database models and constraints validation
  - _Requirements: 1.3, 1.5, 7.5_

- [ ] 3.1 Create token-based text chunker

  - Implement TextChunker class using liteLLM's token_counter function
  - Add configurable delimiters and token size thresholds (300-500 tokens default)
  - Implement paragraph boundary preservation logic
  - Add chunk merging for undersized chunks and splitting for oversized chunks
  - Write comprehensive unit tests for various text inputs and edge cases
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 7.1, 7.2_

  - [ ] 3.2 Implement contextual helper generation system

    - Create ContextualHelperGenerator with auto and manual modes
    - Implement automatic helper generation from metadata and first chunks
    - Add user input collection system for manual helper creation with validation
    - Implement ISO 8601 date validation and field requirement enforcement
    - Add helper storage, versioning, and regeneration capabilities
    - Write unit tests for both automatic and manual helper generation modes
    - _Requirements: 2B.1, 2B.2, 2B.3, 2B.4, 2B.5, 2B.6, 2B.7, 2B.8, 2B.9, 2B.10_

  - [ ] 4.1 Implement Pydantic AI schemas and agents

    - Create Pydantic models for all LLM interactions (classification, summarization, final answers)
    - Implement native structured output support with fallback for unsupported models
    - Create agent factory functions for different LLM providers and output types
    - Add schema validation and error handling for malformed LLM responses
    - Write unit tests with mocked LLM responses to validate schema enforcement
    - _Requirements: 5.1, 5.2, 5.3_

  - [ ] 4.2 Implement batch classification system

    - Create BatchCategoryClassifier for processing multiple chunks in single LLM calls
    - Implement structured batch prompts with chunk IDs and existing category presentation
    - Add batch response validation and individual chunk retry logic for failures
    - Implement configurable batch_size parameter (default 5) with overflow handling
    - Add comprehensive logging for batch processing metrics and performance
    - Write unit tests for batch processing, validation, and retry mechanisms
    - _Requirements: 2A.1, 2A.2, 2A.3, 2A.4, 2A.5, 2A.6, 2A.7, 2A.8, 2A.9, 2A.10_

  - [ ] 5.1 Create category hierarchy management

    - Implement CategoryManager with configurable limits (global and per-level)
    - Add category creation validation with hierarchy depth and parent relationship checks
    - Implement category limit enforcement that forces selection from existing categories when limits reached
    - Add category retrieval methods for presenting existing options to LLM
    - Write unit tests for category creation, validation, and limit enforcement
    - _Requirements: 2.8, 2.9, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ] 5.2 Implement iterative classification workflow

    - Create classification workflow that processes chunks through hierarchy levels sequentially
    - Integrate contextual helpers into classification prompts at each level
    - Implement category reuse logic by presenting existing categories to LLM
    - Add classification result storage linking chunks to leaf-level categories only
    - Write integration tests for complete classification workflow from chunk to storage
    - _Requirements: 2.5, 2.6, 2.7, 2.10, 2.11_

  - [ ] 6.1 Implement query classification and traversal strategies

    - Create QueryStrategyEngine with four configurable strategies (one shot, wide branch, zoom in, branch out)
    - Implement LLM-based query classification with ranked relevance scoring
    - Add category path construction and validation for each strategy
    - Implement deterministic path ordering and deduplication logic
    - Write unit tests for each strategy with various hierarchy configurations
    - _Requirements: 4.1, 8 (all sub-requirements for retrieval strategies)_

  - [ ] 6.2 Implement chunk retrieval and result construction

    - Create SQL retrieval system for chunks by category paths with deterministic ordering
    - Implement fallback logic when leaf categories contain no chunks
    - Add per-call response object construction with timestamps and latency tracking
    - Create query result object assembly with all required metadata fields
    - Write integration tests for complete query workflow from natural language to results
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [ ] 7.1 Create token budget estimation and pruning system

    - Implement ResultAggregator with token counting using liteLLM for budget estimation
    - Create PruningEngine with ranking-based and deterministic order pruning strategies
    - Add budget validation and dropped paths tracking for pruned results
    - Implement error handling when all paths are dropped but budget still exceeded
    - Write unit tests for budget estimation accuracy and pruning logic
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [ ] 7.2 Create summarization system with compression ratio management

    - Implement SummarizationEngine with configurable compression parameters
    - Add chunk partitioning logic for token-safe summarization parts
    - Create per-part summarization with structured LLM prompts and validation
    - Implement character budget validation and retry logic for oversized summaries
    - Add final prompt construction and token budget verification
    - Write comprehensive tests for summarization accuracy and budget compliance
    - _Requirements: 9.7, 9.8, 9.9, 9.10_

  - [ ] 8.1 Implement core MemoirAI class and public methods

    - Create MemoirAI class with comprehensive configuration parameter support
    - Implement ingest_text method integrating chunking, classification, and storage with transaction management
    - Implement query method with strategy selection and result aggregation
    - Add get_category_tree method for hierarchy inspection
    - Add regenerate_contextual_helper method for helper management
    - Write integration tests for all public API methods
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ] 8.2 Implement transaction management and error recovery

    - Add database transaction wrapping for all ingestion operations
    - Implement transaction rollback logic for database write failures during ingestion
    - Create comprehensive error logging with sufficient detail for debugging and retry
    - Add batch-level transaction management treating entire batches as single units
    - Implement clear error messages with retry strategy suggestions for transaction failures
    - Write tests for transaction rollback scenarios and error recovery mechanisms
    - _Requirements: 7B.1, 7B.2, 7B.3, 7B.4, 7B.5, 7B.6, 7B.7_

  - [ ] 8.3 Add comprehensive error handling and edge case management

    - Implement detailed error messages with troubleshooting guidance for all failure modes
    - Add comprehensive logging system with configurable levels and performance metrics
    - Create graceful handling for LLM service unavailability and database connection issues
    - Add input validation and sanitization for all user inputs
    - Implement edge case handling for short/long text and budget exceeded scenarios
    - Write tests for error scenarios and recovery mechanisms
    - _Requirements: 1.5, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9_

  - [ ] 8.2 Add comprehensive error handling and logging

    - Implement detailed error messages with troubleshooting guidance for all failure modes
    - Add comprehensive logging system with configurable levels and performance metrics
    - Create graceful handling for LLM service unavailability and database connection issues
    - Add input validation and sanitization for all user inputs
    - Write tests for error scenarios and recovery mechanisms
    - _Requirements: 1.5, 7.3, 7.4, 7.5, 7.6_

  - [ ] 9.1 Implement configuration management and validation system

    - Create configuration classes for all system parameters with comprehensive validation
    - Add validation for token budget vs chunk size constraints and hierarchy depth limits
    - Implement batch size and category limit validation with clear error messages
    - Add environment variable support for database and LLM provider settings
    - Implement configuration file support with schema validation
    - Create ConfigurationError class with specific guidance for fixing invalid settings
    - Write comprehensive tests for all validation scenarios and error messages
    - _Requirements: 1.1, 1.2, 1.4, 7A.1, 7A.2, 7A.3, 7A.4, 7A.5, 7A.6, 7A.7_

  - [ ] 9.2 Add database backend support and testing

    - Implement and test SQLite backend with file and in-memory modes
    - Implement and test PostgreSQL backend with connection pooling
    - Implement and test MySQL backend with proper charset handling
    - Add database-specific optimization and index strategies
    - Create database backend switching tests and migration compatibility
    - _Requirements: 1.1, 1.3_

  - [ ] 10.1 Implement performance and integration tests

    - Create performance benchmarks for chunking, classification, and query processing
    - Add integration tests with real LLM providers (using test API keys)
    - Implement concurrent access tests for thread safety validation
    - Create memory usage and token consumption monitoring tests
    - Add end-to-end workflow tests with large document processing
    - _Requirements: All requirements - comprehensive validation_

  - [ ] 10.2 Create documentation and examples
    - Write comprehensive API documentation with usage examples
    - Create tutorial notebooks demonstrating key features and use cases
    - Add configuration guide for different deployment scenarios
    - Create troubleshooting guide for common issues and solutions
    - Write performance tuning guide for optimization strategies
    - _Requirements: 6.4, 6.5 (clear error messages and guidance)_

- [ ] 11. Package and distribution setup
  - Create package distribution configuration with proper dependency management
  - Add CLI interface for testing and administration tasks
  - Create Docker containerization for easy deployment
  - Add CI/CD pipeline with automated testing and release management
  - Create installation and quick-start documentation
  - _Requirements: 1.1 (easy installation and setup)_
