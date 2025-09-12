# Requirements Document

## Introduction

The memoirAI library is a Python package that enables LLMs to store and retrieve structured text chunks using a relational SQL database with a configurable category hierarchy. The system supports up to 100 hierarchy levels, with a three-level hierarchy as the default (Category → Sub-Category → Sub-Sub-Category). Unlike vector databases, this system uses discrete classification via LLM prompts to provide explainable, cost-efficient, and auditable memory storage and retrieval. The library will be database-agnostic, supporting SQLite, PostgreSQL, and MySQL, with configurable text chunking and intelligent category management to avoid duplicates.

### Intended Audience and Scope

This library is intended for developers and researchers building applications that require structured memory retrieval. It can be used in prototyping, research projects, or production systems where explainability and auditability are critical. The scope covers ingestion, classification, and retrieval of textual data, not vector search or semantic embedding systems.

## Requirements

### Requirement 1

**User Story:** As a developer, I want to install and configure the memoirAI library with my preferred SQL database and LLM provider, so that I can quickly integrate structured memory capabilities into my application.

#### Acceptance Criteria

WHEN a developer installs the library THEN the system SHALL support SQLite, PostgreSQL, and MySQL databases

WHEN configuring the library THEN the system SHALL accept database connection strings via environment variables or configuration parameters

WHEN initializing the library THEN the system SHALL automatically create required database tables and indexes if they don't exist

WHEN configuring LLM providers THEN the system SHALL support multiple providers through Pydantic AI (OpenAI, Anthropic, etc.)

IF the database connection fails THEN the system SHALL provide clear error messages with troubleshooting guidance

### Requirement 2

**User Story:** As a developer, I want to ingest large text sources that are automatically chunked and categorized, so that the content is stored in a structured, retrievable format.

#### Acceptance Criteria

WHEN ingesting text THEN the system SHALL split content using configurable delimiters (periods and line breaks)

WHEN chunking text THEN the system SHALL respect configurable minimum and maximum length thresholds (default 1000-2000 characters)

WHEN chunking THEN the system SHALL preserve paragraph boundaries and meaningful text segments

WHEN a chunk is created THEN the system SHALL classify it into the category hierarchy using iterative LLM prompts

WHEN classifying THEN the system SHALL prompt to the LLM for each hierarchy level sequentially from level 1 to the configured maximum depth

WHEN prompting for each level THEN the system SHALL present existing categories at that level to encourage reuse and avoid duplicates

WHEN classifying THEN the system SHALL create or choose a category up to N configurable number of categories, default is 128, applied globally across all levels

WHEN classifying IF the max number N of categories is reached globally THEN the system will not create new categories but rather it will be forced to choose from an existing one via pydantic AI Literal type.

WHEN reaching the maximum configured hierarchy depth THEN the system SHALL stop classification and link the chunk to the final category

WHEN classification is complete THEN the system SHALL store the chunk linked to the deepest (leaf-level) category in the database

### Requirement 3

**User Story:** As a developer, I want the system to maintain a configurable category hierarchy in the database, so that data integrity is preserved and queries are predictable across different hierarchy depths.

#### Acceptance Criteria

WHEN configuring the system THEN the system SHALL support hierarchy depths from 1 to 100 levels with 3 levels as the default

WHEN creating categories THEN the system SHALL enforce the configured maximum hierarchy depth

WHEN inserting level 1 categories THEN the system SHALL ensure they have no parent_id

WHEN inserting categories at level 2 or higher THEN the system SHALL require a valid parent_id pointing to the previous level

WHEN creating categories THEN the system SHALL ensure names are unique within the same parent scope

WHEN storing chunks THEN the system SHALL only allow linking to leaf-level categories (deepest configured level)

WHEN querying categories THEN the system SHALL provide indexes on parent_id for efficient traversal

WHEN querying chunks THEN the system SHALL provide indexes on category_id for efficient retrieval

WHEN configuring hierarchy depth THEN the system SHALL validate that the depth is between 1 and 100 levels

### Requirement 4

**User Story:** As a developer, I want to query the stored content using natural language, so that I can retrieve relevant chunks without knowing the exact category structure.

#### Acceptance Criteria

WHEN submitting a query THEN the system SHALL use LLM classification to traverse the category hierarchy level by level, each level is an LLM API call. The LLM will choose a category from the existing categories for that text/knowledge on every API call base until reaching a leaf node.

WHEN classifying a query THEN the system SHALL prompt for each hierarchy level sequentially until reaching a leaf node

WHEN reaching the deepest configured level THEN the system SHALL retrieve chunks linked to the identified category path

IF no chunks exist at the deepest level THEN the system SHALL return and log an error

WHEN returning results THEN the system SHALL include the full category path (category_path) for each chunk, text_content, ranked_relevance from 1-10 (only applicable if the LLM is selecting more than one category at a specific level), created_at timestamp

The final result should be an object containing a list of chunk objects, timestamp, total_latency, dropped_paths (optional)

### Requirement 5

**User Story:** As a developer, I want structured JSON outputs from LLM interactions, so that the classification process is reliable and parseable.

#### Acceptance Criteria

WHEN prompting LLMs THEN the system SHALL use Pydantic schemas to enforce structured JSON responses

WHEN classifying content THEN the system SHALL validate LLM responses against predefined schemas (pydantic naturally already takes care of this out of the box)

### Requirement 6

**User Story:** As a developer, I want a simple Python API for ingestion and retrieval, so that I can integrate the library with minimal code changes.

#### Acceptance Criteria

WHEN using the library THEN the system SHALL provide a simple initialization method accepting database connection parameters

WHEN ingesting content THEN the system SHALL provide an ingest_text() method accepting text strings

WHEN querying content THEN the system SHALL provide a query() method accepting natural language queries

WHEN operations complete THEN the system SHALL return structured response objects with success/error status

WHEN errors occur THEN the system SHALL provide detailed error messages and suggested remediation steps

WHEN using the API THEN the system SHALL support both synchronous and asynchronous operation modes

### Requirement 7

**User Story:** As a developer, I want the library to handle edge cases gracefully, so that my application remains stable under various conditions.

#### Acceptance Criteria

WHEN processing very short text THEN the system SHALL handle content below minimum chunk size appropriately, continuing using the text below the minimum and logging a warning

WHEN processing very long text during chunking THEN the system SHALL handle content above maximum chunk size by splitting into multiple chunks instead of throwing an error

WHEN retrieving results IF the aggregated output exceeds the maximum configured token budget THEN the system SHALL apply the budget handling rules defined in Requirement 9

WHEN LLM services are unavailable THEN the system SHALL provide fallback mechanisms or clear error handling

WHEN database connections are lost THEN the system SHALL implement retry logic with exponential backoff

WHEN invalid SQL characters are present THEN the system SHALL properly escape and sanitize all inputs

### Requirement 8

**User Story:** As a developer, I want to retrieve relevant chunks using natural language without knowing the exact category structure, and I want to control how the system explores the hierarchy so I can balance precision, coverage, and cost.

Retrieval strategies

The system supports four configurable strategies for traversing the hierarchy with an LLM. At each level, the LLM selects categories from the presented options.

**One shot:**

The system chooses a single category at each level until a leaf is reached.

**Wide branch:**

The system chooses N top categories at the first level and again chooses N top subcategories at each subsequent level. N is configurable with default 3.

**Zoom in:**

The system begins with N top categories at level one and at each deeper level selects N minus N2 categories. If N2 is larger than N then the absolute value of N divided by N2, rounded up to at least one, is used. Defaults are N equals 6 and N2 equals 2.

**Branch out:**

The system begins with N top categories at level one and at each deeper level selects N plus N2 categories up to the available options, capped by a per level maximum. Defaults are N equals 6 and N2 equals 2.

### Requirement 9

**User Story:**

Result aggregation and synthesis

The system collects all candidate chunks from every selected leaf category along the explored paths, deduplicates by chunk id or hash, concatenates them in a deterministic order, and feeds them to the LLM subject to a global token budget.

Configuration

allow_exceed_max_token_budget_and_summarize boolean

max_token_budget integer required

Budget handling

If allow_exceed_max_token_budget_and_summarize is true and the aggregated text exceeds max_token_budget, the system splits the aggregated text into M contiguous parts where M equals ceil(total_tokens divided by max_token_budget). Each part is summarized by the LLM using new queries/prompts to the LLM so that its length is approximately max_token_budget divided by M tokens. The final LLM call receives the query plus the M summaries.

If allow_exceed_max_token_budget_and_summarize is false, the system streams chunks in deterministic order and stops pulling more from the database exactly when the estimated budget would be exceeded. Any unqueried paths are recorded as dropped due to budget.

Deterministic order

Order paths by strategy declared order then by category path identifier ascending.

Within a path, order chunks by created_at ascending then chunk_id ascending.

Final LLM call output

Structured JSON validated by Pydantic with fields

answer string

The final output of the program does not require the LLM to add to the final json object:

used_chunks list of objects

dropped_paths optional list of category paths

total_latency double

#### Acceptance criteria

When traversal completes the system shall aggregate all candidate chunks and apply the chosen budget behavior exactly as specified above.

When allow_exceed_max_token_budget_and_summarize is true the system shall produce M summaries whose combined tokens do not exceed max_token_budget by more than 5 percent.

When allow_exceed_max_token_budget_and_summarize is false the system shall stop database reads at the budget boundary and shall return a machine readable list of dropped_paths.

The system shall never perform local semantic ranking or filtering prior to the budget step except for deduplication and removal of exact repeats.

The system shall log for each query the total candidate tokens, budget, whether summarization was used, the number of parts if used, token counts per part, any dropped_paths, token usage, and latency.

The final LLM output shall be returned together with citations to chunk ids and category paths used, and validation errors shall trigger up to three retries with clarified prompts.

Deterministic mode shall be supported by fixing temperature to zero and recording provider parameters in logs.

When submitting a query the system SHALL support the four traversal strategies listed above, selectable by configuration or per call parameter.

When traversing the hierarchy the system SHALL present existing categories at the current level and request the LLM a structured selection of categories per the chosen strategy.

When selecting categories the system SHALL validate and parse LLM outputs using Pydantic schemas, including category identifiers and optional confidence scores.

When a path reaches the deepest configured level the system SHALL retrieve chunks linked to the corresponding leaf categories on every selected path for the current query run.

When returning results the system SHALL aggregate chunks from all selected leaf categories into a single array of chunk objects. Each object SHALL contain text_content, ranked_relevance from 1-10 (only applicable if the LLM is selecting more than one category at a specific level), created_at timestamp, category_path and chosen_strategy

If a selected leaf category contains no chunks this SHALL be treated as an error and be logged.

If a level offers fewer categories than required by the strategy the system SHALL select all available categories at that level and proceed.

The system SHALL record for each query the strategy used, the explored category paths, confidence scores if available, the final ranked list, token usage, and total latency.
