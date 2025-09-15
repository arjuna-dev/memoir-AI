# Requirements Document

## Introduction

The memoirAI library is a Python package that enables LLMs to store and retrieve structured text chunks using a relational SQL database with a configurable category hierarchy. The system supports up to 100 hierarchy levels, with a three-level hierarchy as the default (Category → Sub-Category → Sub-Sub-Category). Unlike vector databases, this system uses discrete classification via LLM prompts to provide explainable, cost-efficient, and auditable memory storage and retrieval. The library will be database-agnostic, supporting SQLite, PostgreSQL, and MySQL, with configurable text chunking and intelligent category management to avoid duplicates.

### Intended Audience and Scope

This library is intended for developers and researchers building applications that require structured memory retrieval. It can be used in prototyping, research projects, or production systems where explainability and auditability are critical. The scope covers ingestion, classification, and retrieval of textual data, not vector search or semantic embedding systems.

## Requirements

### Requirement 1

**User Story:** As a developer, I want to install and configure the memoirAI library with my preferred SQL database and LLM provider, so that I can quickly integrate structured memory capabilities into my application.

#### Acceptance Criteria

1. WHEN a developer installs the library THEN the system SHALL support SQLite, PostgreSQL, and MySQL databases.
2. WHEN configuring the library THEN the system SHALL accept database connection strings via environment variables or configuration parameters.
3. WHEN initializing the library THEN the system SHALL automatically create required database tables and indexes if they don't exist.
4. WHEN configuring LLM providers THEN the system SHALL support multiple providers through Pydantic AI (OpenAI, Anthropic, etc.).
5. IF the database connection fails THEN the system SHALL provide clear error messages with troubleshooting guidance.

---

### Requirement 2

**User Story:** As a developer, I want to ingest large text sources that are automatically chunked and categorized, so that the content is stored in a structured, retrievable format.

#### Acceptance Criteria

1. WHEN ingesting text THEN the system SHALL split content using configurable delimiters (periods and line breaks are the default).
2. WHEN chunking text THEN the system SHALL respect configurable minimum and maximum length thresholds (default 300–500 tokens are the default).
3. WHEN counting tokens THEN the system SHALL use the liteLLM's `token_counter` function (python library) which allows to count tokens for a variety of LLM models.
4. WHEN chunking THEN the system SHALL preserve paragraph boundaries and meaningful text segments (through the use of the pre-configured delimiters).
5. WHEN a chunk is created THEN the system SHALL classify it into the category hierarchy using iterative LLM prompts.
6. WHEN classifying THEN the system SHALL prompt to the LLM for each hierarchy level sequentially from level 1 to the configured maximum depth.
7. WHEN prompting for each level THEN the system SHALL present existing categories at that level to the LLM in the prompt to encourage reuse and avoid duplicates.
8. WHEN classifying THEN the system SHALL enforce a maximum number of categories per level. This limit SHALL be configurable in two ways:
     a. Global configuration: one default N (128) applies uniformly across all levels.
     b. Per-level configuration: individual limits may be specified for each level, overriding the global default where defined.
9. WHEN classifying IF the max number N of categories is reached globally THEN the system will not create new categories but rather it will be forced to choose from an existing one via Pydantic AI Literal type.
10. WHEN reaching the maximum configured hierarchy depth THEN the system SHALL stop classification and link the chunk to the final category.
11. WHEN classification is complete THEN the system SHALL store the chunk linked to the deepest (leaf-level) category in the database.

---

### Requirement 2A

**User Story:** As a developer, I want to classify multiple chunks in a single LLM call, so that ingestion is faster and cheaper.

#### Acceptance Criteria

1. WHEN classifying during ingestion THEN the system SHALL support sending multiple chunks in a single prompt.

2. The prompt SHALL list chunks in the following exact structure and order:

   ```
   Chunk 1:
   """
   <text content>
   """
   Chunk 2:
   """
   <text content>
   """
   ```

3. The LLM response SHALL include only the category decision for each input chunk and SHALL not echo the chunk texts.

4. The response format SHALL be the following JSON:

   ```json
   {
     "chunks": [
       { "id": 1, "category": "<category>" },
       { "id": 2, "category": "<category>" }
     ]
   }
   ```

5. The system SHALL include stable numeric IDs for each chunk in the prompt and require the same IDs in the response.

6. The number of chunks sent per LLM call SHALL be configurable as batch_size with default 5.

7. IF the number of pending chunks exceeds batch_size THEN the system SHALL process them in successive calls while preserving original order.

8. The system SHALL rely on the pydantic-ai framework to enforce that the LLM outputs conform exactly to the defined schema. Any response that does not conform SHALL be rejected automatically.

9. IF any chunk in the batch fails validation THEN the system SHALL retry only that chunk in a follow-up call and SHALL not reprocess successful ones.

10. The system SHALL log per batch the number of chunks sent, number of successes, number of retries, and latency.

---

### Requirement 2B

**User Story:** As a developer, I want the system to build and reuse a short contextual helper per source so that classification prompts have relevant context without repeating whole documents.

#### Acceptance Criteria

1. When auto_source_identification is true and no user provided contextual_helper exists for the source, the system shall generate a contextual_helper once at ingestion time.
2. The system shall derive the contextual_helper from available metadata and content, using in this order: file name, detected headers or title, and the first 5 chunks or a derivation budget of two thousand tokens, whichever comes first.
3. The contextual_helper shall include, when present, author, date, topic, and a concise one paragraph summary of the source.
4. The contextual_helper shall be stored with the source identifier, versioned, and reused for every classification and retrieval prompt that references that source.
5. The contextual_helper shall be included in every classification request prompt in a fixed field named contextual_helper.
6. The contextual_helper shall not exceed three hundred tokens in length.
7. When auto_source_identification is false, the system shall request the user to provide a contextual_helper before ingestion begins and shall pause ingestion until a value is supplied.
   7.1 The user prompt shall collect at least the following fields: date of document creation, author, topic, and a short text description of the source.
   7.2 The date of document creation shall be entered in ISO 8601 format, for example YYYY MM DD or YYYY MM DDTHH MM SS. If the date cannot be parsed the system shall request correction.
   7.3 The short text description shall be limited to two hundred tokens. The final contextual_helper remains subject to the overall three hundred token limit in item 6.
   7.4 If a field is unknown the user may enter the value unknown. The system shall still pause until the user submits all required fields, even if some are unknown.
   7.5 After collecting the fields the system shall compose a single paragraph contextual_helper that includes the provided author, date, topic, and description when present.
   7.6 The system shall display a preview of the composed contextual_helper with an estimated token count and request the user to confirm, edit, or cancel.
   7.7 On confirmation the system shall store the contextual_helper for the source, mark it as user provided, and resume ingestion. On cancel the system shall abort ingestion without side effects.
8. A user provided contextual_helper shall override any previously generated helper for the same source.
9. The system shall expose a regenerate flag that, when set, rebuilds the contextual_helper from current source metadata and the first chunks using the same limits.
10. The system shall log helper creation events including timestamp, token usage, and whether the value was user provided or generated.

---

### Requirement 3

**User Story:** As a developer, I want the system to maintain a configurable category hierarchy in the database, so that data integrity is preserved and queries are predictable across different hierarchy depths.

#### Acceptance Criteria

1. WHEN configuring the system THEN the system SHALL support hierarchy depths from 1 to 100 levels with 3 levels as the default.
2. WHEN creating categories THEN the system SHALL enforce the configured maximum hierarchy depth.
3. WHEN inserting level 1 categories THEN the system SHALL ensure they have no parent_id.
4. WHEN inserting categories at level 2 or higher THEN the system SHALL require a valid parent_id pointing to the previous level.
5. WHEN creating categories THEN the system SHALL ensure names are unique within the same parent scope.
6. WHEN storing chunks THEN the system SHALL only allow linking to leaf-level categories (deepest configured level).
7. WHEN querying categories THEN the system SHALL provide indexes on parent_id for efficient traversal.
8. WHEN querying chunks THEN the system SHALL provide indexes on category_id for efficient retrieval.
9. WHEN configuring hierarchy depth THEN the system SHALL validate that the depth is between 1 and 100 levels.

---

### Requirement 4

**User Story:** As a developer, I want to query the stored content using natural language, so that I can retrieve relevant chunks without knowing the exact category structure.

#### Acceptance Criteria

1. WHEN submitting a query THEN the system SHALL use LLM responses to select categories, traversing the category hierarchy level by level until reaching a leaf node.
   1.1 Each LLM response SHALL conform to a predefined pydantic ai schema and SHALL return only:
   category string
   ranked_relevance integer
   1.2 The ranked_relevance score SHALL range from 1 up to N where N is the maximum number of categories that may be selected at that level. Rankings SHALL be unique and descending with the most relevant category receiving N. If only one category is selected it SHALL receive N.

2. AFTER the LLM finishes selecting categories for the current level or set of paths THEN the system SHALL construct a list of fully qualified category paths from root to leaf.
   2.1 The system SHALL validate that each path ends at a leaf level as configured.
   2.2 The system SHALL deduplicate identical paths before retrieval.

3. FOR each leaf category in the validated paths THEN the system SHALL perform SQL retrieval of linked chunks.
   3.1 The system SHALL select by leaf category identifier and return chunk identifier text_content category_path created_at.
   3.2 The system SHALL order results deterministically by created_at ascending then chunk identifier ascending.
   3.3 The system SHALL support pagination and an optional per path limit without changing the deterministic order.
   3.4 IF a leaf category contains no chunks THEN the system SHALL record an error for that path and proceed with remaining paths.

4. WHEN any LLM call completes THEN the system SHALL construct a per call response object that contains:
   4.1 llm_output which is the structured LLM result category plus ranked_relevance
   4.2 timestamp of the call
   4.3 latency of the call in milliseconds

5. AFTER SQL retrieval the system SHALL assemble a query result object that contains:
   5.1 responses which is the list of all per call response objects in call order
   5.2 chunks which is the list of all retrieved chunk objects each with text_content category_path ranked_relevance and created_at
   5.3 dropped_paths optional if budget or traversal rules prevented some paths from being queried
   5.4 total_latency which is the sum of all LLM and retrieval latencies for the query

6. IF no chunks exist at the deepest level for any selected path THEN the system SHALL return an error and include the empty set of paths in dropped_paths.

7. The query result object SHALL be the final return value of the system.

---

### Requirement 5

**User Story:** As a developer, I want structured JSON outputs from LLM interactions, so that the classification process is reliable and parseable.

#### Acceptance Criteria

1. WHEN prompting LLMs THEN the system SHALL use Pydantic schemas for the Pydantic AI library (different from pydantic) to enforce structured JSON responses.
2. WHEN classifying content THEN Pydantic AI validates LLM responses against predefined schemas so the system does not need to take care fo this.
3. WHEN using Pydantic AI (different from pydantic) with models that support native structured outputs (OpenAI, Grok, Gemini) THEN the system SHALL use the NativeOutput option. If not supported, the system SHALL fall back to standard schema enforcement.

Native Output docs:

Native Output mode uses a model's native "Structured Outputs" feature (aka "JSON Schema response format"), where the model is forced to only output text matching the provided JSON schema. Note that this is not supported by all models, and sometimes comes with restrictions. For example, Anthropic does not support this at all, and Gemini cannot use tools at the same time as structured output, and attempting to do so will result in an error.

To use this mode, you can wrap the output type(s) in the NativeOutput marker class that also lets you specify a name and description if the name and docstring of the type or function are not sufficient.

python

```
from pydantic_ai import Agent, NativeOutput

from tool_output import Fruit, Vehicle

agent = Agent(
'openai:gpt-4o',
output_type=NativeOutput(
[Fruit, Vehicle],
name='Fruit_or_vehicle',
description='Return a fruit or vehicle.'
),
)
result = agent.run_sync('What is a Ford Explorer?')
print(repr(result.output))
#> Vehicle(name='Ford Explorer', wheels=4)

```

---

### Requirement 6

**User Story:** As a developer, I want a simple Python API for ingestion and retrieval, so that I can integrate the library with minimal code changes.

#### Acceptance Criteria

1. WHEN using the library THEN the system SHALL provide a simple initialization method accepting database connection parameters.
2. WHEN ingesting content THEN the system SHALL provide an `ingest_text()` method accepting text strings.
3. WHEN querying content THEN the system SHALL provide a `query()` method accepting natural language queries.
4. WHEN operations complete THEN the system SHALL return structured response objects with success/error status.
5. WHEN errors occur THEN the system SHALL provide detailed error messages and suggested remediation steps.
6. WHEN using the API THEN the system SHALL support both synchronous and asynchronous operation modes.

---

### Requirement 7

**User Story:** As a developer, I want the library to handle edge cases gracefully, so that my application remains stable under various conditions.

#### Acceptance Criteria

1. WHEN processing very short text THEN the system SHALL handle content below minimum chunk size appropriately, continuing using the text below the minimum and logging a warning.
2. WHEN processing very long text during chunking THEN the system SHALL split content above maximum chunk size into multiple chunks instead of throwing an error.
3. WHEN retrieving results IF the aggregated output exceeds the maximum configured token budget THEN the system SHALL apply the budget handling rules defined in Requirement 9.
4. WHEN LLM services are unavailable THEN the system SHALL provide fallback mechanisms or clear error handling.
5. WHEN database connections are lost THEN the system SHALL implement retry logic with exponential backoff.
6. WHEN invalid SQL characters are present THEN the system SHALL properly escape and sanitize all inputs.
7. WHEN initializing the library THEN the system SHALL validate configuration parameters and provide clear error messages for invalid settings.
8. WHEN ingesting content THEN the system SHALL wrap the entire ingestion process in database transactions to ensure data consistency.
9. WHEN database write operations fail during ingestion THEN the system SHALL rollback the entire batch transaction and log the failure for user retry.

---

### Requirement 7A

**User Story:** As a developer, I want the library to validate my configuration settings at initialization, so that I can catch configuration errors early and understand what needs to be fixed.

#### Acceptance Criteria

1. WHEN initializing the library THEN the system SHALL validate that `max_token_budget` is greater than `chunk_min_tokens` plus a reasonable overhead (minimum 100 tokens).
2. WHEN configuring hierarchy depth THEN the system SHALL validate that the value is between 1 and 100 inclusive.
3. WHEN configuring batch size THEN the system SHALL validate that the value is greater than 0 and less than or equal to 50.
4. WHEN configuring token thresholds THEN the system SHALL validate that `chunk_min_tokens` is less than `chunk_max_tokens`.
5. WHEN configuring category limits THEN the system SHALL validate that all limits are positive integers.
6. WHEN configuration validation fails THEN the system SHALL raise a clear ConfigurationError with specific guidance on how to fix the invalid setting.
7. WHEN per-level category limits are specified THEN the system SHALL validate that each level has a corresponding limit configuration.

---

### Requirement 7B

**User Story:** As a developer, I want the library to handle database failures during ingestion gracefully with proper transaction management, so that my data remains consistent even when errors occur.

#### Acceptance Criteria

1. WHEN starting an ingestion operation THEN the system SHALL begin a database transaction before any data modifications.
2. WHEN chunk classification completes successfully THEN the system SHALL proceed to database writes within the same transaction.
3. WHEN any database write operation fails during ingestion THEN the system SHALL rollback the entire transaction to maintain data consistency.
4. WHEN a transaction rollback occurs THEN the system SHALL log the failure with sufficient detail for debugging and user retry.
5. WHEN ingestion completes successfully THEN the system SHALL commit the transaction and confirm all data is persisted.
6. WHEN multiple chunks are processed in a batch THEN the system SHALL treat the entire batch as a single transaction unit.
7. WHEN transaction failures occur THEN the system SHALL provide clear error messages indicating which operation failed and suggest retry strategies.

---

### Requirement 8

**User Story:** As a developer, I want to retrieve relevant chunks using natural language without knowing the exact category structure, and I want to control how the system explores the hierarchy so I can balance precision, coverage, and cost.

**Retrieval strategies**

The system supports four configurable strategies for traversing the hierarchy with an LLM. At each level, the LLM selects categories from the presented options.

- **One shot:** The system chooses a single category at each level until a leaf is reached.
- **Wide branch:** The system chooses N top categories at the first level and again chooses N top subcategories at each subsequent level. N is configurable with default 3.
- **Zoom in:** The system begins with N top categories at level one and at each deeper level selects N minus N2 categories. Defaults are N=6 and N2=2.
- **Branch out:** The system begins with N top categories at level one and at each deeper level selects N plus N2 categories up to the available options, capped by a per-level maximum. Defaults are N=6 and N2=2.

---

### Requirement 9

**User Story:** As a developer, I want the system to aggregate results and limit the final prompt in a clear and deterministic way, using either pruning **or** summarization (never both).

#### Result aggregation and prompt limiting

After traversal completes, the system executes SQL retrieval **once per unique leaf path**, collects all candidate chunks keyed by `chunk_id`, orders them deterministically, and prepares them for the final LLM prompt subject to a global budget.

**Configuration**

1. `prompt_limiting_strategy`: enum `PRUNE` | `SUMMARIZE` (default `PRUNE`).
2. `max_token_budget`: integer, required — maximum tokens allowed for the **final answer prompt** (query + contextual_helper + wrapper text + final content).
3. `use_rankings`: boolean, default `true` (applies only when `PRUNE`).
4. `summarization_instruction_headroom_tokens`: integer, default `1024` — reserved tokens for system + instructions when creating each summarization part.
5. `summary_char_overage_tolerance_percent`: integer, default `5` — allowed overage for the **combined summaries’ characters** versus their calculated character budget.
6. `summary_max_retries`: integer, default `1` — maximum retries for summaries that exceed their per-chunk character targets.

**Budget handling**

1. **Size estimation (always):**
   The system SHALL compute:

   - `fixed_prompt_token_count` = tokens(`contextual_helper` if present + wrapper text) using `liteLLM.token_counter`.
   - `chunks_total_token_count` = sum of tokens for all candidate chunks.
   - `fixed_prompt_char_count` = characters(`contextual_helper` if present + wrapper text).
   - `chunks_total_char_count` = sum of characters for all candidate chunks.

2. **If `prompt_limiting_strategy = PRUNE`:**
   2.1 If `fixed_prompt_token_count + chunks_total_token_count ≤ max_token_budget`, proceed without pruning.
   2.2 If the total exceeds `max_token_budget` **and** `use_rankings = true`, use `ranked_relevance` (per Requirement 8) to drop lowest-ranked paths until the estimate ≤ budget; record all `dropped_paths`.
   2.3 If the total exceeds `max_token_budget` **and** `use_rankings = false`, prune by deterministic order only until ≤ budget; record all `dropped_paths`.
   2.4 If all paths are dropped and the estimate still exceeds budget, return an error including considered and `dropped_paths`.

3. **If `prompt_limiting_strategy = SUMMARIZE`:**
   3.1 If `fixed_prompt_token_count + chunks_total_token_count ≤ max_token_budget`, proceed without summarizing.
   3.2 The system SHALL perform **no pruning** and SHALL ignore `use_rankings`.
   3.3 **Compute compression ratio from tokens, apply to characters:** - `required_compression_ratio = (max_token_budget - fixed_prompt_token_count) / chunks_total_token_count`. - If `required_compression_ratio ≥ 1`, summarization not required; proceed with original chunks. - If `required_compression_ratio ≤ 0`, return an error (budget too small for any content). - For each chunk `i`: - `original_char_count_i = len(chunk_i.text_content)` - `target_char_count_i = max(1, floor(original_char_count_i * required_compression_ratio))`
   3.4 **Partition into token-safe parts (respect max_token_budget also for these LLM calls):** - Define `summarization_part_token_cap = max_token_budget - summarization_instruction_headroom_tokens`. - Walk the ordered chunk list and accumulate chunks into **Part 1**, **Part 2**, … such that each part’s **input tokens** (its chunks + per-part instruction preface) `≤ summarization_part_token_cap`. - Each part MUST contain at least one chunk; preserve original order across and within parts.
   3.5 **LLM summarization calls (per part, multiple chunks at once):** - For each Part `k`, the system SHALL send a prompt listing each chunk in this exact structure:
   `      Chunk <chunk_id> (original_chars=<n>, target_chars=<t>):
      """
      <text_content>
      """
     ` - Instruction to the LLM:
   _“For each chunk above, produce a summary of at most its `target_chars` characters. Preserve key facts, entities, dates, numbers, and definitions. Omit repetition and boilerplate. Return only the JSON schema below.”_ - The LLM SHALL return structured output validated by **Pydantic AI**:
   `json
      {
        "summaries": [
          { "chunk_id": <int>, "summary": "<string>" }
        ]
      }
      `
   3.6 **Validation and retries:** - For every summary, verify `len(summary) ≤ target_char_count_i`. - Any over-limit summaries SHALL be retried **only for the offending chunk_ids** with a stricter instruction, up to `summary_max_retries`.
   3.7 **Combined summaries budget check (characters):** - `combined_summaries_char_count = sum(len(summary_i) for all chunks)`. - Ensure `combined_summaries_char_count ≤ (chunks_total_char_count * required_compression_ratio) * (1 + summary_char_overage_tolerance_percent/100)`. - If this check fails after retries, return an error indicating the character budget could not be met.
   3.8 **Final answer call and token check:** - Construct the **final answer prompt**: query + `contextual_helper` (if any) + wrapper text + **ordered concatenation** of all per-chunk summaries. - Verify with `liteLLM.token_counter` that final prompt tokens `≤ max_token_budget`. - If it exceeds the budget, return an error indicating that the required compression could not be achieved within limits.

**Deterministic order**

1. Order paths by strategy-declared order, then by category path identifier ascending.
2. Within a path, order chunks by `created_at` ascending, then by `chunk_id` ascending.

**Final outputs**

- **LLM final response (via Pydantic AI)**

  - Fields: `answer` (string)
  - Note: the LLM returns **only** the answer.

- **System final result (returned by the library)**
  - Fields:
    - `answer` (string) — the LLM’s answer
    - `used_chunks` (list of `{ chunk_id, category_path }`)
    - `dropped_paths` (optional list)
    - `total_latency` (double)
    - `responses` (list of per-call response objects: `{ timestamp, latency_ms, model, tokens_prompt, tokens_completion }`)

#### Acceptance Criteria

1. WHEN traversal completes THEN the system SHALL perform SQL retrieval once per unique leaf path and order candidate chunks deterministically.
2. WHEN estimating size THEN the system SHALL compute tokens (with `liteLLM.token_counter`) and characters (by string length) for fixed text and chunks.
3. WHEN `prompt_limiting_strategy = PRUNE` AND the token estimate ≤ `max_token_budget` THEN proceed without pruning.
4. WHEN `prompt_limiting_strategy = PRUNE` AND the token estimate > `max_token_budget` AND `use_rankings = true` THEN prune using `ranked_relevance` as defined in Requirement 8; record all `dropped_paths`.
5. WHEN `prompt_limiting_strategy = PRUNE` AND the token estimate > `max_token_budget` AND `use_rankings = false` THEN prune by deterministic order only; record all `dropped_paths`.
6. WHEN `prompt_limiting_strategy = PRUNE` AND all paths are dropped AND the token estimate still exceeds budget THEN return an error including considered and `dropped_paths`.
7. WHEN `prompt_limiting_strategy = SUMMARIZE` AND token estimate ≤ `max_token_budget` THEN proceed without summarization.
8. WHEN `prompt_limiting_strategy = SUMMARIZE` AND the token estimate exceeds `max_token_budget` THEN the system SHALL:
   8.1 compute `required_compression_ratio`,
   8.2 assign per-chunk `target_char_count_i`,
   8.3 partition chunks into token-safe parts using `max_token_budget` and `summarization_instruction_headroom_tokens`,
   8.4 summarize each part as specified, and
   8.5 validate per-chunk lengths and combined character budget.
9. The final answer prompt tokens SHALL be `≤ max_token_budget`; otherwise error.
10. The system SHALL return a **System final result** containing the LLM `answer` plus metadata (`used_chunks`, `dropped_paths`, `total_latency`, `responses`).
11. Deterministic mode SHALL be supported by fixing temperature to zero and recording provider parameters.
12. The system SHALL support the four traversal strategies (Requirement 8) and validate LLM outputs with Pydantic AI where applicable.

```

```
