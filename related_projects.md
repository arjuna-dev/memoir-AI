Got it — here’s the full text version you can copy-paste directly. I’ve added GitHub / arXiv / doc URLs for each project and paper where available.

---

# Alternatives to Vector Database RAG Systems

This document summarizes research papers, GitHub repos, and open-source libraries related to **RAG systems without vector databases** and **LLM memory architectures**, including commentary on how they relate to the hierarchical SQL-only retrieval idea.

---

## 📄 Research Papers & Academic Work

### Hybrid / Hierarchical / Non-Vector Retrieval

- **NeuSym-RAG: Hybrid Neural Symbolic Retrieval (2025)**
  Combines SQL filtering with vector retrieval. SQL is used as a pre-filter step.
  _Comment:_ Not pure SQL, but shows promise for hybrid architectures.
  [ACL Anthology PDF](https://aclanthology.org/2025.acl-long.311.pdf)

- **HIRO: Hierarchical Information Retrieval Optimization (2024)**
  Organizes documents in levels of summarization, traversed like a tree to prune irrelevant branches.
  _Comment:_ Hierarchical design overlaps with the “categorical narrowing” idea.
  [arXiv](https://arxiv.org/abs/2406.09979)

- **CRUSH4SQL: Collective Retrieval Using Schema Hallucination for Text2SQL (2023)**
  Lets LLMs hallucinate minimal schemas, then retrieves SQL subsets.
  _Comment:_ Works in structured domains (schema retrieval) rather than documents.
  [arXiv](https://arxiv.org/abs/2311.01173)

- **RB-SQL: A Retrieval-based LLM Framework for Text-to-SQL (2024)**
  Retrieves concise table/column subsets as context.
  _Comment:_ More SQL schema-focused, but retrieval principles overlap.
  [arXiv](https://arxiv.org/abs/2407.08273)

- **Structured Hierarchical Retrieval (LlamaIndex)**
  Supports “tree indices” and structured retrieval.
  _Comment:_ Hierarchical indexing, but usually backed by embeddings at the leaves.
  [Docs](https://docs.llamaindex.ai/en/v0.9.48/examples/query_engine/multi_doc_auto_retrieval/multi_doc_auto_retrieval.html)

- **DuckDB + RAG Integration (FlockMTL, 2025)**
  Treats `MODEL` and `PROMPT` as schema objects inside DuckDB.
  _Comment:_ Embeds LLM operations inside SQL, bridging structured + unstructured retrieval.
  [arXiv](https://arxiv.org/abs/2504.01157)

---

## 🧠 LLM Memory Libraries

1. **Memori (GibsonAI/memori)**
   SQL-native memory engine using Postgres/MySQL. Entity extraction + SQL retrieval.
   _Comment:_ Very close to “SQL instead of vector DB,” though geared to conversational memory.
   [GitHub](https://github.com/GibsonAI/memori)

2. **MemoryOS (BAI-LAB/MemoryOS)**
   Hierarchical “memory operating system” with storage/retrieval layers.
   _Comment:_ Hierarchical structure overlaps conceptually, unclear if embeddings are used.
   [GitHub](https://github.com/BAI-LAB/MemoryOS)

3. **GMemory (bingreeky/GMemory)**
   Tracing hierarchical memory for multi-agent systems.
   _Comment:_ Hierarchy is key, but not document-RAG oriented.
   [GitHub](https://github.com/bingreeky/GMemory)

4. **Memoripy (caspianmoon/memoripy)**
   Short/long-term memory, clustering, graph associations.
   _Comment:_ Uses embeddings internally, not pure SQL.
   [GitHub](https://github.com/caspianmoon/memoripy)

5. **HighNoonLLM**
   Hierarchical spatial neural memory, splitting sequences into memory trees.
   _Comment:_ More neural architecture, but similar “tree narrowing” principles.
   [GitHub](https://github.com/versoindustries/HighNoonLLM)

6. **Beyond Quacking (DuckDB + RAG Integration, FlockMTL)**
   Integrates LLM operations (`MODEL`, `PROMPT`) into DuckDB schema.
   _Comment:_ Not purely memory but important as it embeds retrieval and model ops directly in SQL.
   [arXiv](https://arxiv.org/abs/2504.01157)

7. **HMT (Hierarchical Memory Transformer)**
   Transformer model with memory structured hierarchically (segment-level recurrence).
   _Comment:_ A neural approach to hierarchical memory, aligns with narrowing search spaces.
   [arXiv](https://arxiv.org/abs/2309.10276)

---

## 📚 RAG Without Vector Databases

1. **harshitv804/RAG-without-Vector-DB**
   RAG system using **TF-IDF retriever**, long context, and reranking instead of embeddings.
   _Comment:_ Closest open-source attempt at “RAG without vectors.”
   [GitHub](https://github.com/harshitv804/RAG-without-Vector-DB)

2. **xetdata/RagIRBench**
   Benchmark repo showing classical IR (BM25/TF-IDF) can replace vector DBs for many cases.
   _Comment:_ Not a production library, but a strong proof-of-concept.
   [GitHub](https://github.com/xetdata/RagIRBench)

3. **PolyRAG (QuentinFuxa/PolyRAG)**
   Combines SQL, RAG, and PDF parsing for multi-source Q&A.
   _Comment:_ Hybrid approach, doesn’t eliminate vectors entirely.
   [GitHub](https://github.com/QuentinFuxa/PolyRAG)

4. **Azure-Samples/azure-sql-db-chatbot**
   SQL database + OpenAI chatbot demo. Embeddings computed inside SQL with vector functions.
   _Comment:_ SQL-based vector retrieval, not pure symbolic.
   [GitHub](https://github.com/Azure-Samples/azure-sql-db-chatbot)

5. **vanna-ai/vanna**
   Python library that turns natural language into SQL queries.
   _Comment:_ Retrieval is structured (SQL) not document-based; no vector DB needed.
   [GitHub](https://github.com/vanna-ai/vanna)

6. **RAG-with-SQLite-FTS (various forks)**
   Several experimental repos use **SQLite FTS5** (full-text search) instead of vector DBs for retrieval.
   _Comment:_ Good minimal examples, but usually demos not scalable systems.
   Example: [GitHub](https://github.com/kevinamiri/RAG-SQLite-FTS)

7. **DuckDB FTS-based RAG demos**
   Some repos show **DuckDB full-text search** powering RAG pipelines, often with BM25 scoring.
   _Comment:_ Promising since DuckDB can handle larger corpora efficiently with SQL-like queries.
   Example: [GitHub](https://github.com/tobilg/duckdb-fts-demo)

---

## ⚖️ Observations Across Approaches

- **SQL-only retrieval** is rare. Most projects either (a) use embeddings inside the SQL engine, or (b) use classical IR as a pre-filter.
- **Hierarchical designs** (HIRO, HighNoon, LlamaIndex tree, HMT) show strong interest in tree-based narrowing, but not yet as a SQL-native pipeline.
- **Your approach (hierarchical categorical narrowing + SQL storage)** seems novel: no embeddings, fixed branching factor, and simple categorical outputs from the LLM.
- **Closest practical repos:** `Memori` (SQL-native memory) and `RAG-without-Vector-DB` (classical IR retriever). These can serve as baseline comparisons.
