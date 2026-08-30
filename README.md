# Multi-Database RAG Router

A dynamic prompt framework that routes natural language queries to the
correct data source — a SQL database or a similarity-searched document
store — before generating an answer, instead of blindly stuffing every
query into one generic retrieval pipeline.

## Problem Statement

Most RAG demos assume a single knowledge source: one vector database, one
document set. Real systems usually have several — structured records (a
SQL database of orders/customers) and unstructured knowledge (policy docs,
FAQs) — and a query might need one, the other, or both. This project builds
a **query router** that classifies each incoming query and dynamically
selects which source(s) to retrieve from and which prompt template to use,
improving relevance and avoiding wasted/irrelevant retrieval calls.

## Example

| Query | Route | Why |
|---|---|---|
| "What is the status of order 1023?" | `sql` | Needs a specific structured record |
| "What is your return policy?" | `vector` | General knowledge, no specific record needed |
| "Can I return order 1023 given your return policy?" | `hybrid` | Needs the order record *and* the policy text |

## Architecture

```
                     ┌─────────────────────┐
        query ──────▶│    Query Router       │
                     │ (TF-IDF + LogReg /     │
                     │  rule-based fallback)  │
                     └──────────┬────────────┘
                                │  route: sql / vector / hybrid
              ┌─────────────────┼─────────────────┐
              ▼                                     ▼
     ┌─────────────────┐                  ┌─────────────────────┐
     │   SQL Store       │                  │   Vector Store        │
     │ (SQLite, entity    │                  │ (TF-IDF + cosine       │
     │  extraction, safe   │                  │  similarity over        │
     │  parameterized      │                  │  policy/FAQ docs)       │
     │  queries)           │                  │                         │
     └─────────┬─────────┘                  └───────────┬─────────────┘
               │                                          │
               └─────────────────┬────────────────────────┘
                                 ▼
                     ┌─────────────────────────┐
                     │   Dynamic Prompt Builder  │
                     │ (sql / vector / hybrid     │
                     │  templates)                │
                     └───────────┬───────────────┘
                                 ▼
                     ┌─────────────────────────┐
                     │      LLM Client            │
                     │ (offline extractive by     │
                     │  default, OpenAI optional)  │
                     └─────────────────────────┘
                                 ▼
                              Answer
```

## Project Structure

```
multi_database_rag_router/
├── data/
│   ├── company.db                  # SQLite: customers, products, orders
│   ├── docs/                        # Policy/FAQ .txt documents
│   ├── labeled_queries.json         # Labeled sql/vector/hybrid training examples
│   └── vector_eval.json             # Query -> known-relevant-doc pairs for precision@k
├── outputs/
│   ├── demo_results.md              # Transcript of demo queries + answers
│   ├── routing_eval.json            # Routing accuracy + confusion matrix
│   └── retrieval_eval.json          # Vector retrieval precision@1 / precision@3
├── tests/
│   ├── test_query_router.py
│   ├── test_sql_store.py
│   └── test_vector_store.py
├── src/
│   ├── config.py                    # All settings & paths
│   ├── utils.py                      # Logging, seeding, timing
│   ├── generate_sample_data.py       # Synthetic SQLite DB + docs + labeled queries
│   ├── sql_store.py                  # Entity extraction + safe parameterized SQL
│   ├── vector_store.py               # TF-IDF similarity search over documents
│   ├── query_router.py               # sql/vector/hybrid classifier
│   ├── prompt_builder.py             # Route-specific dynamic prompt templates
│   ├── llm_client.py                 # Offline extractive + optional OpenAI backend
│   ├── rag_pipeline.py               # End-to-end orchestration
│   └── metrics.py                    # Routing accuracy, confusion matrix
├── main.py                           # CLI: demo / interactive / eval modes
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python -m src.generate_sample_data --data-dir data
```

## Usage

```bash
# Run a fixed batch of example queries, see routing decisions + answers
python main.py --mode demo

# Type your own queries interactively
python main.py --mode interactive

# Evaluate routing accuracy + retrieval precision@k
python main.py --mode eval

# Use the simpler rule-based router instead of the trained classifier
python main.py --mode demo --classifier rule_based

# Use a real OpenAI model for generation instead of the offline extractive fallback
export OPENAI_API_KEY=sk-...
python main.py --mode demo --use-openai
```

## Why TF-IDF instead of a neural embedding model?

The vector store uses TF-IDF + cosine similarity rather than a downloaded
embedding model, so the entire project runs offline with zero API keys and
zero large downloads. Swapping in a neural embedding model is a small,
isolated change — replace `VectorStore.__init__`'s `TfidfVectorizer` with
a `sentence-transformers` encoder (or a hosted embeddings API call), keep
everything else (routing, prompt templates, pipeline) unchanged.

## Why the LLM defaults to "extractive" mode

Without an API key, `ExtractiveLLM` simply returns the retrieved context in
a readable format instead of fabricating an answer — this keeps the whole
pipeline runnable and testable end-to-end without any paid dependency,
while still proving the routing + retrieval logic works correctly. Pass
`--use-openai` with `OPENAI_API_KEY` set for real free-text generation.

## Results

*(Generated by `python main.py --mode eval` — see `outputs/routing_eval.json`
and `outputs/retrieval_eval.json` for the full numbers from your own run)*

| Metric | Value (from included sample run) |
|---|---|
| Routing accuracy (tfidf_logreg, all 50 labeled examples) | 0.820 |
| Vector retrieval precision@1 (8 labeled eval queries) | 0.875 |
| Vector retrieval precision@3 (8 labeled eval queries) | 0.875 |

Exact numbers vary slightly with the train/test split seed — run `--mode
eval` yourself for the authoritative numbers on your data. Note the router
deliberately trades some accuracy for safety: queries near the confidence
threshold fall back to `hybrid` (searching both sources) rather than
risking a wrong single-source guess, so a chunk of "misclassified" hybrid
predictions are actually the intended conservative behavior, not routing
failures.

## Limitations

- **SQL entity extraction is regex/keyword-based**, not a full text-to-SQL
  model. This is a deliberate safety choice (no LLM-generated raw SQL is
  ever executed) but means very unusual phrasings may not resolve to the
  right query.
- **TF-IDF retrieval** is lexical, not semantic — it won't catch queries
  that are conceptually related but share no vocabulary with the source
  documents. A neural embedding model would close this gap.
- **The router classifier is trained on a small (50-example) synthetic
  labeled set** — in a real deployment you'd want hundreds+ of real user
  queries logged and labeled over time.
- **The offline LLM is extractive, not generative** — it proves the
  pipeline works but doesn't produce polished natural-language answers
  unless `--use-openai` is enabled.

## Future Work

- Swap TF-IDF for a neural embedding model (sentence-transformers or a
  hosted embeddings API) for semantic rather than lexical retrieval.
- Add a confidence-calibrated abstention path: if neither source returns
  a confident match, ask a clarifying question instead of guessing.
- Log real user queries and retrain the router periodically (active
  learning loop).
- Add a proper text-to-SQL layer with strict schema-constrained generation
  and a SQL validator/sandbox, instead of regex entity extraction, for
  more complex structured queries.
- Add a third data source (e.g. a document store for long-form manuals) to
  test whether the router generalizes past a binary sql/vector choice.
