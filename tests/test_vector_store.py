"""
test_vector_store.py
Tests TF-IDF based similarity retrieval over the synthetic policy/FAQ docs.
Run with:
    pytest tests/test_vector_store.py -v
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_config
from src.generate_sample_data import generate_docs, generate_vector_eval
from src.vector_store import VectorStore


def _cfg_with_docs(tmp_path):
    cfg = get_config()
    cfg.paths.data_dir = str(tmp_path)
    cfg.paths.docs_dir = os.path.join(str(tmp_path), "docs")
    generate_docs(cfg.paths.docs_dir)
    generate_vector_eval(str(tmp_path))
    return cfg


def test_docs_load_correctly(tmp_path):
    cfg = _cfg_with_docs(tmp_path)
    store = VectorStore(cfg)
    assert len(store.doc_ids) == 6  # matches POLICY_DOCS count
    assert "shipping_policy.txt" in store.doc_ids


def test_search_returns_relevant_doc(tmp_path):
    cfg = _cfg_with_docs(tmp_path)
    store = VectorStore(cfg)
    result = store.search("What is your return policy?", top_k=1)
    assert len(result.docs) >= 1
    assert result.docs[0].doc_id == "return_policy.txt"


def test_search_shipping_query(tmp_path):
    cfg = _cfg_with_docs(tmp_path)
    store = VectorStore(cfg)
    result = store.search("How long does international shipping take?", top_k=1)
    assert result.docs[0].doc_id == "shipping_policy.txt"


def test_no_relevant_doc_returns_empty(tmp_path):
    cfg = _cfg_with_docs(tmp_path)
    cfg.vector.min_similarity = 0.99  # unreasonably high threshold
    store = VectorStore(cfg)
    result = store.search("completely unrelated gibberish xyz", top_k=3)
    assert result.docs == []
    assert "No relevant documents" in result.as_context_text()


def test_precision_at_k(tmp_path):
    cfg = _cfg_with_docs(tmp_path)
    store = VectorStore(cfg)

    import json
    with open(os.path.join(str(tmp_path), "vector_eval.json")) as f:
        eval_examples = json.load(f)

    precision = store.precision_at_k(eval_examples, k=1)
    assert 0.0 <= precision <= 1.0
    assert precision > 0.5  # should do reasonably well on these clear-cut queries
