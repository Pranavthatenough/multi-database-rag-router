"""
test_query_router.py
Tests for the routing classifier -- both rule-based and TF-IDF+LogReg
backends -- to make sure sql/vector/hybrid decisions are sane.
Run with:
    pytest tests/test_query_router.py -v
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_config
from src.query_router import QueryRouter


def _cfg_for(tmp_path, classifier="tfidf_logreg"):
    cfg = get_config()
    cfg.paths.data_dir = str(tmp_path)
    cfg.router.classifier = classifier
    return cfg


def _write_labeled_queries(tmp_path):
    import json
    examples = [
        {"query": "What is the status of order 1023?", "label": "sql"},
        {"query": "Show me all orders for customer 5", "label": "sql"},
        {"query": "What is your return policy?", "label": "vector"},
        {"query": "How do I reset my password?", "label": "vector"},
        {"query": "Can I return order 1023 given your return policy?", "label": "hybrid"},
        {"query": "Is customer 5's last order eligible for warranty?", "label": "hybrid"},
        {"query": "What is the price of a keyboard?", "label": "sql"},
        {"query": "Do you ship internationally?", "label": "vector"},
    ]
    with open(os.path.join(tmp_path, "labeled_queries.json"), "w") as f:
        json.dump(examples, f)
    return examples


def test_rule_based_sql_route(tmp_path):
    cfg = _cfg_for(tmp_path, classifier="rule_based")
    router = QueryRouter(cfg)
    decision = router.classify("What is the status of order 1023?")
    assert decision.route == "sql"


def test_rule_based_vector_route(tmp_path):
    cfg = _cfg_for(tmp_path, classifier="rule_based")
    router = QueryRouter(cfg)
    decision = router.classify("What is your return policy?")
    assert decision.route == "vector"


def test_rule_based_hybrid_route(tmp_path):
    cfg = _cfg_for(tmp_path, classifier="rule_based")
    router = QueryRouter(cfg)
    decision = router.classify("Can I return order 1023 given your return policy?")
    assert decision.route == "hybrid"


def test_tfidf_router_trains_and_predicts(tmp_path):
    examples_json = _write_labeled_queries(tmp_path)
    cfg = _cfg_for(tmp_path, classifier="tfidf_logreg")
    router = QueryRouter(cfg)

    examples = [(e["query"], e["label"]) for e in examples_json]
    accuracy = router.train(examples)
    assert 0.0 <= accuracy <= 1.0

    decision = router.classify("What is the status of order 1023?")
    assert decision.route in ("sql", "vector", "hybrid")
    assert decision.method == "tfidf_logreg"


def test_low_confidence_falls_back_to_hybrid(tmp_path):
    examples_json = _write_labeled_queries(tmp_path)
    cfg = _cfg_for(tmp_path, classifier="tfidf_logreg")
    cfg.router.confidence_threshold = 0.99  # force everything below threshold
    router = QueryRouter(cfg)
    examples = [(e["query"], e["label"]) for e in examples_json]
    router.train(examples)

    decision = router.classify("Some ambiguous query about stuff")
    assert decision.route == "hybrid"


def test_evaluate_returns_fraction(tmp_path):
    examples_json = _write_labeled_queries(tmp_path)
    cfg = _cfg_for(tmp_path, classifier="tfidf_logreg")
    router = QueryRouter(cfg)
    examples = [(e["query"], e["label"]) for e in examples_json]
    router.train(examples)

    accuracy = router.evaluate(examples)
    assert 0.0 <= accuracy <= 1.0
