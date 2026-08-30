"""
test_sql_store.py
Tests entity extraction and safe parameterized query dispatch against a
freshly generated synthetic SQLite database.
Run with:
    pytest tests/test_sql_store.py -v
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_config
from src.generate_sample_data import generate_sql_database
from src.sql_store import SQLStore


def _cfg_with_db(tmp_path):
    cfg = get_config()
    cfg.paths.data_dir = str(tmp_path)
    cfg.paths.sqlite_path = os.path.join(str(tmp_path), "company.db")
    generate_sql_database(cfg.paths.sqlite_path, num_customers=10, num_orders=30, seed=1)
    return cfg


def test_extract_order_id():
    assert SQLStore.extract_order_id("What is the status of order 1023?") == 1023
    assert SQLStore.extract_order_id("order number 1050 please") == 1050
    assert SQLStore.extract_order_id("no id here") is None


def test_extract_customer_id():
    assert SQLStore.extract_customer_id("Show me orders for customer 5") == 5
    assert SQLStore.extract_customer_id("customer id 12") == 12
    assert SQLStore.extract_customer_id("no customer mentioned") is None


def test_lookup_order_returns_rows(tmp_path):
    cfg = _cfg_with_db(tmp_path)
    store = SQLStore(cfg)
    # Order IDs are generated starting at 1001
    result = store.lookup_order(1001)
    assert len(result.rows) == 1
    assert result.rows[0]["order_id"] == 1001
    assert "customer" in result.rows[0]
    assert "product" in result.rows[0]


def test_lookup_order_nonexistent(tmp_path):
    cfg = _cfg_with_db(tmp_path)
    store = SQLStore(cfg)
    result = store.lookup_order(999999)
    assert len(result.rows) == 0
    assert "No matching records" in result.as_context_text()


def test_lookup_customer_orders(tmp_path):
    cfg = _cfg_with_db(tmp_path)
    store = SQLStore(cfg)
    result = store.lookup_customer_orders(customer_id=1, limit=5)
    assert isinstance(result.rows, list)
    for row in result.rows:
        assert "order_id" in row


def test_resolve_and_query_order_lookup(tmp_path):
    cfg = _cfg_with_db(tmp_path)
    store = SQLStore(cfg)
    result = store.resolve_and_query("What is the status of order 1001?")
    assert len(result.rows) == 1
    assert result.rows[0]["order_id"] == 1001


def test_resolve_and_query_no_entity(tmp_path):
    cfg = _cfg_with_db(tmp_path)
    store = SQLStore(cfg)
    result = store.resolve_and_query("What is your return policy?")
    assert result.rows == []
