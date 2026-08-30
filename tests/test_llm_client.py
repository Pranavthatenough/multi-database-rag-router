"""
test_llm_client.py
Tests ExtractiveLLM's ability to correctly detect whether a hybrid prompt
has *any* real retrieved content, even when one of the two sources (SQL or
vector) legitimately found nothing. This guards against a real bug where a
"No matching records" placeholder in one half of a hybrid prompt caused the
whole answer to be discarded even when the other half had good content.
Run with:
    pytest tests/test_llm_client.py -v
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_client import ExtractiveLLM
from src.prompt_builder import build_prompt


def test_sql_only_with_results():
    llm = ExtractiveLLM()
    prompt = build_prompt(
        route="sql", query="What is the status of order 1023?",
        sql_context="Order lookup for order_id=1023:\n - order_id: 1023, status: Refunded",
    )
    answer = llm.generate(prompt.prompt_text)
    assert "couldn't find" not in answer.lower()
    assert "1023" in answer


def test_vector_only_with_results():
    llm = ExtractiveLLM()
    prompt = build_prompt(
        route="vector", query="What is your return policy?",
        vector_context="[return_policy.txt] (similarity=0.46)\nReturn and Refund Policy...",
    )
    answer = llm.generate(prompt.prompt_text)
    assert "couldn't find" not in answer.lower()
    assert "Return and Refund" in answer


def test_sql_only_no_results():
    llm = ExtractiveLLM()
    prompt = build_prompt(
        route="sql", query="What is the status of order 999999?",
        sql_context="No matching records were found in the database.",
    )
    answer = llm.generate(prompt.prompt_text)
    assert "couldn't find" in answer.lower()


def test_hybrid_sql_empty_but_vector_has_content():
    """
    Regression test for the bug: SQL side legitimately has no match, but
    the vector side found a relevant policy doc. The overall answer must
    NOT be discarded just because "No matching records" appears somewhere
    in the combined prompt.
    """
    llm = ExtractiveLLM()
    prompt = build_prompt(
        route="hybrid",
        query="What is your return policy?",
        sql_context="No matching records were found in the database.",
        vector_context="[return_policy.txt] (similarity=0.46)\nReturn and Refund Policy: items can be returned within 30 days.",
    )
    answer = llm.generate(prompt.prompt_text)
    assert "couldn't find" not in answer.lower()
    assert "Return and Refund Policy" in answer


def test_hybrid_vector_empty_but_sql_has_content():
    """Mirror case: vector side empty, SQL side has a real record."""
    llm = ExtractiveLLM()
    prompt = build_prompt(
        route="hybrid",
        query="Can I return order 1023?",
        sql_context="Order lookup for order_id=1023:\n - order_id: 1023, status: Delivered",
        vector_context="No relevant documents were found.",
    )
    answer = llm.generate(prompt.prompt_text)
    assert "couldn't find" not in answer.lower()
    assert "1023" in answer


def test_hybrid_both_empty():
    llm = ExtractiveLLM()
    prompt = build_prompt(
        route="hybrid",
        query="Some nonsense query",
        sql_context="No matching records were found in the database.",
        vector_context="No relevant documents were found.",
    )
    answer = llm.generate(prompt.prompt_text)
    assert "couldn't find" in answer.lower()
