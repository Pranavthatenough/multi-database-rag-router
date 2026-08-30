"""
prompt_builder.py
Builds the final prompt sent to the LLM, using a different template
depending on which route(s) the router selected. This is the "dynamic
prompt framework" piece: structured-only, unstructured-only, and hybrid
queries get differently-shaped instructions and context blocks, rather
than one generic "here's some context, answer the question" template.
"""

from dataclasses import dataclass

SQL_TEMPLATE = """You are a customer support assistant. Answer the user's question \
using ONLY the structured database records below. Be precise and reference \
specific fields (order ID, status, price, etc.) where relevant. If the \
records don't contain enough information, say so clearly.

Database records:
{sql_context}

User question: {query}

Answer:"""

VECTOR_TEMPLATE = """You are a customer support assistant. Answer the user's question \
using ONLY the policy/FAQ excerpts below. Summarize the relevant policy in \
plain language. If none of the excerpts are relevant, say so clearly.

Policy/FAQ excerpts:
{vector_context}

User question: {query}

Answer:"""

HYBRID_TEMPLATE = """You are a customer support assistant. Answer the user's question \
by combining the specific database record below with the relevant policy \
context that follows. Explain how the policy applies to this specific \
record. If either piece is missing, note what additional information would \
be needed.

Database record:
{sql_context}

Relevant policy/FAQ excerpts:
{vector_context}

User question: {query}

Answer:"""


@dataclass
class BuiltPrompt:
    route: str
    prompt_text: str


def build_prompt(route: str, query: str, sql_context: str = "", vector_context: str = "") -> BuiltPrompt:
    if route == "sql":
        text = SQL_TEMPLATE.format(sql_context=sql_context, query=query)
    elif route == "vector":
        text = VECTOR_TEMPLATE.format(vector_context=vector_context, query=query)
    else:  # hybrid
        text = HYBRID_TEMPLATE.format(sql_context=sql_context, vector_context=vector_context, query=query)
    return BuiltPrompt(route=route, prompt_text=text)
