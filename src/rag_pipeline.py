"""
rag_pipeline.py
Orchestrates the full flow for a single query:
    query -> QueryRouter.classify -> retrieve from SQL and/or vector store
          -> prompt_builder.build_prompt (route-specific template)
          -> llm_client.generate
          -> RAGResponse (answer + full trace, for transparency/debugging)
"""

from dataclasses import dataclass, field
from typing import Optional

from src.config import Config
from src.llm_client import BaseLLM, build_llm
from src.prompt_builder import build_prompt
from src.query_router import QueryRouter, RouteDecision
from src.sql_store import SQLStore, SQLResult
from src.vector_store import VectorResult, VectorStore


@dataclass
class RAGResponse:
    query: str
    route: str
    confidence: float
    routing_method: str
    sql_result: Optional[SQLResult]
    vector_result: Optional[VectorResult]
    prompt_text: str
    answer: str


class RAGPipeline:
    def __init__(self, cfg: Config, router: QueryRouter = None,
                 sql_store: SQLStore = None, vector_store: VectorStore = None,
                 llm: BaseLLM = None):
        self.cfg = cfg
        self.router = router or QueryRouter(cfg)
        self.sql_store = sql_store or SQLStore(cfg)
        self.vector_store = vector_store or VectorStore(cfg)
        self.llm = llm or build_llm(cfg)

    def answer(self, query: str) -> RAGResponse:
        decision: RouteDecision = self.router.classify(query)

        sql_result, vector_result = None, None
        sql_context, vector_context = "", ""

        if decision.route in ("sql", "hybrid"):
            sql_result = self.sql_store.resolve_and_query(query)
            sql_context = sql_result.as_context_text()

        if decision.route in ("vector", "hybrid"):
            vector_result = self.vector_store.search(query)
            vector_context = vector_result.as_context_text()

        built = build_prompt(
            route=decision.route,
            query=query,
            sql_context=sql_context,
            vector_context=vector_context,
        )
        answer_text = self.llm.generate(built.prompt_text)

        return RAGResponse(
            query=query,
            route=decision.route,
            confidence=decision.confidence,
            routing_method=decision.method,
            sql_result=sql_result,
            vector_result=vector_result,
            prompt_text=built.prompt_text,
            answer=answer_text,
        )
