"""
config.py
Central configuration for the Multi-Database RAG Router project.
"""

import os
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    data_dir: str = "data"
    docs_dir: str = field(init=False)
    sqlite_path: str = field(init=False)
    output_dir: str = "outputs"

    def __post_init__(self):
        self.docs_dir = os.path.join(self.data_dir, "docs")
        self.sqlite_path = os.path.join(self.data_dir, "company.db")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class RouterConfig:
    # Classifier used to decide which data source(s) a query needs.
    classifier: str = "tfidf_logreg"   # "tfidf_logreg" or "rule_based"
    confidence_threshold: float = 0.55  # below this, fall back to hybrid (search both)
    seed: int = 42


@dataclass
class VectorConfig:
    top_k: int = 3
    min_similarity: float = 0.05  # below this, treat as "no relevant doc found"


@dataclass
class SQLConfig:
    max_rows_returned: int = 10


@dataclass
class LLMConfig:
    provider: str = "extractive"   # "extractive" (offline, no API key) or "openai"
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.0


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    sql: SQLConfig = field(default_factory=SQLConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def get_config() -> Config:
    return Config()
