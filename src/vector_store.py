"""
vector_store.py
Similarity-based retrieval over the policy/FAQ document corpus. Uses
TF-IDF + cosine similarity by default (no model download required, runs
fully offline). The `embed_fn` is swappable -- see the README for how to
plug in sentence-transformers or an API-based embedding model for
production use without changing any other code.
"""

import glob
import os
from dataclasses import dataclass, field
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Config


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float


@dataclass
class VectorResult:
    docs: List[RetrievedDoc] = field(default_factory=list)

    def as_context_text(self) -> str:
        if not self.docs:
            return "No relevant documents were found."
        parts = []
        for d in self.docs:
            parts.append(f"[{d.doc_id}] (similarity={d.score:.2f})\n{d.text}")
        return "\n\n".join(parts)


class VectorStore:
    """
    TF-IDF based similarity search. Loads every .txt file in `docs_dir`,
    vectorizes them once at init, and retrieves top-k most similar documents
    to a query using cosine similarity -- this is the "similarity-based
    context retrieval" the router relies on for unstructured knowledge.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self._load_docs()

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts) if self.doc_texts else None

    def _load_docs(self) -> None:
        paths = sorted(glob.glob(os.path.join(self.cfg.paths.docs_dir, "*.txt")))
        for path in paths:
            with open(path, "r") as f:
                text = f.read()
            self.doc_ids.append(os.path.basename(path))
            self.doc_texts.append(text)

    def search(self, query: str, top_k: int = None) -> VectorResult:
        if self.doc_matrix is None or len(self.doc_texts) == 0:
            return VectorResult(docs=[])

        top_k = top_k or self.cfg.vector.top_k
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_matrix).flatten()

        ranked_idx = sims.argsort()[::-1][:top_k]
        docs = []
        for idx in ranked_idx:
            score = float(sims[idx])
            if score < self.cfg.vector.min_similarity:
                continue
            docs.append(RetrievedDoc(
                doc_id=self.doc_ids[idx],
                text=self.doc_texts[idx],
                score=score,
            ))
        return VectorResult(docs=docs)

    def precision_at_k(self, eval_examples: List[dict], k: int = 1) -> float:
        """
        eval_examples: [{"query": ..., "relevant_doc": "shipping_policy.txt"}, ...]
        Returns fraction of queries where the known-relevant doc appears in
        the top-k retrieved results.
        """
        if not eval_examples:
            return 0.0
        hits = 0
        for ex in eval_examples:
            result = self.search(ex["query"], top_k=k)
            retrieved_ids = {d.doc_id for d in result.docs}
            if ex["relevant_doc"] in retrieved_ids:
                hits += 1
        return hits / len(eval_examples)
