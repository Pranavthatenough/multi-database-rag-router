"""
query_router.py
Classifies an incoming query into one of three routes:
    "sql"     -> structured lookup only (order status, customer records, prices)
    "vector"  -> unstructured knowledge only (policies, FAQs, how-tos)
    "hybrid"  -> needs both a specific record AND general policy knowledge

Two classifier backends are supported:
    - "tfidf_logreg": TF-IDF features + Logistic Regression, trained on the
      labeled examples in data/labeled_queries.json. This is the primary,
      more robust option and the one used for reported routing accuracy.
    - "rule_based": simple keyword heuristics (entity mention + policy
      keyword co-occurrence). Useful as a zero-training-data fallback and
      as a sanity baseline to compare the learned classifier against.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import Config

POLICY_KEYWORDS = [
    "policy", "return", "refund", "warranty", "shipping", "discount",
    "password", "account", "payment", "eligible", "coverage", "claim",
]
ENTITY_KEYWORDS = ["order", "customer"]


@dataclass
class RouteDecision:
    route: str            # "sql" | "vector" | "hybrid"
    confidence: float
    method: str            # "tfidf_logreg" | "rule_based"


class QueryRouter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vectorizer: TfidfVectorizer = None
        self.classifier: LogisticRegression = None
        self._trained = False

    # ---- Training (tfidf_logreg backend) ----

    def load_labeled_queries(self) -> List[Tuple[str, str]]:
        path = os.path.join(self.cfg.paths.data_dir, "labeled_queries.json")
        with open(path, "r") as f:
            data = json.load(f)
        return [(d["query"], d["label"]) for d in data]

    def train(self, examples: List[Tuple[str, str]] = None) -> float:
        """
        Trains the TF-IDF + LogisticRegression classifier. Returns held-out
        accuracy on a train/test split of the labeled examples so routing
        quality is measurable and reportable.
        """
        if examples is None:
            examples = self.load_labeled_queries()

        queries = [q for q, _ in examples]
        labels = [l for _, l in examples]

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                queries, labels, test_size=0.25, random_state=self.cfg.router.seed, stratify=labels
            )
        except ValueError:
            # Stratified split needs at least (1 / test_size) examples per
            # class; too-small datasets (e.g. in unit tests) fall back to a
            # plain unstratified split rather than crashing.
            X_train, X_test, y_train, y_test = train_test_split(
                queries, labels, test_size=0.25, random_state=self.cfg.router.seed
            )

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(X_train_vec, y_train)
        self._trained = True

        accuracy = self.classifier.score(X_test_vec, y_test)
        return accuracy

    # ---- Inference ----

    def classify_rule_based(self, query: str) -> RouteDecision:
        query_lower = query.lower()
        has_entity = any(k in query_lower for k in ENTITY_KEYWORDS)
        has_policy = any(k in query_lower for k in POLICY_KEYWORDS)

        if has_entity and has_policy:
            return RouteDecision(route="hybrid", confidence=0.7, method="rule_based")
        if has_entity:
            return RouteDecision(route="sql", confidence=0.7, method="rule_based")
        if has_policy:
            return RouteDecision(route="vector", confidence=0.7, method="rule_based")
        # Default fallback: try vector search (general knowledge) first
        return RouteDecision(route="vector", confidence=0.4, method="rule_based")

    def classify(self, query: str) -> RouteDecision:
        if self.cfg.router.classifier == "rule_based" or not self._trained:
            return self.classify_rule_based(query)

        query_vec = self.vectorizer.transform([query])
        probs = self.classifier.predict_proba(query_vec)[0]
        classes = self.classifier.classes_
        best_idx = probs.argmax()
        route = str(classes[best_idx])  # cast off numpy.str_ so JSON/logging behave normally
        confidence = float(probs[best_idx])

        if confidence < self.cfg.router.confidence_threshold:
            # Low-confidence predictions default to hybrid: safer to search
            # both sources than risk missing context entirely.
            return RouteDecision(route="hybrid", confidence=confidence, method="tfidf_logreg")

        return RouteDecision(route=route, confidence=confidence, method="tfidf_logreg")

    def evaluate(self, examples: List[Tuple[str, str]] = None) -> float:
        """Routing accuracy against labeled examples, using current classify()."""
        if examples is None:
            examples = self.load_labeled_queries()
        correct = 0
        for query, true_label in examples:
            decision = self.classify(query)
            if decision.route == true_label:
                correct += 1
        return correct / len(examples)
