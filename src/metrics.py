"""
metrics.py
Evaluation metrics for the two things this project actually needs to prove:
  1. Does the router send queries to the right source? (routing accuracy,
     plus a confusion matrix across sql/vector/hybrid)
  2. Is similarity-based retrieval actually finding the right document?
     (precision@k over the labeled vector_eval.json set)
"""

from collections import Counter
from typing import List, Tuple

from sklearn.metrics import classification_report, confusion_matrix


def routing_report(predictions: List[str], true_labels: List[str]) -> dict:
    labels = sorted(set(true_labels) | set(predictions))
    report = classification_report(
        true_labels, predictions, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(true_labels, predictions, labels=labels).tolist()
    return {"labels": labels, "report": report, "confusion_matrix": cm}


def route_distribution(predictions: List[str]) -> dict:
    return dict(Counter(predictions))
