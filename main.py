"""
main.py
CLI entry point for the Multi-Database RAG Router project.

Modes:
    demo         Run a fixed batch of example queries (sql/vector/hybrid),
                 print the routing decision + answer for each, and save a
                 markdown transcript to outputs/.
    interactive  Drop into a REPL where you can type your own queries.
    eval         Train/evaluate the router classifier (accuracy, confusion
                 matrix) and evaluate vector retrieval precision@k, saving
                 both to outputs/.

Examples:
    python -m src.generate_sample_data --data-dir data
    python main.py --mode demo
    python main.py --mode eval
    python main.py --mode interactive
    python main.py --mode demo --classifier rule_based
    python main.py --mode demo --use-openai
"""

import argparse
import json
import os

from src.config import get_config
from src.metrics import routing_report, route_distribution
from src.query_router import QueryRouter
from src.rag_pipeline import RAGPipeline
from src.sql_store import SQLStore
from src.utils import get_logger, set_seed, timer
from src.vector_store import VectorStore

logger = get_logger(__name__)

DEMO_QUERIES = [
    "What is the status of order 1023?",
    "What is your return policy?",
    "Can I return order 1023 given your return policy?",
    "How many orders has customer 5 placed?",
    "Do you ship internationally?",
    "Is customer 5's last order still eligible for a refund?",
    "What's the price of a Mechanical Keyboard?",
    "How do I reset my password?",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Database RAG Router")
    parser.add_argument("--mode", choices=["demo", "interactive", "eval"], default="demo")
    parser.add_argument("--classifier", choices=["tfidf_logreg", "rule_based"], default="tfidf_logreg")
    parser.add_argument("--use-openai", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_pipeline(args, cfg) -> RAGPipeline:
    cfg.router.classifier = args.classifier
    cfg.router.confidence_threshold = args.confidence_threshold
    cfg.vector.top_k = args.top_k
    cfg.llm.provider = "openai" if args.use_openai else "extractive"

    router = QueryRouter(cfg)
    if cfg.router.classifier == "tfidf_logreg":
        acc = router.train()
        logger.info(f"Router classifier trained. Held-out accuracy: {acc:.3f}")

    sql_store = SQLStore(cfg)
    vector_store = VectorStore(cfg)

    return RAGPipeline(cfg, router=router, sql_store=sql_store, vector_store=vector_store)


def run_demo(pipeline: RAGPipeline, cfg):
    lines = ["# Demo Query Results\n"]
    routes = []
    for query in DEMO_QUERIES:
        with timer() as t:
            response = pipeline.answer(query)
        routes.append(response.route)

        logger.info(f"Query: {query!r} -> route={response.route} "
                    f"(confidence={response.confidence:.2f}, method={response.routing_method}, "
                    f"{t.elapsed_ms:.1f}ms)")

        lines.append(f"## Query: {query}")
        lines.append(f"- **Route:** `{response.route}` (confidence={response.confidence:.2f}, "
                      f"method={response.routing_method}, latency={t.elapsed_ms:.1f}ms)")
        lines.append(f"- **Answer:**\n\n{response.answer}\n")

    out_path = os.path.join(cfg.paths.output_dir, "demo_results.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved demo transcript to {out_path}")
    logger.info(f"Route distribution: {route_distribution(routes)}")


def run_interactive(pipeline: RAGPipeline):
    print("Multi-Database RAG Router -- interactive mode. Type 'exit' to quit.\n")
    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        response = pipeline.answer(query)
        print(f"\n[route={response.route}, confidence={response.confidence:.2f}, "
              f"method={response.routing_method}]")
        print(response.answer)
        print()


def run_eval(pipeline: RAGPipeline, cfg):
    router = pipeline.router
    examples = router.load_labeled_queries()

    predictions = [router.classify(q).route for q, _ in examples]
    true_labels = [l for _, l in examples]

    report = routing_report(predictions, true_labels)
    routing_path = os.path.join(cfg.paths.output_dir, "routing_eval.json")
    with open(routing_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved routing evaluation to {routing_path}")
    logger.info(f"Overall routing accuracy: {report['report']['accuracy']:.3f}")

    vector_eval_path = os.path.join(cfg.paths.data_dir, "vector_eval.json")
    if os.path.exists(vector_eval_path):
        with open(vector_eval_path, "r") as f:
            vector_eval_examples = json.load(f)
        p_at_1 = pipeline.vector_store.precision_at_k(vector_eval_examples, k=1)
        p_at_3 = pipeline.vector_store.precision_at_k(vector_eval_examples, k=3)

        retrieval_path = os.path.join(cfg.paths.output_dir, "retrieval_eval.json")
        with open(retrieval_path, "w") as f:
            json.dump({"precision_at_1": p_at_1, "precision_at_3": p_at_3}, f, indent=2)
        logger.info(f"Saved retrieval evaluation to {retrieval_path}")
        logger.info(f"Vector retrieval precision@1={p_at_1:.3f}, precision@3={p_at_3:.3f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = get_config()

    logger.info(f"Config: mode={args.mode} | classifier={args.classifier} | "
                f"use_openai={args.use_openai} | top_k={args.top_k}")

    pipeline = build_pipeline(args, cfg)

    if args.mode == "demo":
        run_demo(pipeline, cfg)
    elif args.mode == "interactive":
        run_interactive(pipeline)
    elif args.mode == "eval":
        run_eval(pipeline, cfg)

    logger.info("Done.")


if __name__ == "__main__":
    main()
