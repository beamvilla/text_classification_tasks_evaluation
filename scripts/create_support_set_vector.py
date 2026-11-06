"""Embed the support sets in data/ with an OpenAI-compatible embedding server, concurrently.

Run from the project root (it reads ./config/models.toml through src):
    python scripts/create_support_set_vector.py
    python scripts/create_support_set_vector.py --workers 16
    python scripts/create_support_set_vector.py --benchmark          # compare worker counts, saves nothing
    python scripts/create_support_set_vector.py --base-url https://api.openai.com/v1 --api-key "$OPENAI_API_KEY" \
        --model text-embedding-3-small

Each support set is written to <output-dir>/<name>.json as
    {"model": ..., "dimension": ..., "items": [{"review": ..., "label": ..., "embedding": [...]}, ...]}
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.repositories.openai_embedding_repository import OpenAIEmbeddingRepository  # noqa: E402

DEFAULT_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "http://localhost:8080/v1")
DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embeddings-inference")

# name -> (csv path, text column, label column)
SUPPORT_SETS = {
    "support_set_intent": (PROJECT_ROOT / "data/support_set_intent.csv", "review", "intent"),
    "support_set_sentiment": (PROJECT_ROOT / "data/support_set_sentiment.csv", "review", "label"),
}


def load_support_set(path: Path, text_column: str, label_column: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return [{"review": r[text_column].strip(), "label": r[label_column].strip()} for r in csv.DictReader(f)]


def embed_all(repo: OpenAIEmbeddingRepository, texts: list[str], model: str, workers: int) -> tuple[list[list[float]], list[float]]:
    """Embed `texts` with `workers` parallel requests. Returns vectors in input order and each request's latency."""

    def one(text: str) -> tuple[list[float], float]:
        start = time.perf_counter()
        vector = repo.get_embedding(text, model=model)
        return vector, time.perf_counter() - start

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, texts))  # map keeps the input order
    return [v for v, _ in results], [t for _, t in results]


def describe(label: str, n: int, total: float, latencies: list[float]) -> str:
    mean = sum(latencies) / len(latencies)
    return f"{label}: {n} texts in {total:.2f}s = {n / total:.1f} texts/s (mean request latency {mean * 1000:.0f} ms)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="default: $EMBEDDING_BASE_URL or %(default)s")
    ap.add_argument("--api-key", default="-", help="a local server ignores it; default %(default)r")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="embedding model name the server expects; default %(default)s")
    ap.add_argument("--workers", type=int, default=8, help="parallel requests; 1 = sequential")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/support_set_vectors")
    ap.add_argument("--benchmark", action="store_true", help="time several worker counts over all texts and exit without saving")
    args = ap.parse_args()

    repo = OpenAIEmbeddingRepository(base_url=args.base_url, api_key=args.api_key)
    datasets = {name: load_support_set(*spec) for name, spec in SUPPORT_SETS.items()}
    all_texts = [row["review"] for rows in datasets.values() for row in rows]
    print(f"{len(all_texts)} texts from {len(datasets)} support sets -> {args.base_url} (model {args.model})")

    if args.benchmark:
        embed_all(repo, all_texts[:4], args.model, 1)  # warm-up so the first run does not pay for connection setup
        for workers in (1, 2, 4, 8, 16, 32):
            start = time.perf_counter()
            _, latencies = embed_all(repo, all_texts, args.model, workers)
            print(describe(f"workers={workers:<2}", len(all_texts), time.perf_counter() - start, latencies))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in datasets.items():
        start = time.perf_counter()
        vectors, latencies = embed_all(repo, [r["review"] for r in rows], args.model, args.workers)
        total = time.perf_counter() - start

        dimensions = {len(v) for v in vectors}
        if len(dimensions) != 1:
            raise RuntimeError(f"{name}: server returned vectors of different sizes: {sorted(dimensions)}")
        items = [{**row, "embedding": vector} for row, vector in zip(rows, vectors)]

        out = args.output_dir / f"{name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "dimension": dimensions.pop(), "items": items}, f, ensure_ascii=False)
        print(describe(f"{name} (workers={args.workers})", len(rows), total, latencies) + f" -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
