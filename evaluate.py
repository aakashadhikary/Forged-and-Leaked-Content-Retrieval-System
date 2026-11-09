"""
evaluate.py — Evaluation harness for the Content Forensics prototype.

RESEARCH DELIVERABLE
─────────────────────
For every transformed variant in the corpus, this script treats it as a query
and checks whether its true original ranks #1 under each of four methods:
  (a) pHash alone
  (b) Face embedding alone
  (c) CLIP alone
  (d) Fusion (weighted average of active components)

Tallies and prints a Top-1 Accuracy table — the headline research result.

USAGE
─────
  python evaluate.py [--corpus corpus/] [--cache cache/fingerprint_index.pkl]
                     [--manifest corpus/manifest.json]
                     [--out eval_results.csv]
                     [--chart accuracy_chart.png]
                     [--weights 0.2 0.4 0.4]
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import textwrap
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless/shell use
import matplotlib.pyplot as plt
import numpy as np

import fingerprints as fp
from fingerprints import DEFAULT_WEIGHTS

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found at {cache_path}. Run index_corpus.py first."
        )
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}.")
    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)
    return {e["filename"]: e for e in entries}


def _clamp(val: float) -> float:
    return max(0.0, val)


def score_pair(
    q_fp: dict,
    c_fp: dict,
    weights: dict[str, float],
) -> dict[str, Optional[float]]:
    """
    Return all four scores for a query–corpus pair.
    Face score is None if either side has no face embedding.
    """
    phash_sim = fp.phash_similarity(q_fp["phash"], c_fp["phash"])

    q_face = q_fp["face_emb"]
    c_face = c_fp["face_emb"]
    face_sim: Optional[float] = None
    if q_face is not None and c_face is not None:
        face_sim = _clamp(fp.cosine_similarity(q_face, c_face))

    clip_sim = _clamp(fp.cosine_similarity(q_fp["clip_emb"], c_fp["clip_emb"]))

    fuse = fp.fusion_score(phash_sim, face_sim, clip_sim, weights)

    return {
        "phash":  phash_sim,
        "face":   face_sim,
        "clip":   clip_sim,
        "fusion": fuse,
    }


def rank_corpus(
    query_name: str,
    query_fp: dict,
    index: dict,
    weights: dict[str, float],
) -> dict[str, int]:
    """
    For each method, compute the rank of the true original among the
    *rest of the corpus* (query itself excluded).

    Returns { method_name: rank_of_original (1-indexed) }.
    If the original is not in the index, returns a large rank.
    """
    # Build manifest entry for query to find its original_id
    # (passed externally as original_id argument)
    scores: dict[str, list[tuple[str, float]]] = {
        "phash":  [],
        "face":   [],
        "clip":   [],
        "fusion": [],
    }

    for fname, c_fp in index.items():
        if fname == query_name:
            continue  # exclude query from its own ranking
        pair = score_pair(query_fp, c_fp, weights)
        # None face → treat as 0 for individual-method ranking only
        scores["phash"].append((fname, pair["phash"]))
        scores["face"].append((fname, pair["face"] if pair["face"] is not None else -1.0))
        scores["clip"].append((fname, pair["clip"]))
        scores["fusion"].append((fname, pair["fusion"]))

    ranks: dict[str, int] = {}
    for method, fname_score_list in scores.items():
        ranked = sorted(fname_score_list, key=lambda x: x[1], reverse=True)
        ranks[method] = {fname: i + 1 for i, (fname, _) in enumerate(ranked)}

    return ranks


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    corpus_dir: Path,
    cache_path: Path,
    manifest_path: Path,
    weights: dict[str, float],
) -> tuple[list[dict], dict[str, float]]:
    """
    Returns:
        per_query_results: list of per-query dicts
        accuracy: { method: top1_accuracy_fraction }
    """
    index = load_cache(cache_path)
    manifest = load_manifest(manifest_path)

    # Only evaluate on variant entries (not distractors, not originals)
    # A "variant" has a non-None original_id and transform != "original"
    variant_entries = [
        entry for entry in manifest.values()
        if entry.get("original_id") is not None
        and entry.get("transform") not in ("original", "distractor")
        and entry["filename"] in index
    ]

    if not variant_entries:
        raise ValueError(
            "No variant entries found in manifest. "
            "Make sure build_corpus.py ran successfully and produced variants."
        )

    print(f"Evaluating {len(variant_entries)} variant queries…\n")
    methods = ["phash", "face", "clip", "fusion"]
    top1_hits = {m: 0 for m in methods}
    per_query: list[dict] = []

    for i, entry in enumerate(variant_entries, start=1):
        query_name = entry["filename"]
        original_id = entry["original_id"]
        true_original_name = f"{original_id}.jpg"
        transform = entry["transform"]

        query_fp = index.get(query_name)
        if query_fp is None:
            print(f"  SKIP {query_name}: not in index.")
            continue

        if true_original_name not in index:
            print(f"  SKIP {query_name}: original {true_original_name} not indexed.")
            continue

        # Rank original among rest-of-corpus
        rank_maps = rank_corpus(query_name, query_fp, index, weights)

        row = {
            "query":          query_name,
            "original_id":    original_id,
            "transform":      transform,
        }
        for m in methods:
            rank = rank_maps[m].get(true_original_name, 9999)
            row[f"{m}_rank"] = rank
            row[f"{m}_top1"] = (rank == 1)
            if rank == 1:
                top1_hits[m] += 1

        per_query.append(row)

        # Progress
        symbol = {True: "✓", False: "✗"}
        hit_str = " ".join(
            f"{m}:{symbol[row[f'{m}_top1']]}" for m in methods
        )
        print(f"  [{i:>3}/{len(variant_entries)}] {query_name:<40} {hit_str}")

    n = len(per_query)
    accuracy = {m: top1_hits[m] / n if n > 0 else 0.0 for m in methods}
    return per_query, accuracy


# ──────────────────────────────────────────────────────────────────────────────
# Display + Export
# ──────────────────────────────────────────────────────────────────────────────

METHOD_LABELS = {
    "phash":  "pHash (baseline)",
    "face":   "Face Embedding (FaceNet)",
    "clip":   "CLIP (ViT-B-32)",
    "fusion": "Fusion (weighted avg)",
}


def print_summary(accuracy: dict[str, float]) -> None:
    print()
    print("=" * 50)
    print("  EVALUATION SUMMARY — Top-1 Accuracy")
    print("=" * 50)
    print(f"  {'Method':<30}  {'Top-1 Acc':>10}")
    print("-" * 50)
    for method in ["phash", "face", "clip", "fusion"]:
        label = METHOD_LABELS[method]
        pct = accuracy[method] * 100
        print(f"  {label:<30}  {pct:>9.1f}%")
    print("=" * 50)
    print()


def export_csv(per_query: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not per_query:
        return
    fieldnames = list(per_query[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write("# Content Forensics — Per-Query Evaluation Results\n")
        f.write("# NOTE: All corpus data is SIMULATED demo data.\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_query)
    print(f"Per-query CSV saved → {out_path}")


def save_accuracy_chart(accuracy: dict[str, float], out_path: Path) -> None:
    """Save a clean horizontal bar chart of Top-1 accuracy per method."""
    methods = ["phash", "face", "clip", "fusion"]
    labels = [METHOD_LABELS[m] for m in methods]
    values = [accuracy[m] * 100 for m in methods]

    # Colour palette — dark theme with colour progression
    colours = ["#4a4a6a", "#5b8dd9", "#3db8a0", "#e06c4b"]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars = ax.barh(labels, values, color=colours, height=0.55, edgecolor="none")

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlim(0, 110)
    ax.set_xlabel("Top-1 Accuracy (%)", color="#cccccc", fontsize=11)
    ax.set_title(
        "Content Forensics — Top-1 Retrieval Accuracy by Method",
        color="white",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax.tick_params(colors="#cccccc", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.xaxis.label.set_color("#aaaaaa")
    ax.yaxis.set_tick_params(pad=8)

    ax.axvline(x=100, color="#555", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Accuracy chart saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Evaluate Top-1 retrieval accuracy for all four fingerprinting methods
            (pHash, face embedding, CLIP, fusion) across all corpus variants.

            This is the main research deliverable: it tells you which method
            is best at finding transformed/manipulated copies of a reference image.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("corpus"),
        help="Corpus directory (default: corpus/)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("cache/fingerprint_index.pkl"),
        help="Fingerprint cache file (default: cache/fingerprint_index.pkl)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("corpus/manifest.json"),
        help="Corpus manifest (default: corpus/manifest.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval_results.csv"),
        help="Per-query CSV output path (default: eval_results.csv)",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=Path("accuracy_chart.png"),
        help="Bar chart output path (default: accuracy_chart.png)",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        metavar=("W_PHASH", "W_FACE", "W_CLIP"),
        default=[DEFAULT_WEIGHTS["phash"], DEFAULT_WEIGHTS["face"], DEFAULT_WEIGHTS["clip"]],
        help="Fusion weights (default: 0.2 0.4 0.4)",
    )
    args = parser.parse_args()

    weights = {
        "phash": args.weights[0],
        "face":  args.weights[1],
        "clip":  args.weights[2],
    }

    print("=" * 60)
    print("Content Forensics — Evaluation")
    print(f"Fusion weights: pHash={weights['phash']}, face={weights['face']}, CLIP={weights['clip']}")
    print("=" * 60 + "\n")

    per_query, accuracy = evaluate(
        corpus_dir=args.corpus,
        cache_path=args.cache,
        manifest_path=args.manifest,
        weights=weights,
    )

    print_summary(accuracy)
    export_csv(per_query, out_path=args.out)
    save_accuracy_chart(accuracy, out_path=args.chart)

    print("\nDone.")


if __name__ == "__main__":
    main()
