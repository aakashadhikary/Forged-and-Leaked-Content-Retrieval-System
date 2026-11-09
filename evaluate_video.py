"""
evaluate_video.py — Evaluation harness for the video pipeline.

For every transformed variant in the video corpus, treats it as a query and
checks whether a video sharing the same source_id ranks in the top results
under each of five methods.

Methods compared:
  (a) pHash alone
  (b) FaceNet alone
  (c) CLIP alone
  (d) Frame fusion WITHOUT temporal alignment (fusion_nt)
  (e) Frame fusion WITH temporal alignment (fusion)

Metrics: Top-1, Top-5, MRR, localization error (for trimmed queries).

USAGE
─────
  python evaluate_video.py [--corpus video_corpus/]
                           [--cache cache/video_index.pkl]
                           [--out video_eval_results.csv]
                           [--summary video_eval_summary.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from tqdm import tqdm

from fingerprints import DEFAULT_WEIGHTS
from search_video import (
    compute_frame_scores,
    no_align_score,
    temporal_align,
    MIN_FRAME_SCORE,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def load_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────────────
METHODS = ["phash", "face", "clip", "fusion_nt", "fusion"]
METHOD_LABELS = {
    "phash":     "pHash (baseline)",
    "face":      "Face (FaceNet)",
    "clip":      "CLIP (ViT-B-32)",
    "fusion_nt": "Fusion (no align)",
    "fusion":    "Fusion + Temporal",
}


def evaluate(
    corpus_dir: Path,
    cache_path: Path,
    manifest_path: Path,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    bin_width: float = 1.0,
) -> tuple[list[dict], dict]:
    """
    Run evaluation. Returns (per_query_results, summary_dict).
    """
    index = load_cache(cache_path)
    manifest = load_manifest(manifest_path)
    manifest_by_id = {e["id"]: e for e in manifest}

    # Variants to evaluate: non-distractor, non-original entries
    variant_entries = [
        e for e in manifest
        if e.get("source_id") is not None
        and e.get("transform") not in ("original", "distractor")
        and e["id"] in index
    ]

    if not variant_entries:
        raise ValueError("No variant entries found in manifest.")

    print(f"Evaluating {len(variant_entries)} variant queries across "
          f"{len(METHODS)} methods…\n")

    per_query: list[dict] = []
    total_query_time = 0.0

    for entry in tqdm(variant_entries, unit="query", desc="Evaluating"):
        query_id = entry["id"]
        source_id = entry["source_id"]
        transform = entry["transform"]
        trim_start_gt = entry.get("trim_start_sec")

        query_fp = index[query_id]
        row: dict = {
            "query_id":   query_id,
            "source_id":  source_id,
            "transform":  transform,
        }

        t0 = time.time()

        for method in METHODS:
            # Score query against all other videos
            scores_list: list[tuple[str, float, dict]] = []

            for vid_id, vid_data in index.items():
                if vid_id == query_id:
                    continue  # leave-one-out

                score_method = method if method != "fusion_nt" else "fusion"
                score_matrix = compute_frame_scores(
                    query_fp, vid_data, method=score_method, weights=weights,
                )

                if method == "fusion":
                    align = temporal_align(
                        query_fp["t"], vid_data["t"], score_matrix,
                        bin_width=bin_width,
                    )
                    vid_score = align["video_score"]
                else:
                    vid_score = no_align_score(score_matrix)
                    align = {"offset_sec": 0.0, "matched_start": 0.0,
                             "matched_end": 0.0}

                scores_list.append((vid_id, vid_score, align))

            # Rank by score
            scores_list.sort(key=lambda x: x[1], reverse=True)

            # Find rank of any entry with the same source_id
            rank = 9999
            pred_offset = 0.0
            for r_idx, (vid_id, _, align_info) in enumerate(scores_list, 1):
                vid_meta = manifest_by_id.get(vid_id, {})
                if vid_meta.get("source_id") == source_id:
                    rank = r_idx
                    pred_offset = align_info.get("offset_sec", 0.0)
                    break

            row[f"{method}_rank"] = rank
            row[f"{method}_top1"] = (rank == 1)
            row[f"{method}_top5"] = (rank <= 5)
            row[f"{method}_rr"] = 1.0 / rank if rank < 9999 else 0.0

            # Localization error for trimmed queries
            if transform == "trimmed" and trim_start_gt is not None:
                row[f"{method}_loc_error"] = abs(pred_offset - (trim_start_gt if trim_start_gt else 0.0))
            else:
                row[f"{method}_loc_error"] = None

        elapsed = time.time() - t0
        total_query_time += elapsed
        per_query.append(row)

    # Build summary
    summary = build_summary(per_query, variant_entries)
    summary["mean_query_latency_sec"] = total_query_time / max(1, len(per_query))
    summary["total_queries"] = len(per_query)

    return per_query, summary


def build_summary(per_query: list[dict],
                  variant_entries: list[dict]) -> dict:
    """Build summary statistics from per-query results."""
    transforms = sorted(set(e["transform"] for e in variant_entries))
    summary: dict = {"transforms": transforms, "methods": METHODS}

    # Overall metrics per method
    overall: dict[str, dict] = {}
    for method in METHODS:
        n = len(per_query)
        top1 = sum(1 for r in per_query if r.get(f"{method}_top1"))
        top5 = sum(1 for r in per_query if r.get(f"{method}_top5"))
        mrr = sum(r.get(f"{method}_rr", 0) for r in per_query) / n if n else 0
        loc_errors = [r[f"{method}_loc_error"]
                      for r in per_query
                      if r[f"{method}_loc_error"] is not None]
        mean_loc = float(np.mean(loc_errors)) if loc_errors else None

        overall[method] = {
            "top1_acc": top1 / n if n else 0,
            "top5_acc": top5 / n if n else 0,
            "mrr":      mrr,
            "mean_loc_error": mean_loc,
        }
    summary["overall"] = overall

    # Per transform × method
    per_transform: dict[str, dict] = {}
    for t in transforms:
        t_rows = [r for r in per_query if r["transform"] == t]
        n = len(t_rows)
        per_transform[t] = {}
        for method in METHODS:
            top1 = sum(1 for r in t_rows if r.get(f"{method}_top1"))
            per_transform[t][method] = top1 / n * 100 if n else 0.0
    summary["per_transform"] = per_transform

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Display + Export
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(summary: dict) -> None:
    """Print formatted summary tables."""
    overall = summary["overall"]

    print()
    print("=" * 70)
    print("  VIDEO EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  {'Method':<25} {'Top-1':>8} {'Top-5':>8} "
          f"{'MRR':>8} {'Loc Err':>10}")
    print("-" * 70)
    for method in METHODS:
        m = overall[method]
        label = METHOD_LABELS[method]
        loc = f"{m['mean_loc_error']:.2f}s" if m["mean_loc_error"] is not None else "n/a"
        print(f"  {label:<25} {m['top1_acc']*100:>7.1f}% "
              f"{m['top5_acc']*100:>7.1f}% {m['mrr']:>8.3f} {loc:>10}")
    print("=" * 70)

    # Per-transform table
    transforms = summary["transforms"]
    per_transform = summary["per_transform"]

    print()
    print("  Per-Transform Top-1 Accuracy (%)")
    print("-" * 70)
    header = f"  {'Transform':<16}" + "".join(
        f"{METHOD_LABELS[m]:>12}" for m in METHODS
    )
    print(header)
    print("-" * 70)
    for t in transforms:
        row = f"  {t:<16}"
        for method in METHODS:
            val = per_transform[t].get(method, 0)
            row += f"{val:>11.1f}%"
        print(row)
    print("-" * 70)

    lat = summary.get("mean_query_latency_sec", 0)
    print(f"\n  Mean query latency: {lat:.2f} s")
    print(f"  Total queries: {summary.get('total_queries', 0)}")
    print()


def export_per_query_csv(per_query: list[dict], out_path: Path) -> None:
    """Export per-query results to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not per_query:
        return
    fieldnames = list(per_query[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write("# Content Forensics — Video Per-Query Evaluation\n")
        f.write("# NOTE: All corpus data is SIMULATED demo data.\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_query:
            clean = {}
            for k, v in row.items():
                if v is None:
                    clean[k] = "n/a"
                elif isinstance(v, bool):
                    clean[k] = str(v)
                elif isinstance(v, float):
                    clean[k] = f"{v:.4f}"
                else:
                    clean[k] = v
            writer.writerow(clean)
    print(f"Per-query CSV saved → {out_path}")


def export_summary_csv(summary: dict, out_path: Path) -> None:
    """Export method × transform summary table to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    transforms = summary["transforms"]
    per_transform = summary["per_transform"]
    overall = summary["overall"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write("# Content Forensics — Video Evaluation Summary\n")
        writer = csv.writer(f)
        # Header
        writer.writerow(["method", "top1_pct", "top5_pct", "mrr",
                          "mean_loc_error"] + transforms)
        for method in METHODS:
            ov = overall[method]
            row = [
                METHOD_LABELS[method],
                f"{ov['top1_acc']*100:.1f}",
                f"{ov['top5_acc']*100:.1f}",
                f"{ov['mrr']:.3f}",
                f"{ov['mean_loc_error']:.2f}" if ov["mean_loc_error"] is not None else "n/a",
            ]
            for t in transforms:
                row.append(f"{per_transform[t].get(method, 0):.1f}")
            writer.writerow(row)
    print(f"Summary CSV saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Evaluate video retrieval accuracy across five methods:
            pHash, FaceNet, CLIP, Fusion (no alignment), Fusion + Temporal.

            Reports Top-1, Top-5, MRR, and localization error for trimmed
            queries. Produces per-query and summary CSVs.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("video_corpus"),
        help="Video corpus directory (default: video_corpus/)",
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("cache/video_index.pkl"),
        help="Video index cache (default: cache/video_index.pkl)",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("video_corpus/manifest.json"),
        help="Manifest path (default: video_corpus/manifest.json)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("video_eval_results.csv"),
        help="Per-query CSV output (default: video_eval_results.csv)",
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("video_eval_summary.csv"),
        help="Summary CSV output (default: video_eval_summary.csv)",
    )
    parser.add_argument(
        "--weights", type=float, nargs=3,
        metavar=("W_PHASH", "W_FACE", "W_CLIP"),
        default=[DEFAULT_WEIGHTS["phash"], DEFAULT_WEIGHTS["face"],
                 DEFAULT_WEIGHTS["clip"]],
        help="Fusion weights (default: 0.2 0.4 0.4)",
    )
    parser.add_argument(
        "--bin-width", type=float, default=1.0,
        help="Offset bin width for temporal alignment (default: 1.0)",
    )
    args = parser.parse_args()

    weights = {
        "phash": args.weights[0],
        "face":  args.weights[1],
        "clip":  args.weights[2],
    }

    print("=" * 60)
    print("Content Forensics — Video Evaluation")
    print(f"Fusion weights: pHash={weights['phash']}, "
          f"face={weights['face']}, CLIP={weights['clip']}")
    print("=" * 60 + "\n")

    per_query, summary = evaluate(
        corpus_dir=args.corpus,
        cache_path=args.cache,
        manifest_path=args.manifest,
        weights=weights,
        bin_width=args.bin_width,
    )

    print_summary(summary)
    export_per_query_csv(per_query, out_path=args.out)
    export_summary_csv(summary, out_path=args.summary)
    print("\nDone.")


if __name__ == "__main__":
    main()

