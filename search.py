"""
search.py — Query the corpus for matches to a target image.

USAGE
─────
  python search.py --target <image_path>
                   [--cache cache/fingerprint_index.pkl]
                   [--manifest corpus/manifest.json]
                   [--out results.csv]
                   [--top 10]
                   [--weights 0.2 0.4 0.4]   # pHash, face, CLIP

OUTPUT
──────
Ranked table printed to console (all columns below), plus a CSV export:
  rank | filename | platform | url | pHash_score | face_score | clip_score
       | fusion_score | transform

NOTE: Platform and URL values are SIMULATED demo data.
      This tool does NOT perform live web searches or platform API calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import fingerprints as fp
from fingerprints import DEFAULT_WEIGHTS

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_cache(cache_path: Path) -> dict:
    import pickle
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found at {cache_path}. "
            "Run index_corpus.py first."
        )
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Return {filename: manifest_entry} for quick lookup."""
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)
    return {e["filename"]: e for e in entries}


def fmt_score(s: Optional[float]) -> str:
    return f"{s:.4f}" if s is not None else "n/a"


def _score_pair(
    query_fp: dict,
    corpus_fp: dict,
) -> tuple[Optional[float], Optional[float], float]:
    """
    Compute (phash_sim, face_sim, clip_sim) for one query–corpus pair.

    face_sim is None if either side has no face embedding.
    """
    phash_sim = fp.phash_similarity(query_fp["phash"], corpus_fp["phash"])

    q_face = query_fp["face_emb"]
    c_face = corpus_fp["face_emb"]
    face_sim: Optional[float] = None
    if q_face is not None and c_face is not None:
        face_sim = fp.cosine_similarity(q_face, c_face)
        # Clamp to [0, 1] for display; cosine can be negative for dissimilar
        face_sim = max(0.0, face_sim)

    clip_sim = fp.cosine_similarity(query_fp["clip_emb"], corpus_fp["clip_emb"])
    clip_sim = max(0.0, clip_sim)

    return phash_sim, face_sim, clip_sim


# ──────────────────────────────────────────────────────────────────────────────
# Core search
# ──────────────────────────────────────────────────────────────────────────────

def search(
    target_path: Path,
    cache_path: Path,
    manifest_path: Path,
    weights: dict[str, float],
    top_k: int,
    corpus_dir: Path = Path("corpus"),
) -> list[dict]:
    """
    Compute fingerprints for target_path and rank all corpus entries.

    Returns a list of result dicts sorted by fusion_score descending.
    Each result includes 'corpus_path' — the absolute path of the matched file.
    """
    print("Computing fingerprints for target image…")
    query_fp = fp.compute_all(target_path)
    if query_fp["face_emb"] is None:
        print("  ⚠  No face detected in the target image.")
        print("     Face embedding similarity will be 'n/a'.")

    index = load_cache(cache_path)
    manifest = load_manifest(manifest_path)

    print(f"\nSearching {len(index)} corpus entries…")
    results = []
    for filename, corpus_fp in index.items():
        phash_sim, face_sim, clip_sim = _score_pair(query_fp, corpus_fp)
        fuse = fp.fusion_score(phash_sim, face_sim, clip_sim, weights)

        meta = manifest.get(filename, {})
        results.append({
            "filename":    filename,
            "corpus_path": str((corpus_dir / filename).resolve()),
            "platform":    meta.get("platform", "unknown"),
            "url":         meta.get("url", "n/a"),
            "transform":   meta.get("transform", "unknown"),
            "phash_score": phash_sim,
            "face_score":  face_sim,
            "clip_score":  clip_sim,
            "fusion_score": fuse,
        })

    results.sort(key=lambda r: r["fusion_score"], reverse=True)
    return results[:top_k]


# ──────────────────────────────────────────────────────────────────────────────
# Display + Export
# ──────────────────────────────────────────────────────────────────────────────

_COL_WIDTHS = {
    "rank":         4,
    "filename":     32,
    "platform":     14,
    "pHash":        8,
    "face":         8,
    "CLIP":         8,
    "fusion":       8,
    "transform":    12,
}


def _header_line() -> str:
    h = (
        f"{'#':>{_COL_WIDTHS['rank']}} "
        f"{'Filename':<{_COL_WIDTHS['filename']}} "
        f"{'Platform':<{_COL_WIDTHS['platform']}} "
        f"{'pHash':>{_COL_WIDTHS['pHash']}} "
        f"{'Face':>{_COL_WIDTHS['face']}} "
        f"{'CLIP':>{_COL_WIDTHS['CLIP']}} "
        f"{'Fusion':>{_COL_WIDTHS['fusion']}} "
        f"{'Transform':<{_COL_WIDTHS['transform']}}"
    )
    return h


def print_results(results: list[dict], target_name: str) -> None:
    print()
    print("=" * 100)
    print(f"  MATCH REPORT — Query: {target_name}")
    print(f"  [Platform / URL values are SIMULATED demo data]")
    print("=" * 100)
    print(_header_line())
    print("-" * 100)
    for i, r in enumerate(results, start=1):
        fname = r["filename"]
        if len(fname) > _COL_WIDTHS["filename"]:
            fname = fname[: _COL_WIDTHS["filename"] - 1] + "…"
        fusion_fmt = f"{r['fusion_score']:.4f}"
        print(
            f"{i:>{_COL_WIDTHS['rank']}} "
            f"{fname:<{_COL_WIDTHS['filename']}} "
            f"{r['platform']:<{_COL_WIDTHS['platform']}} "
            f"{fmt_score(r['phash_score']):>{_COL_WIDTHS['pHash']}} "
            f"{fmt_score(r['face_score']):>{_COL_WIDTHS['face']}} "
            f"{fmt_score(r['clip_score']):>{_COL_WIDTHS['CLIP']}} "
            f"{fusion_fmt:>{_COL_WIDTHS['fusion']}} "
            f"{r['transform']:<{_COL_WIDTHS['transform']}}"
        )
    print("-" * 100)
    if results:
        best = results[0]
        print(f"\n  Top match: {best['filename']}  (fusion={best['fusion_score']:.4f})")
        print(f"  Simulated URL: {best['url']}")

    # ── Corpus file paths for all matches ─────────────────────────────────────
    print()
    print("  Matched corpus file paths:")
    print("  " + "-" * 70)
    for i, r in enumerate(results, start=1):
        print(f"  [{i:>2}] {r['corpus_path']}")
    print()


def export_csv(
    results: list[dict],
    out_path: Path,
    target_name: str,
    target_path: Path | str = "",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_target_path = str(Path(target_path).resolve()) if target_path else target_name
    fieldnames = [
        "rank",
        "query_image",
        "target_image_path",
        "filename",
        "corpus_path",
        "platform",
        "url",
        "phash_score",
        "face_score",
        "clip_score",
        "fusion_score",
        "transform",
    ]
    file_exists = out_path.exists() and out_path.stat().st_size > 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        if not file_exists:
            f.write(f"# Content Forensics Evidence Export\n")
            f.write(f"# NOTE: platform and url values are SIMULATED demo data.\n")
        f.write(f"# Query image: {target_name}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for i, r in enumerate(results, start=1):
            writer.writerow({
                "rank":              i,
                "query_image":       target_name,
                "target_image_path": resolved_target_path,
                "filename":          r["filename"],
                "corpus_path":       r.get("corpus_path", ""),
                "platform":          r["platform"],
                "url":               r["url"],
                "phash_score":       fmt_score(r["phash_score"]),
                "face_score":        fmt_score(r["face_score"]),
                "clip_score":        fmt_score(r["clip_score"]),
                "fusion_score":      f"{r['fusion_score']:.4f}",
                "transform":         r["transform"],
            })
    print(f"CSV evidence list saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Search the corpus for images similar to a target image.
            Uses three fingerprinting methods (pHash, face embedding, CLIP)
            fused into a single ranked similarity score.

            NOTE: All corpus entries are SIMULATED demo data.
                  No live web search or platform API call is made.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Path to the target (query) image",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("cache/fingerprint_index.pkl"),
        help="Fingerprint cache (default: cache/fingerprint_index.pkl)",
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
        default=Path("results.csv"),
        help="CSV output path (default: results.csv)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top results to show (default: 10)",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        metavar=("W_PHASH", "W_FACE", "W_CLIP"),
        default=[DEFAULT_WEIGHTS["phash"], DEFAULT_WEIGHTS["face"], DEFAULT_WEIGHTS["clip"]],
        help="Fusion weights for pHash, face, CLIP (default: 0.2 0.4 0.4)",
    )
    parser.add_argument(
        "--strip",
        nargs="?",
        const="auto",
        default=None,
        help="Generate a visual match strip image in visuals/ (e.g. --strip or --strip custom_name.png)",
    )
    args = parser.parse_args()

    weights = {
        "phash": args.weights[0],
        "face":  args.weights[1],
        "clip":  args.weights[2],
    }

    print("=" * 60)
    print("Content Forensics — Search")
    print("=" * 60)
    print(f"Target image : {args.target}")
    print(f"Fusion weights: pHash={weights['phash']}, face={weights['face']}, CLIP={weights['clip']}")
    print()

    results = search(
        target_path=args.target,
        cache_path=args.cache,
        manifest_path=args.manifest,
        weights=weights,
        top_k=args.top,
    )

    print_results(results, target_name=args.target.name)
    export_csv(results, out_path=args.out, target_name=args.target.name, target_path=args.target)

    if args.strip:
        from visualize_results import plot_match_strip
        safe_name = args.target.name.replace(".", "_")
        out_name = f"match_strip_{safe_name}.png" if args.strip == "auto" else args.strip
        plot_match_strip(args.target.name, args.target, results, out_name, top_n=5)


if __name__ == "__main__":
    main()
