"""
search_video.py — Query the video corpus for matches to a target video/image.

Supports video-to-video and image-to-video search. Uses temporal alignment
to find matching segments and estimate offsets.

USAGE
─────
  python search_video.py --query <video_or_image>
                         [--cache cache/video_index.pkl]
                         [--manifest video_corpus/manifest.json]
                         [--out video_results.csv]
                         [--top-k 10] [--fps 1.0]
                         [--method fusion]
                         [--bin-width 1.0]

NOTE: All corpus entries are SIMULATED demo data.
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

import imagehash
import numpy as np
from PIL import Image

import fingerprints as fp
from fingerprints import DEFAULT_WEIGHTS
import video_fingerprints as vfp

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SCORE_THRESHOLD_HIGH = 0.55
SCORE_THRESHOLD_LOW = 0.35
MIN_FRAME_SCORE = 0.20          # Minimum per-frame score for alignment inliers
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def load_cache(cache_path: Path) -> dict:
    """Load the video index cache."""
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache not found at {cache_path}. Run index_videos.py first."
        )
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Return {video_id: manifest_entry}."""
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)
    return {e["id"]: e for e in entries}


def _phash_to_int(h: imagehash.ImageHash) -> int:
    """Convert ImageHash to a 64-bit integer for vectorised XOR."""
    return int(str(h), 16)


def _phash_int_similarity(a: int, b: int) -> float:
    """1 - normalised Hamming distance between two 64-bit hashes."""
    xor = a ^ b
    hamming = bin(xor).count("1")
    return 1.0 - hamming / 64.0


# ──────────────────────────────────────────────────────────────────────────────
# Frame-level scoring (vectorised)
# ──────────────────────────────────────────────────────────────────────────────
def compute_frame_scores(
    query_fp: dict,
    corpus_fp: dict,
    method: str = "fusion",
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> np.ndarray:
    """
    Compute pairwise fused scores between query frames and corpus frames.

    Returns:
        score_matrix: np.ndarray [n_query, n_corpus]
    """
    nq = len(query_fp["t"])
    nc = len(corpus_fp["t"])

    if method == "phash":
        q_ints = np.array([_phash_to_int(h) for h in query_fp["phash"]])
        c_ints = np.array([_phash_to_int(h) for h in corpus_fp["phash"]])
        scores = np.zeros((nq, nc), dtype=np.float64)
        for i in range(nq):
            for j in range(nc):
                xor = int(q_ints[i]) ^ int(c_ints[j])
                hamming = bin(xor).count("1")
                scores[i, j] = 1.0 - hamming / 64.0
        return scores

    elif method == "face":
        scores = np.full((nq, nc), -1.0, dtype=np.float64)
        for i in range(nq):
            if query_fp["face"][i] is None:
                continue
            for j in range(nc):
                if corpus_fp["face"][j] is None:
                    continue
                sim = float(np.dot(query_fp["face"][i], corpus_fp["face"][j]))
                scores[i, j] = max(0.0, sim)
        return scores

    elif method == "clip":
        # Vectorised: matrix product of L2-normalised embeddings
        q_clip = query_fp["clip"]    # [nq, 512]
        c_clip = corpus_fp["clip"]   # [nc, 512]
        scores = np.clip(q_clip @ c_clip.T, 0.0, 1.0)
        return scores

    else:  # fusion or fusion_nt
        # Compute all three signal matrices
        # CLIP
        q_clip = query_fp["clip"]
        c_clip = corpus_fp["clip"]
        clip_scores = np.clip(q_clip @ c_clip.T, 0.0, 1.0)

        # pHash
        phash_scores = np.zeros((nq, nc), dtype=np.float64)
        q_ints = [_phash_to_int(h) for h in query_fp["phash"]]
        c_ints = [_phash_to_int(h) for h in corpus_fp["phash"]]
        for i in range(nq):
            for j in range(nc):
                xor = int(q_ints[i]) ^ int(c_ints[j])
                hamming = bin(xor).count("1")
                phash_scores[i, j] = 1.0 - hamming / 64.0

        # Face
        face_scores = np.full((nq, nc), np.nan, dtype=np.float64)
        for i in range(nq):
            if query_fp["face"][i] is None:
                continue
            for j in range(nc):
                if corpus_fp["face"][j] is None:
                    continue
                sim = float(np.dot(query_fp["face"][i], corpus_fp["face"][j]))
                face_scores[i, j] = max(0.0, sim)

        # Fuse per-element with adaptive renormalization
        w_p = weights.get("phash", 0.2)
        w_f = weights.get("face", 0.4)
        w_c = weights.get("clip", 0.4)

        has_face = ~np.isnan(face_scores)
        face_scores_safe = np.where(has_face, face_scores, 0.0)

        weight_sum = np.where(has_face, w_p + w_f + w_c, w_p + w_c)
        weighted_sum = (w_p * phash_scores
                        + np.where(has_face, w_f * face_scores_safe, 0.0)
                        + w_c * clip_scores)

        scores = np.where(weight_sum > 0, weighted_sum / weight_sum, 0.0)
        return scores


# ──────────────────────────────────────────────────────────────────────────────
# Temporal alignment
# ──────────────────────────────────────────────────────────────────────────────
def temporal_align(
    query_t: np.ndarray,
    corpus_t: np.ndarray,
    score_matrix: np.ndarray,
    min_score: float = MIN_FRAME_SCORE,
    bin_width: float = 1.0,
) -> dict:
    """
    Perform temporal alignment between query and corpus frames.

    For each query frame, find the best-matching corpus frame. Collect
    (query_t, corpus_t, score) triples above min_score. Bin offsets
    (corpus_t - query_t) and find the most populated bin (merged with
    neighbours).

    Returns:
        {
          "video_score":     float,
          "matched_start":   float,  # earliest corpus_t of inliers
          "matched_end":     float,  # latest corpus_t of inliers
          "offset_sec":      float,  # estimated offset
          "inlier_count":    int,
          "n_query_frames":  int,
        }
    """
    nq = len(query_t)
    # For each query frame, find best corpus frame
    best_corpus_idx = np.argmax(score_matrix, axis=1)
    best_scores = score_matrix[np.arange(nq), best_corpus_idx]

    # Filter by minimum score
    mask = best_scores >= min_score
    if not np.any(mask):
        return {
            "video_score": 0.0,
            "matched_start": 0.0,
            "matched_end": 0.0,
            "offset_sec": 0.0,
            "inlier_count": 0,
            "n_query_frames": nq,
        }

    valid_query_t = query_t[mask]
    valid_corpus_t = corpus_t[best_corpus_idx[mask]]
    valid_scores = best_scores[mask]

    # Compute offsets and bin them
    offsets = valid_corpus_t - valid_query_t

    if len(offsets) == 0:
        return {
            "video_score": 0.0,
            "matched_start": 0.0,
            "matched_end": 0.0,
            "offset_sec": 0.0,
            "inlier_count": 0,
            "n_query_frames": nq,
        }

    # Create bins
    min_off = float(np.min(offsets))
    max_off = float(np.max(offsets))
    if max_off - min_off < bin_width:
        # All offsets fit in one bin
        inlier_mask = np.ones(len(offsets), dtype=bool)
    else:
        n_bins = max(1, int((max_off - min_off) / bin_width) + 2)
        bin_edges = np.linspace(min_off - bin_width / 2,
                                max_off + bin_width / 2, n_bins + 1)
        bin_indices = np.digitize(offsets, bin_edges) - 1

        # Find most populated bin (merge with neighbours)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)

        # Merge each bin with its neighbour and find the best
        best_bin_score = -1
        best_bin_idx = 0
        for b in range(n_bins):
            count = bin_counts[b]
            if b > 0:
                count += bin_counts[b - 1]
            if b < n_bins - 1:
                count += bin_counts[b + 1]
            if count > best_bin_score:
                best_bin_score = count
                best_bin_idx = b

        # Inliers: frames in winning bin or its neighbours
        inlier_bins = {best_bin_idx}
        if best_bin_idx > 0:
            inlier_bins.add(best_bin_idx - 1)
        if best_bin_idx < n_bins - 1:
            inlier_bins.add(best_bin_idx + 1)
        inlier_mask = np.array([bin_indices[i] in inlier_bins
                                for i in range(len(offsets))])

    inlier_scores = valid_scores[inlier_mask]
    inlier_corpus_t = valid_corpus_t[inlier_mask]
    inlier_offsets = offsets[inlier_mask]

    n_inliers = len(inlier_scores)
    if n_inliers == 0:
        return {
            "video_score": 0.0,
            "matched_start": 0.0,
            "matched_end": 0.0,
            "offset_sec": 0.0,
            "inlier_count": 0,
            "n_query_frames": nq,
        }

    mean_score = float(np.mean(inlier_scores))
    coverage = n_inliers / nq
    video_score = mean_score * coverage

    return {
        "video_score":    round(video_score, 6),
        "matched_start":  round(float(np.min(inlier_corpus_t)), 3),
        "matched_end":    round(float(np.max(inlier_corpus_t)), 3),
        "offset_sec":     round(float(np.median(inlier_offsets)), 3),
        "inlier_count":   n_inliers,
        "n_query_frames": nq,
    }


def no_align_score(score_matrix: np.ndarray) -> float:
    """Baseline: mean of per-query-frame best scores (no temporal alignment)."""
    if score_matrix.size == 0:
        return 0.0
    best_per_query = np.max(score_matrix, axis=1)
    return float(np.mean(best_per_query))


# ──────────────────────────────────────────────────────────────────────────────
# Main search
# ──────────────────────────────────────────────────────────────────────────────
def search_video(
    query_path: Path,
    cache_path: Path,
    manifest_path: Path,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    top_k: int = 10,
    fps: float = 1.0,
    method: str = "fusion",
    bin_width: float = 1.0,
    corpus_dir: Path = Path("video_corpus"),
) -> list[dict]:
    """
    Search the video index for matches to a query video or image.

    Returns a list of result dicts sorted by video_score descending.
    """
    index = load_cache(cache_path)
    manifest = load_manifest(manifest_path)

    # Fingerprint the query
    is_img = _is_image(query_path)
    if is_img:
        print("Query is an IMAGE — performing image-to-video search.")
        pil_img = Image.open(query_path).convert("RGB")
        query_fp = {
            "t":     np.array([0.0]),
            "phash": [vfp.phash_from_pil(pil_img)],
            "clip":  vfp.clip_from_pil(pil_img).reshape(1, -1),
            "face":  [vfp.face_from_pil(pil_img)],
        }
    else:
        print(f"Query is a VIDEO — sampling at {fps} fps.")
        query_fp = vfp.fingerprint_video(query_path, fps=fps)
        print(f"  Sampled {len(query_fp['t'])} frames from query.")

    # Determine query file name for leave-one-out exclusion
    query_name = query_path.name
    query_stem = query_path.stem

    print(f"\nSearching {len(index)} indexed video(s)…")
    results = []
    t_start = time.time()

    use_alignment = (method != "fusion_nt") and not is_img

    for vid_id, vid_data in index.items():
        # Leave-one-out: skip if the query IS this corpus entry
        vid_meta = vid_data.get("meta", {})
        vid_path_name = vid_meta.get("path", "")
        if (vid_id == query_stem or
                Path(vid_path_name).name == query_name):
            continue

        score_method = method if method != "fusion_nt" else "fusion"
        score_matrix = compute_frame_scores(
            query_fp, vid_data, method=score_method, weights=weights,
        )

        if use_alignment:
            align = temporal_align(
                query_fp["t"], vid_data["t"], score_matrix,
                bin_width=bin_width,
            )
            video_score = align["video_score"]
        else:
            video_score = no_align_score(score_matrix)
            # For image queries or fusion_nt, get best frame info
            if score_matrix.size > 0:
                best_flat = int(np.argmax(score_matrix))
                best_q, best_c = divmod(best_flat, score_matrix.shape[1])
                align = {
                    "matched_start": round(float(vid_data["t"][best_c]), 3),
                    "matched_end":   round(float(vid_data["t"][best_c]), 3),
                    "offset_sec":    0.0,
                    "inlier_count":  1,
                    "n_query_frames": len(query_fp["t"]),
                }
            else:
                align = {
                    "matched_start": 0.0, "matched_end": 0.0,
                    "offset_sec": 0.0, "inlier_count": 0,
                    "n_query_frames": len(query_fp["t"]),
                }

        meta = manifest.get(vid_id, {})
        video_rel_path = meta.get("path", f"{vid_id}.mp4")
        corpus_file_path = str((corpus_dir / video_rel_path).resolve())

        # Confidence
        if video_score >= SCORE_THRESHOLD_HIGH:
            confidence = "HIGH"
            status = "FLAGGED_FOR_REMOVAL"
        elif video_score >= SCORE_THRESHOLD_LOW:
            confidence = "MEDIUM"
            status = "REVIEW"
        else:
            confidence = "LOW"
            status = ""

        results.append({
            "video_id":         vid_id,
            "corpus_path":      corpus_file_path,
            "platform":         meta.get("platform", "unknown"),
            "url":              meta.get("url", "n/a"),
            "video_score":      video_score,
            "confidence":       confidence,
            "transform_hint":   meta.get("transform", ""),
            "matched_start_sec": align["matched_start"],
            "matched_end_sec":  align["matched_end"],
            "offset_sec":       align["offset_sec"],
            "inlier_count":     align["inlier_count"],
            "status":           status,
            "source_id":        meta.get("source_id"),
        })

    results.sort(key=lambda r: r["video_score"], reverse=True)
    elapsed = time.time() - t_start
    print(f"  Search completed in {elapsed:.2f} s")
    return results[:top_k]


# ──────────────────────────────────────────────────────────────────────────────
# Display + Export
# ──────────────────────────────────────────────────────────────────────────────
def print_results(results: list[dict], query_name: str) -> None:
    """Print a readable console summary."""
    print()
    print("=" * 110)
    print(f"  VIDEO MATCH REPORT — Query: {query_name}")
    print(f"  [Platform / URL values are SIMULATED demo data]")
    print("=" * 110)
    header = (f"{'#':>4} {'Video ID':<28} {'Platform':<14} "
              f"{'Score':>8} {'Conf':>6} {'Start':>7} {'End':>7} "
              f"{'Offset':>8} {'Inliers':>8} {'Status':<20} {'Transform':<14}")
    print(header)
    print("-" * 110)

    for i, r in enumerate(results, start=1):
        vid_id = r["video_id"]
        if len(vid_id) > 28:
            vid_id = vid_id[:27] + "…"
        status = r["status"] if r["status"] else "—"
        print(
            f"{i:>4} {vid_id:<28} {r['platform']:<14} "
            f"{r['video_score']:>8.4f} {r['confidence']:>6} "
            f"{r['matched_start_sec']:>7.1f} {r['matched_end_sec']:>7.1f} "
            f"{r['offset_sec']:>8.2f} {r['inlier_count']:>8} "
            f"{status:<20} {r['transform_hint']:<14}"
        )
    print("-" * 110)

    if results:
        best = results[0]
        print(f"\n  Top match: {best['video_id']}  "
              f"(score={best['video_score']:.4f})")
        print(f"  Matched segment: {best['matched_start_sec']:.1f}s – "
              f"{best['matched_end_sec']:.1f}s  "
              f"(offset ≈ {best['offset_sec']:.2f}s)")
        print(f"  Simulated URL: {best['url']}")

    print()
    print("  Matched corpus video paths:")
    print("  " + "-" * 80)
    for i, r in enumerate(results, start=1):
        print(f"  [{i:>2}] {r['video_id']:<28} -> {r.get('corpus_path', '')}")
    print()


def export_csv(
    results: list[dict],
    out_path: Path,
    query_name: str,
    query_path: Path | str = "",
) -> None:
    """Export results to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_query_path = str(Path(query_path).resolve()) if query_path else query_name
    fieldnames = [
        "rank", "query_name", "query_path", "video_id", "corpus_path",
        "platform", "url", "video_score",
        "confidence", "transform_hint", "matched_start_sec",
        "matched_end_sec", "offset_sec", "inlier_count", "status",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        f.write("# Content Forensics — Video Search Results\n")
        f.write("# NOTE: platform and url values are SIMULATED demo data.\n")
        f.write(f"# Query: {query_name}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for i, r in enumerate(results, start=1):
            row = dict(r)
            row["rank"] = i
            row["query_name"] = query_name
            row["query_path"] = resolved_query_path
            row["video_score"] = f"{r['video_score']:.6f}"
            writer.writerow(row)
    print(f"CSV results saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Search the video corpus for matches to a query video or image.

            Supports video-to-video (with temporal alignment) and
            image-to-video (best frame match) search.

            NOTE: All corpus entries are SIMULATED demo data.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", type=Path, required=True,
        help="Path to the query video or image",
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("cache/video_index.pkl"),
        help="Video index cache (default: cache/video_index.pkl)",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("video_corpus/manifest.json"),
        help="Video corpus manifest (default: video_corpus/manifest.json)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("video_results.csv"),
        help="CSV output path (default: video_results.csv)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top results (default: 10)",
    )
    parser.add_argument(
        "--fps", type=float, default=1.0,
        help="Frame sampling rate for query video (default: 1.0)",
    )
    parser.add_argument(
        "--method", type=str, default="fusion",
        choices=["phash", "face", "clip", "fusion", "fusion_nt"],
        help="Scoring method (default: fusion). fusion_nt = fusion without "
             "temporal alignment.",
    )
    parser.add_argument(
        "--bin-width", type=float, default=1.0,
        help="Offset bin width in seconds for temporal alignment (default: 1.0)",
    )
    parser.add_argument(
        "--weights", type=float, nargs=3,
        metavar=("W_PHASH", "W_FACE", "W_CLIP"),
        default=[DEFAULT_WEIGHTS["phash"], DEFAULT_WEIGHTS["face"],
                 DEFAULT_WEIGHTS["clip"]],
        help="Fusion weights (default: 0.2 0.4 0.4)",
    )
    parser.add_argument(
        "--paths-only", action="store_true",
        help="Print only the matched video names and their file paths in the terminal",
    )
    args = parser.parse_args()

    weights = {
        "phash": args.weights[0],
        "face":  args.weights[1],
        "clip":  args.weights[2],
    }

    print("=" * 60)
    print("Content Forensics — Video Search")
    print("=" * 60)
    print(f"Query  : {args.query}")
    print(f"Method : {args.method}")
    print(f"Weights: pHash={weights['phash']}, face={weights['face']}, "
          f"CLIP={weights['clip']}")
    print()

    results = search_video(
        query_path=args.query,
        cache_path=args.cache,
        manifest_path=args.manifest,
        weights=weights,
        top_k=args.top_k,
        fps=args.fps,
        method=args.method,
        bin_width=args.bin_width,
    )

    if args.paths_only:
        print()
        print(f"Matched video files for query: {args.query.name}")
        print("=" * 80)
        for i, r in enumerate(results, start=1):
            print(f"[{i:>2}] {r['video_id']:<28} : {r.get('corpus_path', '')}")
        print("=" * 80)
    else:
        print_results(results, query_name=args.query.name)
    export_csv(results, out_path=args.out, query_name=args.query.name, query_path=args.query)


if __name__ == "__main__":
    main()

