"""
index_videos.py — Compute and cache per-frame fingerprints for every video
in the corpus.

USAGE
─────
  python index_videos.py [--corpus video_corpus/]
                         [--cache cache/video_index.pkl]
                         [--fps 1.0] [--rebuild]

The cache is a pickle file mapping:
  { video_id: { "t": ndarray, "phash": list, "clip": ndarray, "face": list,
                "meta": dict } }

Run this once after build_video_corpus.py. Subsequent search_video.py /
evaluate_video.py calls load from cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import textwrap
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tqdm import tqdm

import video_fingerprints as vfp

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_cache(cache_path: Path) -> dict:
    """Load existing video index cache from disk."""
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache_path: Path, index: dict) -> None:
    """Save video index cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(index, f)


def index_videos(
    corpus_dir: Path,
    cache_path: Path,
    fps: float = 1.0,
    rebuild: bool = False,
) -> dict:
    """
    Fingerprint every video in corpus_dir listed in manifest.json.

    Returns the full index dict.
    """
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run build_video_corpus.py first."
        )
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Found {len(manifest)} entries in manifest.")
    index = {} if rebuild else load_cache(cache_path)

    already_cached = sum(1 for e in manifest if e["id"] in index)
    to_process = [e for e in manifest if rebuild or e["id"] not in index]

    if already_cached and not rebuild:
        print(f"  {already_cached} video(s) already cached — skipping.")
    print(f"  Computing fingerprints for {len(to_process)} video(s)…\n")

    if not to_process:
        print("Nothing to do. Use --rebuild to recompute all.")
        return index

    # Trigger lazy model loads
    print("Loading models… (this may take a moment on first run)\n")
    from fingerprints import _get_clip_model, _get_face_models
    _get_face_models()
    _get_clip_model()

    total_start = time.time()
    skipped = 0

    for entry in tqdm(to_process, unit="vid", desc="Indexing"):
        vid_id = entry["id"]
        vid_path = corpus_dir / entry["path"]

        if not vid_path.exists():
            logger.warning("Video not found: %s — skipping.", vid_path)
            skipped += 1
            continue

        t0 = time.time()
        try:
            fp = vfp.fingerprint_video(vid_path, fps=fps)
        except Exception as exc:
            logger.warning("Failed to fingerprint %s: %s — skipping.",
                           vid_id, exc)
            skipped += 1
            continue

        elapsed = time.time() - t0
        index[vid_id] = {
            "t":     fp["t"],
            "phash": fp["phash"],
            "clip":  fp["clip"],
            "face":  fp["face"],
            "meta":  entry,
        }
        tqdm.write(f"  {vid_id}: {len(fp['t'])} frames, {elapsed:.1f} s")

    save_cache(cache_path, index)

    total_elapsed = time.time() - total_start
    print(f"\n✓ Index saved → {cache_path}")
    print(f"  Total indexed: {len(index)} video(s)")
    print(f"  Indexing time: {total_elapsed:.1f} s")
    if skipped:
        print(f"  Skipped: {skipped} video(s) (unreadable or missing)")
    return index


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Compute and cache per-frame fingerprints (pHash, face, CLIP)
            for all videos in the corpus. Run once after build_video_corpus.py;
            results are reused by search_video.py and evaluate_video.py.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus", type=Path, default=Path("video_corpus"),
        help="Video corpus directory (default: video_corpus/)",
    )
    parser.add_argument(
        "--cache", type=Path, default=Path("cache/video_index.pkl"),
        help="Cache file path (default: cache/video_index.pkl)",
    )
    parser.add_argument(
        "--fps", type=float, default=1.0,
        help="Frame sampling rate in fps (default: 1.0)",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Recompute all fingerprints, ignoring existing cache",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Content Forensics — Video Indexer")
    print("=" * 60)
    index_videos(args.corpus, args.cache, fps=args.fps, rebuild=args.rebuild)
    print("\nDone. Run search_video.py or evaluate_video.py next.")


if __name__ == "__main__":
    main()

