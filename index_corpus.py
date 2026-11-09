"""
index_corpus.py — Compute and cache fingerprints for every image in the corpus.

USAGE
─────
  python index_corpus.py [--corpus corpus/] [--cache cache/fingerprint_index.pkl]
                         [--force]

Options:
  --corpus   Path to the corpus directory (default: corpus/)
  --cache    Path to save the fingerprint cache (default: cache/fingerprint_index.pkl)
  --force    Recompute fingerprints even for images already in the cache

The cache is a pickle file mapping:
  { filename: { "phash": ImageHash, "face_emb": ndarray|None, "clip_emb": ndarray } }

Run this once after build_corpus.py. Subsequent search.py / evaluate.py calls
load from cache and skip the expensive model inference step.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import textwrap
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tqdm import tqdm  # type: ignore

import fingerprints as fp

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache_path: Path, index: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(index, f)


def index_corpus(corpus_dir: Path, cache_path: Path, force: bool = False) -> dict:
    corpus_files = sorted(
        p for p in corpus_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not corpus_files:
        raise FileNotFoundError(f"No images found in corpus directory: {corpus_dir}")

    print(f"Found {len(corpus_files)} image(s) in '{corpus_dir}'.")
    index = {} if force else load_cache(cache_path)

    already_cached = sum(1 for p in corpus_files if p.name in index)
    to_process = [p for p in corpus_files if (force or p.name not in index)]

    if already_cached and not force:
        print(f"  {already_cached} image(s) already cached — skipping.")
    print(f"  Computing fingerprints for {len(to_process)} image(s)…\n")

    if not to_process:
        print("Nothing to do. Use --force to recompute all.")
        return index

    print("Loading models… (this may take a moment on first run)\n")
    # Trigger lazy loads so the progress bar is accurate
    _ = fp._get_face_models()
    _ = fp._get_clip_model()

    face_miss_count = 0

    for img_path in tqdm(to_process, unit="img", desc="Indexing"):
        try:
            result = fp.compute_all(img_path)
        except Exception as exc:
            print(f"\n  WARNING: Failed to process {img_path.name}: {exc}")
            continue

        index[img_path.name] = result

        if result["face_emb"] is None:
            face_miss_count += 1

    save_cache(cache_path, index)

    print(f"\n✓ Index saved → {cache_path}")
    print(f"  Total indexed: {len(index)} image(s)")
    if face_miss_count:
        print(
            f"  No face detected in {face_miss_count} image(s) "
            "(face score will be 'n/a' for those entries)."
        )
    return index


def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Compute and cache pHash / face / CLIP fingerprints for all corpus images.
            Run once after build_corpus.py; results are reused by search.py and evaluate.py.
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
        help="Cache file path (default: cache/fingerprint_index.pkl)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute all fingerprints, ignoring existing cache",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Content Forensics — Corpus Indexer")
    print("=" * 60)
    index_corpus(args.corpus, args.cache, force=args.force)
    print("\nDone. Run search.py or evaluate.py next.")


if __name__ == "__main__":
    main()
