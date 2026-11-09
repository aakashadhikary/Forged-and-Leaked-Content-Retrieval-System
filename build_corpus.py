"""
build_corpus.py — Corpus builder for Content Forensics prototype.

ETHICAL STATEMENT
─────────────────
• This script processes images supplied by the USER from the sample_images/
  folder. The pipeline is designed for SYNTHETIC / DEMO data only.
• Never use real leaked, intimate, or non-consented images of any person.
• All generated corpus entries are labelled SIMULATED in the manifest.
• Simulated platform names and URLs are fictional and do NOT represent any
  real platform, person, or content.

USAGE
─────
  python build_corpus.py --src sample_images/ --out corpus/ [--distractors 5]

For each image found in --src, four transformed variants are generated:
  _cropped      — center crop 80% + resize back to original dimensions
  _flipped      — horizontal mirror flip
  _watermarked  — semi-transparent "DEMO WATERMARK" text overlay
  _jpeg20       — JPEG recompression at quality=20

A manifest (corpus/manifest.json) is written mapping every file to a
simulated platform label and a fake demo URL. All entries are tagged
  "note": "SIMULATED — synthetic demo data only, not a real platform"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SIMULATED_PLATFORMS = [
    "DemoSite-A",
    "DemoForum-B",
    "DemoBoard-C",
    "DemoGallery-D",
    "DemoNet-E",
]
SIMULATED_NOTE = "SIMULATED — synthetic demo data only, not a real platform"
TRANSFORM_NAMES = ["cropped", "flipped", "watermarked", "jpeg20"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ──────────────────────────────────────────────────────────────────────────────
# Transform helpers
# ──────────────────────────────────────────────────────────────────────────────

def _center_crop_resize(img: Image.Image, crop_fraction: float = 0.80) -> Image.Image:
    """Crop the central `crop_fraction` of the image, then resize back."""
    w, h = img.size
    new_w = int(w * crop_fraction)
    new_h = int(h * crop_fraction)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.LANCZOS)


def _horizontal_flip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def _add_watermark(img: Image.Image, text: str = "DEMO WATERMARK") -> Image.Image:
    """Overlay semi-transparent diagonal text across the image."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to use a system font; fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", size=max(20, img.size[0] // 12))
    except OSError:
        font = ImageFont.load_default()

    # Tile text diagonally across the image
    text_w = img.size[0] // 2
    text_h = img.size[1] // 4
    for y in range(-img.size[1], img.size[1] * 2, text_h):
        for x in range(-img.size[0], img.size[0] * 2, text_w):
            draw.text((x, y), text, fill=(255, 255, 255, 80), font=font)

    watermarked = Image.alpha_composite(img.convert("RGBA"), overlay)
    return watermarked.convert("RGB")


def _jpeg_recompress(img: Image.Image, quality: int = 20) -> Image.Image:
    """Simulate JPEG recompression artefacts at the given quality level."""
    import io

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()  # .copy() detaches from the BytesIO buffer


# ──────────────────────────────────────────────────────────────────────────────
# Distractor generation (fully offline, no network required)
# ──────────────────────────────────────────────────────────────────────────────

def _generate_distractor(idx: int, size: tuple[int, int] = (512, 512)) -> Image.Image:
    """
    Create a synthetic distractor image using random coloured noise
    combined with simple geometric shapes.  These images do NOT resemble
    any real person and are intended purely as unrelated 'filler' corpus
    entries to simulate content from other topics on a platform.
    """
    rng = np.random.default_rng(seed=idx * 31337)  # reproducible per index

    # Base: random coloured noise
    noise = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(noise, mode="RGB")

    # Overlay a few semi-transparent coloured rectangles
    draw = ImageDraw.Draw(img, "RGBA")
    for _ in range(6):
        x0 = int(rng.integers(0, size[0]))
        y0 = int(rng.integers(0, size[1]))
        x1 = int(rng.integers(x0, size[0]))
        y1 = int(rng.integers(y0, size[1]))
        colour = tuple(rng.integers(0, 256, 4).tolist())
        draw.rectangle([x0, y0, x1, y1], fill=colour)

    return img.convert("RGB")


# ──────────────────────────────────────────────────────────────────────────────
# Main build logic
# ──────────────────────────────────────────────────────────────────────────────

def build_corpus(src_dir: Path, out_dir: Path, num_distractors: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(
        p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not source_files:
        raise FileNotFoundError(
            f"No images found in {src_dir}. "
            "Place .jpg / .png source images there first."
        )

    print(f"Found {len(source_files)} source image(s) in '{src_dir}'.")
    manifest: list[dict] = []

    for idx, src_path in enumerate(source_files, start=1):
        original_id = f"original_{idx:03d}"
        platform = random.choice(SIMULATED_PLATFORMS)

        # ── Save original ──────────────────────────────────────────────────
        orig_name = f"{original_id}.jpg"
        orig_dest = out_dir / orig_name
        img = Image.open(src_path).convert("RGB")
        img.save(orig_dest, format="JPEG", quality=95)
        print(f"  [{idx}/{len(source_files)}] Saved original → {orig_name}")

        manifest.append({
            "filename":    orig_name,
            "original_id": original_id,
            "transform":   "original",
            "platform":    platform,
            "url":         f"http://demo-corpus.local/{platform.lower()}/{orig_name}",
            "note":        SIMULATED_NOTE,
        })

        # ── Transforms ────────────────────────────────────────────────────
        transforms = {
            "cropped":     _center_crop_resize(img),
            "flipped":     _horizontal_flip(img),
            "watermarked": _add_watermark(img),
            "jpeg20":      _jpeg_recompress(img, quality=20),
        }
        for t_name, t_img in transforms.items():
            variant_platform = random.choice(SIMULATED_PLATFORMS)
            variant_name = f"{original_id}_{t_name}.jpg"
            variant_dest = out_dir / variant_name

            # jpeg20 is already JPEG-encoded; others save at quality 95
            if t_name == "jpeg20":
                t_img.save(variant_dest, format="JPEG", quality=95)
            else:
                t_img.save(variant_dest, format="JPEG", quality=95)

            manifest.append({
                "filename":    variant_name,
                "original_id": original_id,
                "transform":   t_name,
                "platform":    variant_platform,
                "url":         f"http://demo-corpus.local/{variant_platform.lower()}/{variant_name}",
                "note":        SIMULATED_NOTE,
            })
        print(f"           Generated 4 variants.")

    # ── Distractors ───────────────────────────────────────────────────────
    print(f"\nGenerating {num_distractors} synthetic distractor image(s)…")
    for d_idx in range(1, num_distractors + 1):
        d_platform = random.choice(SIMULATED_PLATFORMS)
        d_name = f"distractor_{d_idx:03d}.jpg"
        d_img = _generate_distractor(idx=d_idx)
        d_img.save(out_dir / d_name, format="JPEG", quality=90)
        manifest.append({
            "filename":    d_name,
            "original_id": None,
            "transform":   "distractor",
            "platform":    d_platform,
            "url":         f"http://demo-corpus.local/{d_platform.lower()}/{d_name}",
            "note":        SIMULATED_NOTE,
        })
    print(f"  Saved {num_distractors} distractor(s).")

    # ── Write manifest ────────────────────────────────────────────────────
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written → {manifest_path}")
    print(f"Total corpus entries: {len(manifest)}")
    return manifest


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Build a synthetic forensics test corpus from source images.

            Generates 4 transformed variants per image (cropped, flipped,
            watermarked, jpeg20) plus N synthetic distractor images, all
            labelled SIMULATED in the manifest.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("sample_images"),
        help="Folder containing your source images (default: sample_images/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("corpus"),
        help="Output folder for the corpus (default: corpus/)",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=5,
        help="Number of synthetic distractor images to generate (default: 5)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Content Forensics — Corpus Builder")
    print("All generated data is SIMULATED / synthetic demo only.")
    print("=" * 60)
    build_corpus(args.src, args.out, args.distractors)
    print("\nDone. You can now run index_corpus.py to compute fingerprints.")


if __name__ == "__main__":
    main()
