"""
build_video_corpus.py — Video corpus builder for Content Forensics prototype.

ETHICAL STATEMENT
─────────────────
• This script processes videos supplied by the USER from sample_videos/.
  If no videos are present, it generates fully SYNTHETIC clips using OpenCV.
• Never use real leaked, intimate, or non-consented videos of any person.
• All generated corpus entries are labelled SIMULATED in the manifest.
• Simulated platform names and URLs are fictional and do NOT represent any
  real platform, person, or content.

USAGE
─────
  python build_video_corpus.py [--src sample_videos/] [--out video_corpus/]
                               [--seed 42] [--num-sources 10]
                               [--num-distractors 4]

For each source clip the following transformed variants are generated:
  _h264_crf35   — heavy H.264 re-encode (CRF 35)
  _cropped      — center crop 80 % area, rescaled to original size
  _flipped      — horizontal flip
  _watermarked  — semi-transparent repeated text overlay (~30 % alpha)
  _360p         — downscale to 360 p
  _fps15        — frame-rate reduced to 15 fps
  _trimmed      — keep a 5-8 s segment at a seeded random offset
  _speed125     — 1.25 × speed
  _brightness   — brightness / colour shift
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

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
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

TRANSFORM_NAMES = [
    "h264_crf35", "cropped", "flipped", "watermarked",
    "360p", "fps15", "trimmed", "speed125", "brightness",
]


# ──────────────────────────────────────────────────────────────────────────────
# ffmpeg detection
# ──────────────────────────────────────────────────────────────────────────────
def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


# ──────────────────────────────────────────────────────────────────────────────
# Video helpers
# ──────────────────────────────────────────────────────────────────────────────
def _video_info(path: Path) -> dict:
    """Return fps, width, height, frame_count, duration for a video file."""
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = fc / fps if fps > 0 else 0.0
    finally:
        cap.release()
    return {"fps": fps, "width": w, "height": h, "frame_count": fc, "duration": dur}


def _read_all_frames(path: Path, max_dim: int = 720) -> tuple[list[np.ndarray], float, int, int]:
    """Read all frames from a video. Returns (frames, fps, w, h). Caps max dimension to max_dim to bound memory."""
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_w, target_h = w, h
        if max(w, h) > max_dim and max(w, h) > 0:
            scale = max_dim / max(w, h)
            target_w = int(w * scale)
            target_h = int(h * scale)
            if target_w % 2 != 0:
                target_w += 1
            if target_h % 2 != 0:
                target_h += 1

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if (target_w, target_h) != (w, h):
                frame = cv2.resize(frame, (target_w, target_h))
            frames.append(frame)
    finally:
        cap.release()
    return frames, fps, target_w, target_h


def _write_video(frames: list[np.ndarray], out_path: Path,
                 fps: float, w: int, h: int) -> None:
    """Write frames to an mp4 file using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    try:
        for f in frames:
            if f.shape[1] != w or f.shape[0] != h:
                f = cv2.resize(f, (w, h))
            writer.write(f)
    finally:
        writer.release()


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic clip generators
# ──────────────────────────────────────────────────────────────────────────────
def _synth_bouncing_circle(n_frames: int, w: int, h: int,
                           rng: np.random.Generator) -> list[np.ndarray]:
    """Bouncing red circle on black background."""
    frames = []
    cx, cy = w // 4, h // 4
    dx, dy = 4, 3
    r = 40
    for _ in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(img, (cx, cy), r, (0, 0, 255), -1)
        cx += dx; cy += dy
        if cx - r < 0 or cx + r >= w: dx = -dx
        if cy - r < 0 or cy + r >= h: dy = -dy
        cx = max(r, min(w - r - 1, cx))
        cy = max(r, min(h - r - 1, cy))
        frames.append(img)
    return frames


def _synth_rainbow_scroll(n_frames: int, w: int, h: int,
                          rng: np.random.Generator) -> list[np.ndarray]:
    """Scrolling horizontal rainbow gradient."""
    frames = []
    for t in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(w):
            hue = int(((x + t * 3) % w) / w * 180)
            img[:, x] = (hue, 255, 200)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        frames.append(img)
    return frames


def _synth_rotating_text(n_frames: int, w: int, h: int,
                         rng: np.random.Generator) -> list[np.ndarray]:
    """Rotating text on dark blue background."""
    frames = []
    for t in range(n_frames):
        img = np.full((h, w, 3), (80, 30, 10), dtype=np.uint8)
        angle = t * 2 % 360
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        overlay = np.zeros_like(img)
        cv2.putText(overlay, "SYNTHETIC CLIP 3", (w // 8, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        rotated = cv2.warpAffine(overlay, M, (w, h))
        img = cv2.add(img, rotated)
        frames.append(img)
    return frames


def _synth_expanding_rectangles(n_frames: int, w: int, h: int,
                                rng: np.random.Generator) -> list[np.ndarray]:
    """Expanding/contracting green rectangles."""
    frames = []
    for t in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        scale = abs(math.sin(t * 0.05))
        for i in range(1, 6):
            s = int(scale * i * min(w, h) / 12)
            cv2.rectangle(img, (w // 2 - s, h // 2 - s),
                         (w // 2 + s, h // 2 + s), (0, 200, 50), 3)
        frames.append(img)
    return frames


def _synth_checkerboard(n_frames: int, w: int, h: int,
                        rng: np.random.Generator) -> list[np.ndarray]:
    """Diagonal moving yellow checkerboard pattern."""
    frames = []
    tile = 40
    for t in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        off = t * 2
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                if ((x + off) // tile + (y + off) // tile) % 2 == 0:
                    cv2.rectangle(img, (x, y), (x + tile, y + tile),
                                 (0, 220, 240), -1)
        frames.append(img)
    return frames


def _synth_concentric_circles(n_frames: int, w: int, h: int,
                              rng: np.random.Generator) -> list[np.ndarray]:
    """Pulsing concentric circles (cyan)."""
    frames = []
    for t in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        phase = t * 0.1
        for r in range(20, max(w, h), 30):
            thick = max(1, int(3 * abs(math.sin(phase + r * 0.02))))
            cv2.circle(img, (w // 2, h // 2), r, (200, 200, 0), thick)
        frames.append(img)
    return frames


def _synth_snowflakes(n_frames: int, w: int, h: int,
                      rng: np.random.Generator) -> list[np.ndarray]:
    """Falling white dots on dark gray."""
    n_flakes = 80
    xs = rng.integers(0, w, n_flakes)
    ys = rng.integers(0, h, n_flakes)
    speeds = rng.integers(2, 6, n_flakes)
    frames = []
    for _ in range(n_frames):
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        for i in range(n_flakes):
            cv2.circle(img, (int(xs[i]), int(ys[i])), 3, (255, 255, 255), -1)
            ys[i] = (ys[i] + speeds[i]) % h
        frames.append(img)
    return frames


def _synth_striped_bars(n_frames: int, w: int, h: int,
                        rng: np.random.Generator) -> list[np.ndarray]:
    """Horizontal scrolling striped bars (multi-color)."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    bar_h = h // len(colors)
    frames = []
    for t in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for i, c in enumerate(colors):
            y0 = i * bar_h
            y1 = y0 + bar_h
            shift = (t * 5 * (i + 1)) % w
            img[y0:y1, :shift] = c
            img[y0:y1, shift:] = tuple(max(0, v - 100) for v in c)
        frames.append(img)
    return frames


def _synth_spiral(n_frames: int, w: int, h: int,
                  rng: np.random.Generator) -> list[np.ndarray]:
    """Rotating spiral pattern (magenta)."""
    frames = []
    for t in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for r in range(5, max(w, h) // 2, 3):
            angle = r * 0.15 + t * 0.08
            px = int(w / 2 + r * math.cos(angle))
            py = int(h / 2 + r * math.sin(angle))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(img, (px, py), 2, (200, 0, 200), -1)
        frames.append(img)
    return frames


def _synth_flickering_grid(n_frames: int, w: int, h: int,
                           rng: np.random.Generator) -> list[np.ndarray]:
    """Grid of flickering colored squares."""
    tile = 60
    cols = w // tile + 1
    rows = h // tile + 1
    frames = []
    for t in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for gy in range(rows):
            for gx in range(cols):
                color = tuple(int(v) for v in rng.integers(50, 256, 3))
                x0, y0 = gx * tile, gy * tile
                cv2.rectangle(img, (x0, y0), (x0 + tile, y0 + tile), color, -1)
        frames.append(img)
    return frames


_SYNTH_GENERATORS = [
    ("bouncing_circle", _synth_bouncing_circle),
    ("rainbow_scroll", _synth_rainbow_scroll),
    ("rotating_text", _synth_rotating_text),
    ("expanding_rects", _synth_expanding_rectangles),
    ("checkerboard", _synth_checkerboard),
    ("concentric", _synth_concentric_circles),
    ("snowflakes", _synth_snowflakes),
    ("striped_bars", _synth_striped_bars),
    ("spiral", _synth_spiral),
    ("flickering_grid", _synth_flickering_grid),
]


def generate_synthetic_clips(out_dir: Path, num_sources: int,
                             seed: int) -> list[Path]:
    """Generate synthetic video clips if no real clips are available."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    fps = 24.0
    w, h = 640, 480
    paths: list[Path] = []

    print("\n⚠  No source clips found — generating SYNTHETIC video clips.")
    print("   These clips contain only geometric shapes / patterns.")
    print("   They do NOT depict any real person.\n")

    for idx in range(min(num_sources, len(_SYNTH_GENERATORS))):
        name, gen_fn = _SYNTH_GENERATORS[idx]
        duration = random.uniform(10.0, 30.0)
        n_frames = int(duration * fps)
        clip_path = out_dir / f"synthetic_{idx + 1:03d}_{name}.mp4"

        logger.info("Generating synthetic clip %d/%d: %s (%.1f s, %d frames)",
                    idx + 1, num_sources, name, duration, n_frames)

        frames = gen_fn(n_frames, w, h, rng)
        _write_video(frames, clip_path, fps, w, h)
        paths.append(clip_path)
        print(f"  [{idx + 1}/{num_sources}] Saved → {clip_path.name}  "
              f"({duration:.1f} s, {n_frames} frames)")

    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Transform implementations
# ──────────────────────────────────────────────────────────────────────────────
def _ffmpeg_run(args: list[str]) -> bool:
    """Run ffmpeg with the given args. Returns True on success."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-y"] + args,
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg failed: %s", result.stderr[:300])
            return False
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("ffmpeg error: %s", exc)
        return False


def _transform_h264_crf35(src: Path, dst: Path, use_ffmpeg: bool,
                          **kw) -> bool:
    """Heavy H.264 re-encode (CRF 35)."""
    if use_ffmpeg:
        return _ffmpeg_run(["-i", str(src), "-c:v", "libx264", "-crf", "35",
                           "-preset", "fast", str(dst)])
    # OpenCV fallback: re-encode with low quality
    frames, fps, w, h = _read_all_frames(src)
    if not frames:
        return False
    # Simulate quality loss by downscaling and upscaling
    lossy = []
    for f in frames:
        small = cv2.resize(f, (w // 3, h // 3))
        back = cv2.resize(small, (w, h))
        lossy.append(back)
    _write_video(lossy, dst, fps, w, h)
    return True


def _transform_cropped(src: Path, dst: Path, use_ffmpeg: bool,
                       **kw) -> bool:
    """Center crop 80% area, rescaled to original size."""
    frames, fps, w, h = _read_all_frames(src)
    if not frames:
        return False
    cw, ch = int(w * 0.894), int(h * 0.894)  # sqrt(0.80) ≈ 0.894
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    out = []
    for f in frames:
        crop = f[y0:y0 + ch, x0:x0 + cw]
        resized = cv2.resize(crop, (w, h))
        out.append(resized)
    _write_video(out, dst, fps, w, h)
    return True


def _transform_flipped(src: Path, dst: Path, use_ffmpeg: bool,
                       **kw) -> bool:
    """Horizontal flip."""
    if use_ffmpeg:
        return _ffmpeg_run(["-i", str(src), "-vf", "hflip",
                           "-c:v", "libx264", "-crf", "18", str(dst)])
    frames, fps, w, h = _read_all_frames(src)
    if not frames:
        return False
    out = [cv2.flip(f, 1) for f in frames]
    _write_video(out, dst, fps, w, h)
    return True


def _transform_watermarked(src: Path, dst: Path, use_ffmpeg: bool,
                           **kw) -> bool:
    """Semi-transparent repeated text overlay at ~30% alpha."""
    frames, fps, w, h = _read_all_frames(src)
    if not frames:
        return False
    # Create watermark overlay once
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    text = "DEMO WATERMARK"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, w / 800)
    for y in range(0, h, int(h / 4)):
        for x in range(0, w, int(w / 3)):
            cv2.putText(overlay, text, (x, y + 30), font, scale,
                       (255, 255, 255), 2)
    alpha = 0.30
    out = [cv2.addWeighted(f, 1.0, overlay, alpha, 0) for f in frames]
    _write_video(out, dst, fps, w, h)
    return True


def _transform_360p(src: Path, dst: Path, use_ffmpeg: bool,
                    **kw) -> bool:
    """Downscale to 360p height."""
    if use_ffmpeg:
        return _ffmpeg_run(["-i", str(src), "-vf",
                           "scale=-2:360", "-c:v", "libx264",
                           "-crf", "22", str(dst)])
    frames, fps, w, h = _read_all_frames(src)
    if not frames:
        return False
    new_h = 360
    new_w = int(w * new_h / h)
    if new_w % 2 != 0:
        new_w += 1
    out = [cv2.resize(f, (new_w, new_h)) for f in frames]
    _write_video(out, dst, fps, new_w, new_h)
    return True


def _transform_fps15(src: Path, dst: Path, use_ffmpeg: bool,
                     **kw) -> bool:
    """Frame rate reduced to 15 fps."""
    if use_ffmpeg:
        return _ffmpeg_run(["-i", str(src), "-r", "15",
                           "-c:v", "libx264", "-crf", "22", str(dst)])
    frames, fps, w, h = _read_all_frames(src)
    if not frames or fps <= 0:
        return False
    ratio = 15.0 / fps
    if ratio >= 1.0:
        _write_video(frames, dst, 15.0, w, h)
        return True
    out = []
    for i in range(int(len(frames) * ratio)):
        src_idx = min(int(i / ratio), len(frames) - 1)
        out.append(frames[src_idx])
    _write_video(out, dst, 15.0, w, h)
    return True


def _transform_trimmed(src: Path, dst: Path, use_ffmpeg: bool,
                       trim_start: float = 0.0, trim_dur: float = 6.0,
                       **kw) -> bool:
    """Keep a segment starting at trim_start for trim_dur seconds."""
    if use_ffmpeg:
        return _ffmpeg_run(["-i", str(src), "-ss", f"{trim_start:.2f}",
                           "-t", f"{trim_dur:.2f}", "-c:v", "libx264",
                           "-crf", "22", str(dst)])
    frames, fps, w, h = _read_all_frames(src)
    if not frames or fps <= 0:
        return False
    start_frame = int(trim_start * fps)
    end_frame = min(start_frame + int(trim_dur * fps), len(frames))
    if start_frame >= len(frames):
        start_frame = 0
        end_frame = min(int(trim_dur * fps), len(frames))
    out = frames[start_frame:end_frame]
    if not out:
        return False
    _write_video(out, dst, fps, w, h)
    return True


def _transform_speed125(src: Path, dst: Path, use_ffmpeg: bool,
                        **kw) -> bool:
    """1.25× speed (drop every 5th frame)."""
    if use_ffmpeg:
        return _ffmpeg_run(["-i", str(src), "-filter:v",
                           "setpts=0.8*PTS", "-c:v", "libx264",
                           "-crf", "22", str(dst)])
    frames, fps, w, h = _read_all_frames(src)
    if not frames:
        return False
    # Drop every 5th frame to approximate 1.25× speed
    out = [f for i, f in enumerate(frames) if i % 5 != 4]
    _write_video(out, dst, fps, w, h)
    return True


def _transform_brightness(src: Path, dst: Path, use_ffmpeg: bool,
                          **kw) -> bool:
    """Brightness/colour shift (+40 brightness, slight hue shift)."""
    if use_ffmpeg:
        return _ffmpeg_run(["-i", str(src), "-vf",
                           "eq=brightness=0.15:saturation=1.3",
                           "-c:v", "libx264", "-crf", "22", str(dst)])
    frames, fps, w, h = _read_all_frames(src)
    if not frames:
        return False
    out = []
    for f in frames:
        bright = cv2.convertScaleAbs(f, alpha=1.1, beta=40)
        hsv = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + 10) % 180
        out.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    _write_video(out, dst, fps, w, h)
    return True


_TRANSFORM_FUNCS = {
    "h264_crf35": _transform_h264_crf35,
    "cropped":    _transform_cropped,
    "flipped":    _transform_flipped,
    "watermarked": _transform_watermarked,
    "360p":       _transform_360p,
    "fps15":      _transform_fps15,
    "trimmed":    _transform_trimmed,
    "speed125":   _transform_speed125,
    "brightness": _transform_brightness,
}


# ──────────────────────────────────────────────────────────────────────────────
# Distractor generation
# ──────────────────────────────────────────────────────────────────────────────
def _generate_distractor_clip(idx: int, out_dir: Path, seed: int) -> Path:
    """Generate a synthetic distractor clip unrelated to source clips."""
    rng = np.random.default_rng(seed + idx * 99991)
    fps = 24.0
    w, h = 640, 480
    duration = rng.uniform(8.0, 15.0)
    n_frames = int(duration * fps)
    path = out_dir / f"distractor_{idx:03d}.mp4"

    frames = []
    base_color = tuple(int(v) for v in rng.integers(0, 200, 3))
    for t in range(n_frames):
        img = np.full((h, w, 3), base_color, dtype=np.uint8)
        # Random noise pattern unique per distractor
        noise = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
        img = cv2.add(img, noise)
        # Add a moving shape unique to this distractor
        cx = int((t * 3 + idx * 50) % w)
        cy = int(h / 2 + 100 * math.sin(t * 0.05 + idx))
        cv2.rectangle(img, (cx - 30, cy - 30), (cx + 30, cy + 30),
                     (int(rng.integers(100, 256)),
                      int(rng.integers(100, 256)),
                      int(rng.integers(100, 256))), -1)
        frames.append(img)
    _write_video(frames, path, fps, w, h)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Main build logic
# ──────────────────────────────────────────────────────────────────────────────
def build_video_corpus(
    src_dir: Path,
    out_dir: Path,
    seed: int = 42,
    num_sources: int = 10,
    num_distractors: int = 4,
) -> list[dict]:
    """Build the video corpus and return the manifest entries."""
    random.seed(seed)
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_ffmpeg = _has_ffmpeg()

    if use_ffmpeg:
        print("Using ffmpeg for video transforms.")
    else:
        print("⚠  ffmpeg not found on PATH — using OpenCV fallback.")
        print("   Some transforms may have lower quality.\n")

    # Discover or generate source clips
    src_dir.mkdir(parents=True, exist_ok=True)
    source_files = sorted(
        p for p in src_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not source_files:
        source_files = generate_synthetic_clips(src_dir, num_sources, seed)

    source_files = source_files[:num_sources]
    print(f"\nFound {len(source_files)} source clip(s) in '{src_dir}'.")

    manifest: list[dict] = []

    for idx, src_path in enumerate(source_files, start=1):
        source_id = f"source_{idx:03d}"
        info = _video_info(src_path)
        platform = random.choice(SIMULATED_PLATFORMS)

        # Copy original to corpus
        orig_name = f"{source_id}.mp4"
        orig_dest = out_dir / orig_name
        shutil.copy2(src_path, orig_dest)

        manifest.append({
            "id":             source_id,
            "path":           orig_name,
            "source_id":      source_id,
            "transform":      "original",
            "trim_start_sec": None,
            "duration":       round(info["duration"], 2),
            "fps":            round(info["fps"], 2),
            "platform":       platform,
            "url":            f"SIMULATED://{platform}/{source_id}",
            "note":           SIMULATED_NOTE,
        })

        print(f"\n  [{idx}/{len(source_files)}] {src_path.name} → {orig_name} "
              f"({info['duration']:.1f} s)")

        # Generate transforms
        for t_name in tqdm(TRANSFORM_NAMES, desc=f"    Transforms",
                           leave=False):
            variant_id = f"{source_id}_{t_name}"
            variant_name = f"{variant_id}.mp4"
            variant_dest = out_dir / variant_name
            variant_platform = random.choice(SIMULATED_PLATFORMS)

            extra_kw: dict = {}
            trim_start: Optional[float] = None

            if t_name == "trimmed":
                max_start = max(0.0, info["duration"] - 8.0)
                trim_start = round(rng.uniform(0.0, max(0.1, max_start)), 2)
                trim_dur = round(rng.uniform(5.0, 8.0), 2)
                extra_kw = {"trim_start": trim_start, "trim_dur": trim_dur}

            transform_fn = _TRANSFORM_FUNCS[t_name]
            success = transform_fn(orig_dest, variant_dest, use_ffmpeg,
                                   **extra_kw)

            if not success:
                logger.warning("Transform %s failed for %s, skipping.",
                               t_name, source_id)
                continue

            # Get variant info
            try:
                v_info = _video_info(variant_dest)
            except Exception:
                v_info = {"duration": 0, "fps": 0}

            manifest.append({
                "id":             variant_id,
                "path":           variant_name,
                "source_id":      source_id,
                "transform":      t_name,
                "trim_start_sec": trim_start,
                "duration":       round(v_info["duration"], 2),
                "fps":            round(v_info["fps"], 2),
                "platform":       variant_platform,
                "url":            f"SIMULATED://{variant_platform}/{variant_id}",
                "note":           SIMULATED_NOTE,
            })

        print(f"    Generated {len(TRANSFORM_NAMES)} variants.")

    # Distractors
    print(f"\nGenerating {num_distractors} distractor clip(s)…")
    for d_idx in range(1, num_distractors + 1):
        d_path = _generate_distractor_clip(d_idx, out_dir, seed)
        d_info = _video_info(d_path)
        d_platform = random.choice(SIMULATED_PLATFORMS)
        d_id = f"distractor_{d_idx:03d}"

        manifest.append({
            "id":             d_id,
            "path":           d_path.name,
            "source_id":      None,
            "transform":      "distractor",
            "trim_start_sec": None,
            "duration":       round(d_info["duration"], 2),
            "fps":            round(d_info["fps"], 2),
            "platform":       d_platform,
            "url":            f"SIMULATED://{d_platform}/{d_id}",
            "note":           SIMULATED_NOTE,
        })
        print(f"  Saved distractor → {d_path.name}")

    # Write manifest
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
            Build a synthetic video forensics test corpus from source clips.

            Generates 9 transformed variants per clip (h264_crf35, cropped,
            flipped, watermarked, 360p, fps15, trimmed, speed125, brightness)
            plus N synthetic distractor clips, all labelled SIMULATED in the
            manifest.

            If no source clips exist in --src, generates 10 synthetic clips
            using OpenCV (no network required).
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src", type=Path, default=Path("sample_videos"),
        help="Folder containing source video clips (default: sample_videos/)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("video_corpus"),
        help="Output folder for the video corpus (default: video_corpus/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--num-sources", type=int, default=10,
        help="Number of source clips to use/generate (default: 10)",
    )
    parser.add_argument(
        "--num-distractors", type=int, default=4,
        help="Number of distractor clips to generate (default: 4)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Content Forensics — Video Corpus Builder")
    print("All generated data is SIMULATED / synthetic demo only.")
    print("=" * 60)

    build_video_corpus(
        src_dir=args.src,
        out_dir=args.out,
        seed=args.seed,
        num_sources=args.num_sources,
        num_distractors=args.num_distractors,
    )
    print("\nDone. You can now run index_videos.py to compute fingerprints.")


if __name__ == "__main__":
    main()

