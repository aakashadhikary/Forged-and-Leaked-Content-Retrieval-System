"""
video_fingerprints.py — Frame sampling and per-frame fingerprinting for videos.

Reuses the model singletons from fingerprints.py (pHash, FaceNet, CLIP)
to compute per-frame fingerprints without duplicating model loading.

ETHICAL NOTE: This module processes video files supplied by the user.
              It does NOT fetch, store, or transmit any real personal data.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import imagehash
import numpy as np
import torch
from PIL import Image

# Import model singletons and device from the existing image pipeline
from fingerprints import DEVICE, _get_clip_model, _get_face_models

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# PIL-based fingerprint extractors (reuse models, no disk I/O)
# ──────────────────────────────────────────────────────────────────────────────

def phash_from_pil(img: Image.Image) -> imagehash.ImageHash:
    """Compute perceptual hash (pHash) from a PIL Image directly."""
    return imagehash.phash(img.convert("RGB"))


def face_from_pil(img: Image.Image) -> Optional[np.ndarray]:
    """Detect face and return 512-d L2-normalised embedding, or None."""
    mtcnn, facenet = _get_face_models()
    rgb = img.convert("RGB")
    try:
        face_tensor = mtcnn(rgb)
    except Exception as exc:
        logger.warning("Face detection error: %s", exc)
        return None
    if face_tensor is None:
        return None
    face_batch = face_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = facenet(face_batch)
    emb = embedding[0].cpu().numpy()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def clip_from_pil(img: Image.Image) -> np.ndarray:
    """Compute 512-d L2-normalised CLIP embedding from a PIL Image."""
    model, preprocess = _get_clip_model()
    tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model.encode_image(tensor)
    emb = features[0].cpu().float().numpy()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def clip_batch_from_pil(images: list[Image.Image]) -> np.ndarray:
    """Batch CLIP embedding for multiple PIL images. Returns [N, 512]."""
    if not images:
        return np.empty((0, 512), dtype=np.float32)
    model, preprocess = _get_clip_model()
    tensors = torch.stack([preprocess(img.convert("RGB")) for img in images])
    tensors = tensors.to(DEVICE)
    with torch.no_grad():
        features = model.encode_image(tensors)
    embs = features.cpu().float().numpy()
    # L2-normalise each row
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embs = embs / norms
    return embs


# ──────────────────────────────────────────────────────────────────────────────
# Frame sampling
# ──────────────────────────────────────────────────────────────────────────────

def sample_frames(
    path: str | Path,
    fps: float = 1.0,
    max_frames: int | None = None,
) -> list[tuple[float, Image.Image]]:
    """
    Sample frames from a video at the given FPS rate.

    Args:
        path:       Path to the video file.
        fps:        Sampling rate in frames per second (e.g. 1.0 = 1 frame/s).
        max_frames: Maximum number of frames to return (None = no limit).

    Returns:
        List of (timestamp_sec, PIL.Image RGB) tuples.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            logger.warning("Cannot open video: %s", path)
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps or video_fps <= 0 or np.isnan(video_fps):
            video_fps = 25.0
            logger.warning("Invalid FPS metadata for %s, defaulting to 25.",
                           path.name)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.warning("Zero-frame video: %s", path)
            return []

        duration = total_frames / video_fps
        # Compute desired sample timestamps
        interval = 1.0 / fps
        timestamps = np.arange(0, duration, interval)

        if max_frames is not None:
            timestamps = timestamps[:max_frames]

        results: list[tuple[float, Image.Image]] = []
        for ts in timestamps:
            frame_idx = int(ts * video_fps)
            if frame_idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            results.append((round(ts, 3), pil_img))

        return results

    except Exception as exc:
        logger.warning("Error sampling frames from %s: %s", path, exc)
        return []
    finally:
        cap.release()


def sample_frames_dedup(
    path: str | Path,
    fps: float = 1.0,
    max_frames: int | None = None,
    hamming_threshold: int = 3,
) -> list[tuple[float, Image.Image]]:
    """
    Like sample_frames but skip near-duplicate frames.

    A frame is skipped if its pHash Hamming distance to the previously
    kept frame is below `hamming_threshold`.
    """
    all_frames = sample_frames(path, fps, max_frames=None)
    if not all_frames:
        return []

    kept: list[tuple[float, Image.Image]] = []
    prev_hash: Optional[imagehash.ImageHash] = None

    for ts, img in all_frames:
        h = phash_from_pil(img)
        if prev_hash is not None:
            dist = prev_hash - h  # Hamming distance
            if dist < hamming_threshold:
                continue
        kept.append((ts, img))
        prev_hash = h
        if max_frames is not None and len(kept) >= max_frames:
            break

    return kept


# ──────────────────────────────────────────────────────────────────────────────
# Video fingerprinting
# ──────────────────────────────────────────────────────────────────────────────

def fingerprint_frames(
    frames: list[tuple[float, Image.Image]],
    face_every_n: int = 1,
    batch_size: int = 16,
) -> dict:
    """
    Compute fingerprints for a list of pre-sampled frames.

    Args:
        frames:       List of (timestamp, PIL.Image) tuples.
        face_every_n: Run face detection every N frames (1 = every frame).
        batch_size:   Batch size for CLIP inference.

    Returns:
        {
          "t":     np.ndarray of timestamps [N],
          "phash": list of imagehash.ImageHash [N],
          "clip":  np.ndarray [N, 512] L2-normalized,
          "face":  list of (np.ndarray | None) [N],
        }
    """
    if not frames:
        raise ValueError("No frames to fingerprint.")

    timestamps = np.array([ts for ts, _ in frames], dtype=np.float64)
    images = [img for _, img in frames]
    n = len(images)

    # pHash — fast, per-frame
    phashes = [phash_from_pil(img) for img in images]

    # CLIP — batch processing
    clip_embs = []
    for i in range(0, n, batch_size):
        batch = images[i:i + batch_size]
        embs = clip_batch_from_pil(batch)
        clip_embs.append(embs)
    clip_all = np.vstack(clip_embs) if clip_embs else np.empty((0, 512))

    # Face — only every face_every_n frames (MTCNN is slow)
    face_embs: list[Optional[np.ndarray]] = [None] * n
    for i in range(0, n, max(1, face_every_n)):
        face_embs[i] = face_from_pil(images[i])

    return {
        "t":     timestamps,
        "phash": phashes,
        "clip":  clip_all,
        "face":  face_embs,
    }


def fingerprint_video(
    path: str | Path,
    fps: float = 1.0,
    face_every_n: int = 1,
    batch_size: int = 16,
    dedup: bool = False,
    dedup_threshold: int = 3,
) -> dict:
    """
    Sample and fingerprint a video file.

    Args:
        path:             Path to the video.
        fps:              Sampling rate (frames per second).
        face_every_n:     Run face detection every N frames.
        batch_size:       Batch size for CLIP inference.
        dedup:            If True, skip near-duplicate frames.
        dedup_threshold:  Hamming distance threshold for dedup.

    Returns:
        Dict with keys "t", "phash", "clip", "face" (see fingerprint_frames).
    """
    if dedup:
        frames = sample_frames_dedup(path, fps,
                                     hamming_threshold=dedup_threshold)
    else:
        frames = sample_frames(path, fps)

    if not frames:
        raise ValueError(f"No frames sampled from {path}")

    return fingerprint_frames(frames, face_every_n, batch_size)

