"""
fingerprints.py — Shared fingerprinting logic for Content Forensics pipeline.

Three independent similarity signals:
  1. pHash         (imagehash)                  — perceptual hash, weak baseline
  2. Face embedding (facenet-pytorch)            — 512-d identity vector
  3. CLIP embedding (open_clip, ViT-B-32/openai) — 512-d visual-semantic vector

NOTE: No model training or fine-tuning. All models are used as pretrained
      off-the-shelf feature extractors only.

ETHICAL NOTE: This module processes images supplied by the user. It does NOT
              fetch, store, or transmit any real personal data. All corpus
              images are expected to be synthetic/demo data as documented in
              build_corpus.py. Never use real leaked or intimate images with
              this tool.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
import torch
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Device selection (CPU-first; uses CUDA if available)
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Fingerprint device: %s", DEVICE)


# ──────────────────────────────────────────────────────────────────────────────
# Lazy-loaded model singletons
# ──────────────────────────────────────────────────────────────────────────────
_mtcnn = None
_facenet = None
_clip_model = None
_clip_preprocess = None


def _get_face_models():
    """Return (MTCNN, InceptionResnetV1), loading once on first call."""
    global _mtcnn, _facenet
    if _mtcnn is None:
        from facenet_pytorch import MTCNN, InceptionResnetV1  # type: ignore

        _mtcnn = MTCNN(
            image_size=160,
            keep_all=False,  # return only the most prominent face
            device=DEVICE,
            post_process=True,  # returns tensor normalised to [-1, 1]
        )
        _facenet = (
            InceptionResnetV1(pretrained="vggface2")
            .eval()
            .to(DEVICE)
        )
    return _mtcnn, _facenet


def _get_clip_model():
    """Return (model, preprocess), loading once on first call."""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import open_clip  # type: ignore

        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _clip_model = _clip_model.eval().to(DEVICE)
    return _clip_model, _clip_preprocess


# ──────────────────────────────────────────────────────────────────────────────
# Fingerprint extractors
# ──────────────────────────────────────────────────────────────────────────────

def phash_fingerprint(img_path: str | Path) -> imagehash.ImageHash:
    """Compute perceptual hash (pHash) for an image."""
    img = Image.open(img_path).convert("RGB")
    return imagehash.phash(img)


def face_fingerprint(img_path: str | Path) -> Optional[np.ndarray]:
    """
    Detect the most prominent face, align it, and return a 512-d embedding.

    Returns None if no face is detected — caller must handle this gracefully.
    No exception is raised on detection failure.
    """
    mtcnn, facenet = _get_face_models()
    img = Image.open(img_path).convert("RGB")

    try:
        face_tensor = mtcnn(img)  # shape (3, 160, 160) or None
    except Exception as exc:
        logger.warning("Face detection error on %s: %s", img_path, exc)
        return None

    if face_tensor is None:
        logger.debug("No face detected in %s", img_path)
        return None

    face_batch = face_tensor.unsqueeze(0).to(DEVICE)  # (1, 3, 160, 160)
    with torch.no_grad():
        embedding = facenet(face_batch)  # (1, 512)
    # L2-normalise so cosine similarity == dot product
    emb = embedding[0].cpu().numpy()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def clip_fingerprint(img_path: str | Path) -> np.ndarray:
    """
    Compute a 512-d CLIP (ViT-B-32/openai) visual embedding.

    Single-image only — no video/temporal pooling. Always returns a vector.
    """
    model, preprocess = _get_clip_model()
    img = Image.open(img_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

    with torch.no_grad():
        features = model.encode_image(tensor)  # (1, 512)
    # L2-normalise
    emb = features[0].cpu().float().numpy()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


# ──────────────────────────────────────────────────────────────────────────────
# Similarity functions
# ──────────────────────────────────────────────────────────────────────────────

def phash_similarity(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> float:
    """1 - normalised Hamming distance. Range [0, 1]; 1 = identical."""
    return 1.0 - (h1 - h2) / len(h1.hash) ** 2


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity of two L2-normalised vectors. Range [-1, 1]."""
    # Both vectors are pre-normalised, so dot product suffices.
    return float(np.dot(v1, v2))


# ──────────────────────────────────────────────────────────────────────────────
# Fusion scorer
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {"phash": 0.2, "face": 0.4, "clip": 0.4}


def fusion_score(
    phash_sim: Optional[float],
    face_sim: Optional[float],
    clip_sim: Optional[float],
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> float:
    """
    Weighted-average fusion of up to three similarity scores.

    If a component is None (e.g. no face detected in query OR corpus image),
    it is excluded from both the numerator and the denominator — the remaining
    weights are renormalised so they sum to 1.0.

    Returns a float in roughly [0, 1].
    """
    scores = {
        "phash": phash_sim,
        "face": face_sim,
        "clip": clip_sim,
    }
    active_weight_sum = 0.0
    weighted_score_sum = 0.0

    for key, sim in scores.items():
        if sim is not None:
            w = weights.get(key, 0.0)
            weighted_score_sum += w * sim
            active_weight_sum += w

    if active_weight_sum == 0.0:
        return 0.0

    return weighted_score_sum / active_weight_sum


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: compute all fingerprints for one image
# ──────────────────────────────────────────────────────────────────────────────

def compute_all(img_path: str | Path) -> dict:
    """
    Compute pHash, face embedding, and CLIP embedding for a single image.

    Returns:
        {
          "phash":    imagehash.ImageHash,
          "face_emb": np.ndarray | None,
          "clip_emb": np.ndarray,
        }
    """
    img_path = Path(img_path)
    return {
        "phash":    phash_fingerprint(img_path),
        "face_emb": face_fingerprint(img_path),
        "clip_emb": clip_fingerprint(img_path),
    }
