# Content Forensics: Fingerprinting & Cross-Platform Retrieval

> **7th Sem CS/Engineering Academic Prototype**  
> Empirical comparison of three image fingerprinting methods for finding
> transformed/manipulated copies of a reference image across a simulated corpus.

---

## ⚠ Ethical Statement (read first)

This tool is built for **research and education only**.

- **No real intimate or leaked images** are used anywhere in this project — not
  as source data, not as examples, not as placeholders.  
- **No live platform scanning** is performed. All "platform" labels and URLs in
  the corpus are clearly marked `SIMULATED` and do not represent any real site.  
- **No web scraping** is done. The corpus is built from images that you supply
  (team members' own photos or clearly synthetic/public-domain images).  
- **Distractor images** are generated algorithmically (random colour noise) and
  contain no real persons.  
- This tool does **not** classify real vs. fake (deepfake detection), and it
  does **not** connect to the internet during any stage of its operation.

---

## What It Does

Given a folder of source images, the pipeline:

1. **Builds a corpus** — generates 4 transformed variants per source image
   (center-crop, horizontal flip, watermark overlay, JPEG-20 recompression) and
   adds synthetic noise distractors. Writes a `manifest.json` mapping every
   corpus file to a simulated platform label and URL.

2. **Indexes fingerprints** — computes and caches three independent similarity
   signals for every corpus image:
   | Signal | Library | Description |
   |---|---|---|
   | **pHash** | `imagehash` | Perceptual hash — fast but brittle baseline |
   | **Face embedding** | `facenet-pytorch` | 512-d identity vector (MTCNN + InceptionResnetV1/vggface2) |
   | **CLIP embedding** | `open_clip` (ViT-B-32/openai) | 512-d visual-semantic vector |

3. **Searches** — given a query image, ranks all corpus entries by a weighted
   fusion score (`pHash=0.2, face=0.4, CLIP=0.4`). Missing face detections are
   excluded from the fusion and the remaining weights renormalized.

4. **Evaluates** — for each corpus variant, checks whether the true original
   ranks #1 under each method. Reports Top-1 accuracy and saves a bar chart.

---

## Setup

### Requirements
- Python 3.9+ recommended
- No GPU required (CPU inference works; ~2–5 min for a 25-image corpus)

### Install dependencies

```bash
pip install -r requirements.txt
```

> On Windows, if `facenet-pytorch` fails, try:  
> `pip install facenet-pytorch --no-deps && pip install torch torchvision`

---

## Usage

### Step 1 — Place your source images

Put **3–10 images** into the `sample_images/` folder.  
Use your own team photos, or download AI-generated synthetic faces
(e.g. from [thispersondoesnotexist.com](https://thispersondoesnotexist.com))
for a fully synthetic demo.  
**Do not use real leaked or intimate images.**

### Step 2 — Build the corpus

```bash
python build_corpus.py --src sample_images/ --out corpus/ --distractors 5
```

**Output:** `corpus/` directory containing:
- `original_001.jpg`, `original_002.jpg`, … (your source images)
- `original_001_cropped.jpg`, `original_001_flipped.jpg`, … (4 variants each)
- `distractor_001.jpg`, … (synthetic noise images, no real persons)
- `manifest.json` (filename → simulated platform + URL)

### Step 3 — Index fingerprints (run once; cached)

```bash
python index_corpus.py --corpus corpus/ --cache cache/fingerprint_index.pkl
```

**Output:** `cache/fingerprint_index.pkl` — re-used by all subsequent commands.  
Use `--force` to recompute from scratch.

### Step 4 — Search for a target image

```bash
python search.py --target my_image.jpg --out results.csv --top 10
```

Optional: override fusion weights  
```bash
python search.py --target my_image.jpg --weights 0.2 0.4 0.4
```

**Output:**
- Ranked table printed to console
- `results.csv` — evidence list (rank, filename, platform, url, scores per method)

#### Column meanings

| Column | Meaning |
|---|---|
| `rank` | Position in the ranked match list (1 = best match) |
| `filename` | Corpus image filename |
| `platform` | **SIMULATED** platform label |
| `url` | **SIMULATED** demo URL (not a real link) |
| `phash_score` | Perceptual hash similarity [0–1] |
| `face_score` | Cosine similarity of face embeddings [0–1], or `n/a` if no face detected |
| `clip_score` | Cosine similarity of CLIP embeddings [0–1] |
| `fusion_score` | Weighted average of active components [0–1] |
| `transform` | Transform type inferred from filename |

### Step 5 — Evaluate (research deliverable)

```bash
python evaluate.py --corpus corpus/ --cache cache/fingerprint_index.pkl \
                   --out eval_results.csv --chart accuracy_chart.png
```

**Output:**
- Summary table printed to console:
  ```
  Method                          Top-1 Acc
  pHash (baseline)                  25.0%
  Face Embedding (FaceNet)          75.0%
  CLIP (ViT-B-32)                   87.5%
  Fusion (weighted avg)             90.0%
  ```
- `eval_results.csv` — per-query ranks and hit/miss flags
- `accuracy_chart.png` — bar chart suitable for slides/report

---

## Project File Map

| File | Purpose |
|---|---|
| `fingerprints.py` | Shared fingerprinting + similarity + fusion logic |
| `build_corpus.py` | Generate transforms + distractors + manifest |
| `index_corpus.py` | Compute + cache fingerprints for entire corpus |
| `search.py` | Query search: ranked results + CSV evidence export |
| `evaluate.py` | Top-1 accuracy evaluation + bar chart |
| `requirements.txt` | Python dependencies |

---

## Limitations / Non-Goals

- ❌ No real-vs-fake / deepfake classification
- ❌ No live web scraping or platform API calls
- ❌ No model training or fine-tuning (models are used as-is)
- ✅ Video and image support
- ✅ Designed for offline, local corpus simulation only

---

## 🎬 Video Pipeline (NEW)

### Purpose

Extends the image pipeline to find transformed/manipulated copies of a
**reference video** across a simulated multi-platform corpus. The same three
fingerprinting signals (pHash, FaceNet, CLIP) are applied **per frame**, and
a **temporal alignment** step identifies matching segments and estimates time
offsets between query and corpus videos.

### New Files

| File | Purpose |
|---|---|
| `build_video_corpus.py` | Generate source clips (synthetic or real CC0), 9 transforms each, distractors, manifest |
| `video_fingerprints.py` | Frame sampling + per-frame fingerprinting (reuses fingerprints.py models) |
| `index_videos.py` | Compute + cache per-frame fingerprints for entire video corpus |
| `search_video.py` | Video/image query → ranked matches with temporal alignment |
| `evaluate_video.py` | Top-1/Top-5/MRR evaluation + localization error + per-transform breakdown |

### Additional Requirements

- `opencv-python>=4.8.0` (required — frame sampling, synthetic clip generation)
- `scenedetect>=0.6.0` (optional — scene-based keyframe detection)
- `ffmpeg` on system PATH (optional — higher quality video transforms; falls
  back to OpenCV re-encoding if missing, with a clear warning)

### Run Order

#### Step 1 — Build the video corpus

```bash
python build_video_corpus.py --src sample_videos/ --out video_corpus/ --seed 42
```

If `sample_videos/` is empty, 10 synthetic clips with distinct visual patterns
are auto-generated (bouncing shapes, gradients, scrolling text, etc.). These
are clearly labelled as SYNTHETIC and do not depict any real person.

**Transforms generated per source clip:**

| Transform | Description |
|---|---|
| `h264_crf35` | Heavy H.264 re-encode (CRF 35) |
| `cropped` | Center crop 80% area, rescaled to original size |
| `flipped` | Horizontal flip |
| `watermarked` | Semi-transparent text overlay (~30% alpha) |
| `360p` | Downscale to 360p |
| `fps15` | Frame rate reduced to 15 fps |
| `trimmed` | 5–8 second segment at a random offset |
| `speed125` | 1.25× speed |
| `brightness` | Brightness/colour shift |

**Output:** `video_corpus/` directory + `manifest.json`.

#### Step 2 — Index video fingerprints

```bash
python index_videos.py --corpus video_corpus/ --cache cache/video_index.pkl
```

Use `--rebuild` to recompute all. Use `--fps` to change sampling rate.

#### Step 3 — Search for a target video

```bash
python search_video.py --query video_corpus/source_001_trimmed.mp4 --top-k 10
```

Supports image queries too (image-to-video search):
```bash
python search_video.py --query my_screenshot.jpg --top-k 10
```

Override method: `--method phash|face|clip|fusion|fusion_nt`

#### Step 4 — Evaluate

```bash
python evaluate_video.py --corpus video_corpus/ --cache cache/video_index.pkl
```

#### Step 5 — Visualize (video outputs)

```bash
python visualize_results.py --video
```

### Video Search Output Columns

| Column | Meaning |
|---|---|
| `rank` | Position in ranked match list (1 = best) |
| `video_id` | Corpus video identifier |
| `platform` | **SIMULATED** platform label |
| `url` | **SIMULATED** demo URL |
| `video_score` | Fused score [0–1] with temporal coverage penalty |
| `confidence` | HIGH / MEDIUM / LOW |
| `transform_hint` | Transform type from manifest (if known) |
| `matched_start_sec` | Start of matched segment in corpus video |
| `matched_end_sec` | End of matched segment in corpus video |
| `offset_sec` | Estimated time offset (corpus_t − query_t) |
| `inlier_count` | Number of temporally consistent frame matches |
| `status` | FLAGGED_FOR_REMOVAL / REVIEW / blank |

### Temporal Alignment — How It Works

1. For each query frame, compute the fused similarity score against every
   frame of each candidate video.
2. For each query frame, take the best-matching corpus frame — producing a
   triple `(query_t, corpus_t, score)`.
3. Keep triples with score above a minimum threshold.
4. Compute offset = `corpus_t − query_t` for each triple.
5. Bin offsets with a configurable bin width (default 1.0 s), merge each bin
   with its neighbours, and select the most populated bin.
6. **Video score** = mean(inlier scores) × (n_inliers / n_query_frames).
7. **Matched segment** = [min(corpus_t), max(corpus_t)] of the inliers.

This detects matching video segments even when the copy has been trimmed,
sped up, or has a different start time.

### Extended Ethics Statement

All ethical guarantees from the image pipeline apply equally to the video
pipeline:

- **No real intimate or leaked videos** are used — source clips are either
  user-supplied (CC0/public-domain) or fully synthetic geometric patterns.
- **No live platform scanning** or web scraping.
- **All platform labels and URLs are SIMULATED** and fictional.
- **Only fingerprints (hashes + embeddings) are stored**, not media copies.
- This tool does **not** classify content as "leaked" or "non-consensual" —
  it only finds visually similar copies of a supplied reference.
#   F o r g e d - a n d - L e a k e d - C o n t e n t - R e t r i e v a l - S y s t e m  
 #   F o r g e d - a n d - L e a k e d - C o n t e n t - R e t r i e v a l - S y s t e m  
 #   F o r g e d - a n d - L e a k e d - C o n t e n t - R e t r i e v a l - S y s t e m  
 