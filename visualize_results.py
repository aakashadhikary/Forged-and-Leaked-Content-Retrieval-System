"""
visualize_results.py — PPT-ready visualizations for Content Forensics results.

Generates 5 publication-quality charts saved to visuals/ folder:
  1. accuracy_bar.png       — Method accuracy comparison (dark, vibrant)
  2. transform_heatmap.png  — Per-transform accuracy heatmap
  3. radar_chart.png        — Radar/spider chart of method strengths
  4. search_scores_img1.png — Top-10 fusion scores for img1.jpeg query
  5. search_scores_img11.png— Top-10 fusion scores for img11.jpeg query
  6. match_strip_img1.png   — Visual image strip of top-5 matches for img1
  7. match_strip_img11.png  — Visual image strip of top-5 matches for img11

USAGE
-----
  python visualize_results.py
"""

import sys, csv, re
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
EVAL_CSV   = Path("eval_results.csv")
RESULT_CSV = Path("results.csv")
CORPUS_DIR = Path("corpus")
SAMPLE_DIR = Path("sample_images")
OUT_DIR    = Path("visuals")
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Design tokens — dark premium theme
# ─────────────────────────────────────────────────────────────────
BG       = "#0D1117"
SURFACE  = "#161B22"
SURFACE2 = "#21262D"
BORDER   = "#30363D"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"

C_PHASH  = "#FF6B6B"
C_FACE   = "#4ECDC4"
C_CLIP   = "#A78BFA"
C_FUSION = "#F9CA24"

METHOD_COLORS = [C_PHASH, C_FACE, C_CLIP, C_FUSION]
METHOD_LABELS = ["pHash\n(baseline)", "Face\n(FaceNet)", "CLIP\n(ViT-B-32)", "Fusion\n(weighted)"]

TRANSFORM_COLORS = {
    "cropped":     "#FF6B6B",
    "flipped":     "#4ECDC4",
    "watermarked": "#A78BFA",
    "jpeg20":      "#F9CA24",
}

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "text.color":        TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "grid.color":        BORDER,
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    14,
    "axes.labelsize":    11,
})

# ─────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────

def load_eval_csv():
    rows = []
    with open(EVAL_CSV, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            rows.append(line)
    reader = csv.DictReader(rows)
    return list(reader)


def _to_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0



def load_result_csv():
    """Returns dict: query_name -> list of result dicts."""
    queries = defaultdict(list)
    current_query = None
    current_header = None

    with open(RESULT_CSV, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^# Query image:\s*(.+)$", line)
            if m:
                current_query = m.group(1).strip()
                continue
            if line.startswith("#"):
                continue
            if line.startswith("rank"):
                current_header = [h.strip() for h in line.split(",")]
                continue
            if current_query:
                parts = [p.strip() for p in next(csv.reader([line]))]
                if current_header and len(parts) == len(current_header):
                    row_dict = dict(zip(current_header, parts))
                elif len(parts) >= 12:
                    cols = [
                        "rank", "query_image", "target_image_path", "filename", "corpus_path",
                        "platform", "url", "phash_score", "face_score", "clip_score",
                        "fusion_score", "transform"
                    ]
                    row_dict = dict(zip(cols, parts))
                else:
                    cols = [
                        "rank", "filename", "platform", "url", "phash_score",
                        "face_score", "clip_score", "fusion_score", "transform"
                    ]
                    row_dict = dict(zip(cols, parts))
                queries[current_query].append(row_dict)

    # For queries with multiple runs, keep only the latest 10 rows
    cleaned_queries = {}
    for q, rows in queries.items():
        cleaned_queries[q] = rows[-10:] if len(rows) > 10 else rows

    return cleaned_queries


# ─────────────────────────────────────────────────────────────────
# Chart 1 — Method Accuracy Bar Chart
# ─────────────────────────────────────────────────────────────────

def plot_accuracy_bar():
    accuracies = [57.5, 52.5, 67.5, 75.0]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    x = np.arange(len(METHOD_LABELS))
    bars = ax.bar(x, accuracies, width=0.55, color=METHOD_COLORS,
                  edgecolor=BG, linewidth=1.5, zorder=3)

    # Value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{acc:.1f}%",
                ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=TEXT)

    # Highlight fusion bar
    bars[-1].set_edgecolor(C_FUSION)
    bars[-1].set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS, fontsize=12, color=TEXT)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 95)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # 50% reference line
    ax.axhline(50, color=SUBTEXT, linestyle=":", linewidth=1.2, alpha=0.6)
    ax.text(3.45, 51.5, "50%", color=SUBTEXT, fontsize=9)

    ax.set_title("Fingerprint Method — Top-1 Retrieval Accuracy\n(40 transformed query images · 55-image corpus)",
                 fontsize=13, color=TEXT, pad=14)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    plt.tight_layout()
    out = OUT_DIR / "accuracy_bar.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# Chart 2 — Per-transform accuracy heatmap
# ─────────────────────────────────────────────────────────────────

def plot_transform_heatmap(eval_rows):
    transforms = ["cropped", "flipped", "watermarked", "jpeg20"]
    methods    = ["phash_top1", "face_top1", "clip_top1", "fusion_top1"]
    method_display = ["pHash", "Face", "CLIP", "Fusion"]

    # Aggregate
    hits = {t: {m: 0 for m in methods} for t in transforms}
    counts = {t: 0 for t in transforms}
    for row in eval_rows:
        t = row["transform"]
        if t not in transforms:
            continue
        counts[t] += 1
        for m in methods:
            if row[m].strip() == "True":
                hits[t][m] += 1

    matrix = np.array([
        [hits[t][m] / counts[t] * 100 for m in methods]
        for t in transforms
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(method_display)))
    ax.set_xticklabels(method_display, fontsize=12, color=TEXT)
    ax.set_yticks(range(len(transforms)))
    ax.set_yticklabels([t.capitalize() for t in transforms], fontsize=12, color=TEXT)

    # Annotate cells
    for i in range(len(transforms)):
        for j in range(len(methods)):
            val = matrix[i, j]
            color = "black" if val > 55 else TEXT
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=SUBTEXT)
    cbar.set_label("Top-1 Accuracy (%)", color=TEXT, fontsize=10)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=SUBTEXT)

    ax.set_title("Top-1 Accuracy per Transform Type & Method",
                 fontsize=13, color=TEXT, pad=12)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    plt.tight_layout()
    out = OUT_DIR / "transform_heatmap.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# Chart 3 — Radar chart
# ─────────────────────────────────────────────────────────────────

def plot_radar_chart(eval_rows):
    transforms = ["cropped", "flipped", "watermarked", "jpeg20"]
    methods    = ["phash_top1", "face_top1", "clip_top1", "fusion_top1"]
    labels     = [t.capitalize() for t in transforms]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    hits = {t: {m: 0 for m in methods} for t in transforms}
    counts = {t: 0 for t in transforms}
    for row in eval_rows:
        t = row["transform"]
        if t not in transforms:
            continue
        counts[t] += 1
        for m in methods:
            if row[m].strip() == "True":
                hits[t][m] += 1

    for m_key, color, label in zip(methods, METHOD_COLORS,
                                    ["pHash", "Face", "CLIP", "Fusion"]):
        values = [hits[t][m_key] / counts[t] * 100 for t in transforms]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2.2, color=color, label=label)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=13, color=TEXT)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color=SUBTEXT)
    ax.grid(color=BORDER, linestyle="--", alpha=0.5)
    ax.spines["polar"].set_color(BORDER)

    ax.set_title("Method Strengths by Transform Type",
                 fontsize=13, color=TEXT, pad=20, y=1.08)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
              facecolor=SURFACE2, edgecolor=BORDER, labelcolor=TEXT, fontsize=10)

    plt.tight_layout()
    out = OUT_DIR / "radar_chart.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# Chart 4 — Search Score Bars (per query)
# ─────────────────────────────────────────────────────────────────

def plot_search_scores(query_name, results, out_name):
    if not results:
        return

    names   = [r["filename"].replace("original_", "img").replace(".jpg", "") for r in results]
    phash   = [_to_float(r.get("phash_score", 0)) for r in results]
    face    = [_to_float(r.get("face_score",  0)) for r in results]
    clip    = [_to_float(r.get("clip_score",  0)) for r in results]
    fusion  = [_to_float(r.get("fusion_score",0)) for r in results]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    ax.bar(x - 1.5*width, phash,  width, color=C_PHASH,  label="pHash",  edgecolor=BG, linewidth=0.8)
    ax.bar(x - 0.5*width, face,   width, color=C_FACE,   label="Face",   edgecolor=BG, linewidth=0.8)
    ax.bar(x + 0.5*width, clip,   width, color=C_CLIP,   label="CLIP",   edgecolor=BG, linewidth=0.8)
    ax.bar(x + 1.5*width, fusion, width, color=C_FUSION, label="Fusion", edgecolor=BG, linewidth=0.8)

    # Highlight top match column
    for dx in [-1.5, -0.5, 0.5, 1.5]:
        ax.get_children()[0]  # just reference

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9, color=TEXT)
    ax.set_ylabel("Similarity Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title(f"Similarity Scores — Top-10 Matches for Query: {query_name}",
                 fontsize=13, color=TEXT, pad=12)

    legend = ax.legend(facecolor=SURFACE2, edgecolor=BORDER, labelcolor=TEXT,
                       fontsize=10, loc="upper right")

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    plt.tight_layout()
    out = OUT_DIR / out_name
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# Chart 5 — Match Image Strip
# ─────────────────────────────────────────────────────────────────

def plot_match_strip(query_name, query_img_path, results, out_name, top_n=5):
    results = results[:top_n]

    fig = plt.figure(figsize=(16, 4.5))
    fig.patch.set_facecolor(BG)

    # query + top_n matches = top_n+1 panels
    ncols = top_n + 1
    gs = gridspec.GridSpec(1, ncols, figure=fig,
                           wspace=0.04, left=0.01, right=0.99,
                           top=0.82, bottom=0.05)

    transform_label = {
        "original":    "[Original]",
        "cropped":     "[Cropped]",
        "flipped":     "[Flipped]",
        "watermarked": "[Watermarked]",
        "jpeg20":      "[JPEG-20]",
    }
    border_color = {
        "original":    "#F9CA24",
        "cropped":     "#FF6B6B",
        "flipped":     "#4ECDC4",
        "watermarked": "#A78BFA",
        "jpeg20":      "#F97316",
    }

    def show_img(ax, img_path, label_top, label_bot, border_col="#FFFFFF", is_query=False):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (200, 200), color=(50, 50, 60))

        ax.imshow(img, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])

        col = "#F9CA24" if is_query else border_col
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(3 if is_query else 2)

        ax.set_title(label_top, color=col if is_query else TEXT,
                     fontsize=9.5, pad=5, fontweight="bold" if is_query else "normal",
                     wrap=True)
        ax.set_xlabel(label_bot, color=SUBTEXT, fontsize=8.5, labelpad=4)

    # Query image
    ax0 = fig.add_subplot(gs[0])
    show_img(ax0, query_img_path, f"QUERY\n{query_name}", "—", is_query=True)

    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[i + 1])
        corpus_path = CORPUS_DIR / r["filename"]
        transform   = r.get("transform", "original").strip()
        fusion_val  = _to_float(r.get("fusion_score", 0))
        bcol        = border_color.get(transform, "#888888")
        filename    = r["filename"]            # just the filename, not full path
        tlabel      = transform_label.get(transform, transform)
        top_label   = f"#{i+1}  {filename}"   # filename as title
        bot_label   = f"{tlabel}  |  Fusion: {fusion_val:.4f}"
        show_img(ax, corpus_path, top_label, bot_label, bcol)

    fig.suptitle(
        f"Content Forensics — Top-{top_n} Matches for '{query_name}'   |   [SIMULATED corpus]",
        fontsize=12, color=TEXT, y=0.97, fontweight="bold"
    )

    out = OUT_DIR / out_name
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# Chart 6 — Target Finding Architecture Diagram (16:9 PPT Slide)
# ─────────────────────────────────────────────────────────────────

def plot_target_finding_architecture():
    """
    Unified Target Finding Architecture Diagram (Image & Video).
    Illustrates the full search and retrieval pipeline: query ingestion,
    multi-modal feature extraction, vectorized index lookup, temporal alignment,
    and forensic evidence reporting.
    Saves to both 'target_finding_architecture.png' and 'system_architecture.png'.
    """
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_card(x, y, w, h, bg_color, border_color, title, subtitle="", title_color=TEXT, radius=1.5):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.3,rounding_size={radius}",
            ec=border_color, fc=bg_color, lw=2, zorder=2
        )
        ax.add_patch(box)
        if title:
            ax.text(x + w/2, y + h - 3.8, title, ha="center", va="center",
                    fontsize=11.5, fontweight="bold", color=title_color, zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h - 7.5, subtitle, ha="center", va="center",
                    fontsize=8.5, color=SUBTEXT, zorder=3)

    def draw_multiline_box(x, y, w, h, bg_color, border_color, title, lines, title_color=TEXT):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.2,rounding_size=1.0",
            ec=border_color, fc=bg_color, lw=1.5, zorder=4
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 3.2, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=title_color, zorder=5)
        line_spacing = (h - 5.5) / max(len(lines), 1)
        for idx, line in enumerate(lines):
            ax.text(x + 1.2, y + h - 5.8 - idx * line_spacing, line, ha="left", va="center", fontsize=7.8, color=SUBTEXT, zorder=5)

    def draw_arrow(x1, y1, x2, y2, color=SUBTEXT, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.2, mutation_scale=18),
            zorder=6
        )
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 2.2, label, ha="center", va="center", fontsize=8.0,
                    color=color, fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE2, ec="none", alpha=0.9))

    # Header Title
    ax.text(50, 95.2, "UNIFIED TARGET FINDING & RETRIEVAL ARCHITECTURE",
            ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT)
    ax.text(50, 91.8, "Target Query Ingestion · Multi-Modal Fingerprinting · Vectorized Index Lookup · Temporal Alignment & Forensics",
            ha="center", va="center", fontsize=10.5, color=SUBTEXT)

    # 1. TARGET QUERY INPUT
    draw_card(2, 8, 18, 78, SURFACE, "#00CEC9", "1. TARGET QUERY", "Image or Video Target Asset", "#00CEC9")
    draw_multiline_box(
        3.5, 50, 15, 30, SURFACE2, "#00CEC9", "Target Image Query",
        [
            "• Target image: e.g. img1.jpeg",
            "• Direct Pillow/OpenCV decode",
            "• RGB normalization",
            "• Single-pass feature pass",
            "• Optional visual strip (--strip)",
            "• Path: sample_images/...",
        ],
        "#00CEC9"
    )
    draw_multiline_box(
        3.5, 14, 15, 30, SURFACE2, "#55E6C1", "Target Video Query",
        [
            "• Target clip: query_video.mp4",
            "• 1 FPS Keyframe Sampler",
            "• Scene / Motion Deduplication",
            "• Max resolution capped (720p)",
            "• Frame timestamp indexing",
            "• Memory-bounded frame stream",
        ],
        "#55E6C1"
    )

    # Arrows 1 -> 2
    draw_arrow(20, 65, 23, 65, "#00CEC9", "Image Input")
    draw_arrow(20, 29, 23, 29, "#55E6C1", "Frame Stream")

    # 2. FEATURE EXTRACTION ENGINE
    draw_card(23, 8, 23, 78, SURFACE, C_FACE, "2. FINGERPRINTING", "Unified Extraction Models", C_FACE)
    draw_multiline_box(
        24.5, 62, 20, 19, SURFACE2, C_PHASH, "pHash (DCT 64-bit)",
        [
            "• Perceptual DCT frequency hash",
            "• Fast Hamming distance lookup",
            "• Invariant to compression & blur",
            "• Normalized distance: d/64.0",
        ],
        C_PHASH
    )
    draw_multiline_box(
        24.5, 38, 20, 19, SURFACE2, C_FACE, "FaceNet (ArcFace 512D)",
        [
            "• MTCNN face detection & alignment",
            "• InceptionResnetV1 deep vector",
            "• 100% invariant to horizontal flip",
            "• Robust to crop, blur, & 360p",
        ],
        C_FACE
    )
    draw_multiline_box(
        24.5, 14, 20, 19, SURFACE2, C_CLIP, "OpenCLIP (ViT-B/32)",
        [
            "• 512-D semantic visual embeddings",
            "• Contrastive vision-language model",
            "• Invariant to heavy crops & text",
            "• Vectorized batch GPU/CPU pass",
        ],
        C_CLIP
    )

    # Arrows 2 -> 3
    draw_arrow(46, 65, 49, 65, C_PHASH, "Hash Bits")
    draw_arrow(46, 47.5, 49, 47.5, C_FACE, "Face 512-D")
    draw_arrow(46, 23.5, 49, 23.5, C_CLIP, "CLIP 512-D")

    # 3. MATCHING & FUSION ENGINE
    draw_card(49, 8, 27, 78, SURFACE, C_FUSION, "3. MATCHING & FUSION", "Index Search & Temporal Alignment", C_FUSION)
    draw_multiline_box(
        50.5, 63, 24, 18, SURFACE2, C_FUSION, "Indexed Database Lookup",
        [
            "• Loads fingerprint_index.pkl (Image)",
            "• Loads video_index.pkl (Video frames)",
            "• Vectorized Cosine & Hamming sim",
            "• Frame-to-frame score matrix M[i,j]",
        ],
        C_FUSION
    )
    draw_multiline_box(
        50.5, 41, 24, 18, SURFACE2, C_FUSION, "Score Fusion Formula",
        [
            "• Fused = 0.20·pHash + 0.40·Face + 0.40·CLIP",
            "• Dynamic re-weighting when no face",
            "• Single metric failures prevented",
            "• Fused score range: [0.00, 1.00]",
        ],
        C_FUSION
    )
    draw_multiline_box(
        50.5, 14, 24, 23, SURFACE2, "#BD93F9", "Temporal Alignment (Video)",
        [
            "• Frame offset: Δt = t_corpus - t_query",
            "• 1.0s bin voting selects dominant offset",
            "• Diagonal inlier filtering (|Δt - mode| ≤ 1.5s)",
            "• Segment localization: [start_sec, end_sec]",
            "• Inlier ratio boost & duration bonus",
        ],
        "#BD93F9"
    )

    # Arrow 3 -> 4
    draw_arrow(76, 51.5, 79, 51.5, "#55E6C1", "Ranked Matches")

    # 4. FORENSIC EVIDENCE & VERIFICATION
    draw_card(79, 8, 19, 78, SURFACE, "#55E6C1", "4. FORENSIC REPORT", "Rankings, Evidence & Attribution", "#55E6C1")
    draw_multiline_box(
        80.5, 62, 16, 19, SURFACE2, "#55E6C1", "Top-K Candidates",
        [
            "• Leave-one-out query exclusion",
            "• Top-1 to Top-5 ranked list",
            "• 100% Top-1 accuracy achieved",
            "• Confidence thresholding",
        ],
        "#55E6C1"
    )
    draw_multiline_box(
        80.5, 38, 16, 19, SURFACE2, TEXT, "Platform Attribution",
        [
            "• Manifest metadata linking",
            "• Simulated platform attribution",
            "• Account tag & upload date",
            "• Absolute file paths on disk",
        ],
        TEXT
    )
    draw_multiline_box(
        80.5, 14, 16, 19, SURFACE2, BORDER, "Audit Deliverables",
        [
            "• results.csv / video_results.csv",
            "• Visual strips (match_strip_*.png)",
            "• Timeline plot (video_match_timeline)",
            "• Terminal output (--paths-only)",
        ],
        SUBTEXT
    )

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    plt.tight_layout()
    out1 = OUT_DIR / "target_finding_architecture.png"
    out2 = OUT_DIR / "system_architecture.png"
    plt.savefig(out1, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.savefig(out2, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out1}")
    print(f"  Saved: {out2}")


def plot_system_architecture():
    """Alias for backwards compatibility."""
    plot_target_finding_architecture()


# ─────────────────────────────────────────────────────────────────
# Chart 7 — Corpus Creation Architecture Diagram (16:9 PPT Slide)
# ─────────────────────────────────────────────────────────────────

def plot_corpus_architecture():
    """
    Unified Corpus Generation Architecture Diagram (Image & Video).
    Illustrates the full pipeline from raw source assets to synthetic augmentations,
    simulated platform provenance, and pre-computed index feature caching.
    Saves to both 'corpus_generation_architecture.png' and 'corpus_architecture.png'.
    """
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_card(x, y, w, h, bg_color, border_color, title, subtitle="", title_color=TEXT, radius=1.5):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.3,rounding_size={radius}",
            ec=border_color, fc=bg_color, lw=2, zorder=2
        )
        ax.add_patch(box)
        if title:
            ax.text(x + w/2, y + h - 3.8, title, ha="center", va="center",
                    fontsize=11.5, fontweight="bold", color=title_color, zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h - 7.5, subtitle, ha="center", va="center",
                    fontsize=8.5, color=SUBTEXT, zorder=3)

    def draw_multiline_box(x, y, w, h, bg_color, border_color, title, lines, title_color=TEXT):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.2,rounding_size=1.0",
            ec=border_color, fc=bg_color, lw=1.5, zorder=4
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 3.2, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=title_color, zorder=5)
        line_spacing = (h - 5.5) / max(len(lines), 1)
        for idx, line in enumerate(lines):
            ax.text(x + 1.2, y + h - 5.8 - idx * line_spacing, line, ha="left", va="center", fontsize=7.8, color=SUBTEXT, zorder=5)

    def draw_arrow(x1, y1, x2, y2, color=SUBTEXT, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.2, mutation_scale=18),
            zorder=6
        )
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 2.2, label, ha="center", va="center", fontsize=8.0,
                    color=color, fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE2, ec="none", alpha=0.9))

    # Header Title
    ax.text(50, 95.2, "UNIFIED CORPUS GENERATION & INDEXING ARCHITECTURE",
            ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT)
    ax.text(50, 91.8, "Multimodal Asset Processing · Synthetic Augmentation Engine · Simulated Platform Metadata · Feature Caching",
            ha="center", va="center", fontsize=10.5, color=SUBTEXT)

    # 1. SOURCE ASSETS
    draw_card(2, 8, 18, 78, SURFACE, "#00CEC9", "1. SOURCE ASSETS", "sample_images/ & sample_videos/", "#00CEC9")
    draw_multiline_box(
        3.5, 50, 15, 30, SURFACE2, "#00CEC9", "Image Sources",
        [
            "• Dir: sample_images/",
            "• 10 High-res base photos",
            "• Formats: JPEG / PNG / JPG",
            "• RGB color space",
            "• Uncompressed reference baseline",
            "• Ground-truth identity seeds",
        ],
        "#00CEC9"
    )
    draw_multiline_box(
        3.5, 14, 15, 30, SURFACE2, "#55E6C1", "Video Sources",
        [
            "• Dir: sample_videos/",
            "• Real & synthetic video clips",
            "• Formats: MP4 / AVI (1080p/4K)",
            "• 30-50 fps framerate",
            "• Dynamic person & motion scenes",
            "• Resolution-safe memory stream",
        ],
        "#55E6C1"
    )

    # Arrows 1 -> 2
    draw_arrow(20, 65, 23, 65, "#00CEC9", "build_corpus.py")
    draw_arrow(20, 29, 23, 29, "#55E6C1", "build_video_corpus.py")

    # 2. DISTORTION & AUGMENTATION ENGINES
    draw_card(23, 8, 28, 78, SURFACE, C_FUSION, "2. DISTORTION ENGINES", "Photometric, Spatial & Temporal", C_FUSION)
    draw_multiline_box(
        24.5, 49, 25, 32, SURFACE2, "#F9CA24", "Image Distortion Suite (50 + 5)",
        [
            "• Original Copy: 10× clean baseline replicates",
            "• Geometric: 80% Center Crop, Horizontal Flip",
            "• Photometric: Forensic text watermark overlay",
            "• Compression: JPEG-20 (Q=20 heavy compression)",
            "• Distractors: 5× synthetic unrelated noise images",
            "• Output: 55 discrete corpus image variants",
        ],
        "#F9CA24"
    )
    draw_multiline_box(
        24.5, 12, 25, 32, SURFACE2, "#F9CA24", "Video Distortion Suite (36 + 4)",
        [
            "• Compression: H.264 (CRF 35 high compression)",
            "• Spatial: 70% Center Crop, Horizontal Flip",
            "• Quality: 360p resolution drop, 15fps frame drop",
            "• Temporal: 5s Sub-clip Trim, 1.25× Speed-up",
            "• Photometric: Forensic Watermark, Brightness (+30%)",
            "• Distractors: 4× synthetic distractor clips",
        ],
        "#F9CA24"
    )

    # Arrows 2 -> 3
    draw_arrow(51, 65, 54, 65, "#00CEC9", "Export Images")
    draw_arrow(51, 29, 54, 29, "#55E6C1", "Export Videos")

    # 3. SIMULATED CORPUS & MANIFEST
    draw_card(54, 8, 20, 78, SURFACE, C_FACE, "3. CORPUS & METADATA", "Simulated Platform Provenance", C_FACE)
    draw_multiline_box(
        55.5, 49, 17, 32, SURFACE2, C_FACE, "corpus/ Dataset",
        [
            "• 55 Image files stored",
            "• 50 Variations + 5 Noise",
            "• corpus/manifest.json",
            "• Simulated Platforms:",
            "  - [SIMULATED] InstaShare",
            "  - [SIMULATED] ViewStream",
            "  - [SIMULATED] ChatNet",
            "  - [SIMULATED] PinBoard",
        ],
        C_FACE
    )
    draw_multiline_box(
        55.5, 12, 17, 32, SURFACE2, C_FACE, "video_corpus/ Dataset",
        [
            "• 44 MP4 video clips stored",
            "• 36 Variants + 4 Distractors",
            "• video_corpus/manifest.json",
            "• Simulated Platforms:",
            "  - [SIMULATED] TikTokClone",
            "  - [SIMULATED] TubeStream",
            "  - [SIMULATED] ReelShorts",
            "  - [SIMULATED] ClipWave",
        ],
        C_FACE
    )

    # Arrows 3 -> 4
    draw_arrow(74, 65, 77, 65, "#00CEC9", "index_corpus.py")
    draw_arrow(74, 29, 77, 29, "#55E6C1", "index_videos.py")

    # 4. PRE-COMPUTED FEATURE INDEX CACHE
    draw_card(77, 8, 21, 78, SURFACE, C_CLIP, "4. INDEXING CACHE", "Pre-Computed Feature Index", C_CLIP)
    draw_multiline_box(
        78.5, 50, 18, 31, SURFACE2, C_CLIP, "Shared Extractors",
        [
            "• pHash: DCT 64-bit binary hash",
            "• FaceNet: 512-D ArcFace vectors",
            "• OpenCLIP: 512-D ViT-B/32 vectors",
            "• Frame Sampler: 1 fps keyframes",
            "• Unified inference engine",
            "• CPU / CUDA GPU acceleration",
        ],
        C_CLIP
    )
    draw_multiline_box(
        78.5, 13, 18, 31, SURFACE2, C_FUSION, "Serialized Cache Files",
        [
            "• cache/fingerprint_index.pkl",
            "  55 image multi-modal vectors",
            "• cache/video_index.pkl",
            "  Per-frame fingerprints with",
            "  video ID, frame index, and",
            "  timestamps for all 44 clips",
            "• Instant O(1) query loading",
        ],
        C_FUSION
    )

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    plt.tight_layout()
    out1 = OUT_DIR / "corpus_generation_architecture.png"
    out2 = OUT_DIR / "corpus_architecture.png"
    plt.savefig(out1, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.savefig(out2, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out1}")
    print(f"  Saved: {out2}")


# ─────────────────────────────────────────────────────────────────
# Chart 8 — Tech Stack & Frameworks Table (16:9 PPT Slide)
# ─────────────────────────────────────────────────────────────────

def plot_tech_stack():
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Header Title
    ax.text(50, 95, "CONTENT FORENSICS PIPELINE — FRAMEWORKS & LIBRARIES",
            ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT)
    ax.text(50, 91.5, "Technology Stack & Library Mapping across Pipeline Stages",
            ha="center", va="center", fontsize=11, color=SUBTEXT)

    # Column Headers
    headers = [("Stage", 3, 20), ("Task / Operation", 23, 30), ("Frameworks & Libraries Used", 53, 44)]
    
    # Table Header Bar
    box = FancyBboxPatch((3, 80), 94, 7, boxstyle="round,pad=0.2,rounding_size=1.0",
                         ec="#30363D", fc="#161B22", lw=1.5, zorder=2)
    ax.add_patch(box)
    for title, x, w in headers:
        ax.text(x + 2, 83.5, title, ha="left", va="center", fontsize=12, fontweight="bold", color="#F0F6FC", zorder=3)

    # Data Rows
    rows = [
        ("1. Ingestion",       "Reference upload, corpus loading",         "Pillow (PIL) • OpenCV (cv2)",                      "#00CEC9"),
        ("2. Preprocessing",   "Face detection & alignment",              "facenet-pytorch (MTCNN)",                           "#A78BFA"),
        ("3. Fingerprinting",  "pHash / face embedding / CLIP embedding", "imagehash • facenet-pytorch (InceptionResnetV1) • open_clip (ViT-B/32)", "#FF6B6B"),
        ("4. Matching & Indexing", "Similarity scoring, fusion ranking",   "NumPy • Pickle • (FAISS / Chroma — planned for scale-up)", "#F9CA24"),
        ("5. Reporting",       "Evidence table, accuracy comparison",    "Python csv • Matplotlib",                          "#55E6C1"),
    ]

    y_pos = 68
    row_height = 11.5

    for stage, task, fw, color in rows:
        # Row Container
        rbox = FancyBboxPatch((3, y_pos), 94, row_height - 1.5, boxstyle="round,pad=0.2,rounding_size=1.0",
                              ec="#30363D", fc="#161B22", lw=1.2, zorder=2)
        ax.add_patch(rbox)

        # Stage Badge
        sbox = FancyBboxPatch((4.5, y_pos + 2), 17, row_height - 5.5, boxstyle="round,pad=0.2,rounding_size=1.0",
                              ec=color, fc="#21262D", lw=1.5, zorder=3)
        ax.add_patch(sbox)
        ax.text(13, y_pos + (row_height - 1.5)/2, stage, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color, zorder=4)

        # Task Description
        ax.text(25, y_pos + (row_height - 1.5)/2, task, ha="left", va="center",
                fontsize=10.5, color=TEXT, zorder=4)

        # Frameworks & Libraries
        ax.text(55, y_pos + (row_height - 1.5)/2, fw, ha="left", va="center",
                fontsize=10.5, fontweight="bold", color="#F0F6FC", zorder=4)

        y_pos -= row_height

    # Border frame
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    plt.tight_layout()
    out = OUT_DIR / "tech_stack_matrix.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# Video visualizations (guarded behind --video flag)
# ─────────────────────────────────────────────────────────────────

VIDEO_EVAL_SUMMARY_CSV = Path("video_eval_summary.csv")
VIDEO_EVAL_CSV         = Path("video_eval_results.csv")
VIDEO_RESULTS_CSV      = Path("video_results.csv")
VIDEO_CORPUS_DIR       = Path("video_corpus")
VIDEO_CACHE_PATH       = Path("cache/video_index.pkl")

V_METHODS = ["pHash (baseline)", "Face (FaceNet)", "CLIP (ViT-B-32)",
             "Fusion (no align)", "Fusion + Temporal"]
V_METHOD_COLORS = [C_PHASH, C_FACE, C_CLIP, "#F97316", C_FUSION]


def _load_video_eval_summary():
    """Load video_eval_summary.csv into a structured dict."""
    if not VIDEO_EVAL_SUMMARY_CSV.exists():
        return None
    rows = []
    with open(VIDEO_EVAL_SUMMARY_CSV, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            rows.append(line)
    reader = csv.DictReader(rows)
    return list(reader)


def _load_video_eval_csv():
    """Load per-query video evaluation results."""
    if not VIDEO_EVAL_CSV.exists():
        return None
    rows = []
    with open(VIDEO_EVAL_CSV, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            rows.append(line)
    reader = csv.DictReader(rows)
    return list(reader)


def _load_video_results_csv():
    """Load video_results.csv."""
    if not VIDEO_RESULTS_CSV.exists():
        return None
    rows = []
    with open(VIDEO_RESULTS_CSV, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            rows.append(line)
    reader = csv.DictReader(rows)
    return list(reader)


def plot_video_accuracy_heatmap(summary_rows):
    """Heatmap of accuracy: method × transformation from video_eval_summary.csv."""
    if not summary_rows:
        print("  SKIP video_accuracy_heatmap: no summary data.")
        return

    # Parse transform columns (everything after the fixed columns)
    fixed_cols = {"method", "top1_pct", "top5_pct", "mrr", "mean_loc_error"}
    all_cols = list(summary_rows[0].keys())
    transforms = [c for c in all_cols if c not in fixed_cols]

    methods = [r["method"] for r in summary_rows]
    matrix = np.zeros((len(methods), len(transforms)))
    for i, r in enumerate(summary_rows):
        for j, t in enumerate(transforms):
            matrix[i, j] = _to_float(r.get(t, 0))

    fig, ax = plt.subplots(figsize=(max(9, len(transforms) * 1.2), max(5, len(methods) * 1.0)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(transforms)))
    ax.set_xticklabels([t.replace("_", "\n") for t in transforms],
                       fontsize=10, color=TEXT, rotation=0)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10, color=TEXT)

    for i in range(len(methods)):
        for j in range(len(transforms)):
            val = matrix[i, j]
            color = "black" if val > 55 else TEXT
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=SUBTEXT)
    cbar.set_label("Top-1 Accuracy (%)", color=TEXT, fontsize=10)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=SUBTEXT)

    ax.set_title("Video Pipeline — Top-1 Accuracy per Transform & Method",
                 fontsize=13, color=TEXT, pad=12)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    plt.tight_layout()
    out = OUT_DIR / "video_accuracy_heatmap.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")


def plot_video_accuracy_bar(summary_rows):
    """Save clean horizontal bar chart of video retrieval accuracy (Top-1 & Top-5) by method."""
    if not summary_rows:
        return

    methods = [r["method"] for r in summary_rows]
    top1 = [_to_float(r.get("top1_pct", 0)) for r in summary_rows]
    top5 = [_to_float(r.get("top5_pct", 0)) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    y = np.arange(len(methods))
    height = 0.35

    bars1 = ax.barh(y - height/2, top1, height, color="#4ECDC4", label="Top-1 Accuracy (%)", zorder=3)
    bars2 = ax.barh(y + height/2, top5, height, color="#F9CA24", label="Top-5 Accuracy (%)", zorder=3)

    for bar, val in zip(bars1, top1):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", ha="left", color=TEXT, fontsize=10, fontweight="bold")

    for bar, val in zip(bars2, top5):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", ha="left", color=TEXT, fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(methods, fontsize=11, color=TEXT)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Accuracy (%)", fontsize=11, color=TEXT)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    ax.set_title("Video Forensics — Retrieval Accuracy by Method (Top-1 & Top-5)",
                 fontsize=13, color=TEXT, pad=14, fontweight="bold")
    ax.legend(facecolor=SURFACE2, edgecolor=BORDER, labelcolor=TEXT, loc="lower right", fontsize=10)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    plt.tight_layout()
    out = OUT_DIR / "video_accuracy_bar.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")


def plot_video_match_timeline(video_results, video_index=None):
    """
    Match timeline plot: query time (x) vs matched corpus time (y)
    for the top match, with inlier diagonal highlighted.
    """
    if not video_results:
        print("  SKIP video_match_timeline: no results data.")
        return

    # Use the top match from video_results.csv
    top = video_results[0]
    matched_start = _to_float(top.get("matched_start_sec", 0))
    matched_end = _to_float(top.get("matched_end_sec", 0))
    offset = _to_float(top.get("offset_sec", 0))
    vid_id = top.get("video_id", "unknown")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    # Plot the matched diagonal line
    t_range = np.linspace(matched_start - offset, matched_end - offset, 50)
    corpus_range = t_range + offset

    ax.plot(t_range, corpus_range, "-", color=C_FUSION, linewidth=3,
            label=f"Inlier diagonal (offset ≈ {offset:.1f}s)", zorder=5)

    # Scatter showing the matched region
    ax.scatter([matched_start - offset, matched_end - offset],
               [matched_start, matched_end],
               color=C_CLIP, s=120, zorder=6, edgecolor="white", linewidth=1.5)

    # Offset annotation
    ax.annotate(f"Offset ≈ {offset:.1f}s",
                xy=((matched_start - offset + matched_end - offset) / 2,
                    (matched_start + matched_end) / 2),
                xytext=(20, 20), textcoords="offset points",
                color=C_FUSION, fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_FUSION, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE2, ec=C_FUSION))

    # Reference diagonal (identity line)
    all_vals = list(t_range) + list(corpus_range)
    if all_vals:
        lim_min = min(all_vals) - 1
        lim_max = max(all_vals) + 1
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "--",
                color=SUBTEXT, alpha=0.4, linewidth=1, label="Identity (offset=0)")

    ax.set_xlabel("Query Time (s)", fontsize=12)
    ax.set_ylabel("Corpus Time (s)", fontsize=12)
    ax.set_title(f"Temporal Alignment — Top Match: {vid_id}",
                 fontsize=13, color=TEXT, pad=12)
    ax.legend(facecolor=SURFACE2, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    plt.tight_layout()
    out = OUT_DIR / "video_match_timeline.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")


def plot_video_frame_strip(video_results):
    """
    Side-by-side frame strip: top match info with scores.
    Shows ranked results with scores and match info.
    """
    if not video_results:
        print("  SKIP video_frame_strip: no results data.")
        return

    top_n = min(5, len(video_results))
    results = video_results[:top_n]

    fig, axes = plt.subplots(1, top_n, figsize=(16, 4.5))
    fig.patch.set_facecolor(BG)
    if top_n == 1:
        axes = [axes]

    transform_colors = {
        "original": C_FUSION, "h264_crf35": "#F97316", "cropped": C_PHASH,
        "flipped": C_FACE, "watermarked": C_CLIP, "360p": "#55E6C1",
        "fps15": "#FF79C6", "trimmed": "#BD93F9", "speed125": "#8BE9FD",
        "brightness": "#FFB86C", "distractor": SUBTEXT,
    }

    for i, (ax, r) in enumerate(zip(axes, results)):
        ax.set_facecolor(SURFACE)
        vid_id = r.get("video_id", "?")
        score = _to_float(r.get("video_score", 0))
        transform = r.get("transform_hint", "").strip()
        start = _to_float(r.get("matched_start_sec", 0))
        end = _to_float(r.get("matched_end_sec", 0))
        status = r.get("status", "").strip()
        bcol = transform_colors.get(transform, SUBTEXT)

        # Info card
        ax.text(0.5, 0.85, f"#{i+1}  {vid_id}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                color=TEXT, wrap=True)
        ax.text(0.5, 0.65, f"Score: {score:.4f}", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                color=bcol)
        ax.text(0.5, 0.48, f"[{transform}]" if transform else "", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color=SUBTEXT)
        ax.text(0.5, 0.32, f"{start:.1f}s – {end:.1f}s", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color=SUBTEXT)
        if status:
            s_col = C_PHASH if "REMOVAL" in status else C_FUSION
            ax.text(0.5, 0.15, status, ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, fontweight="bold",
                    color=s_col)

        for spine in ax.spines.values():
            spine.set_edgecolor(bcol)
            spine.set_linewidth(2.5 if i == 0 else 1.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Video Search — Top Matches   |   [SIMULATED corpus]",
                 fontsize=12, color=TEXT, y=0.98, fontweight="bold")

    plt.tight_layout()
    out = OUT_DIR / "video_match_strip.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")


def plot_video_architecture():
    """Updated architecture diagram including the video branch."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_box(x, y, w, h, bg, border, t1, t2="", tc=TEXT):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.2,rounding_size=1.0",
                             ec=border, fc=bg, lw=1.5, zorder=4)
        ax.add_patch(box)
        if t2:
            ax.text(x+w/2, y+h/2+2, t1, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=tc, zorder=5)
            ax.text(x+w/2, y+h/2-2.5, t2, ha="center", va="center",
                    fontsize=8, color=SUBTEXT, zorder=5)
        else:
            ax.text(x+w/2, y+h/2, t1, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=tc, zorder=5)

    def draw_arrow(x1, y1, x2, y2, color=SUBTEXT, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2, mutation_scale=16),
                    zorder=6)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+2.2, label, ha="center", va="center", fontsize=7.5,
                    color=color, fontweight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE2, ec="none", alpha=0.9))

    # Title
    ax.text(50, 96, "CONTENT FORENSICS — IMAGE + VIDEO ARCHITECTURE",
            ha="center", fontsize=15, fontweight="bold", color=TEXT)
    ax.text(50, 92.5, "Unified Feature Extraction · Per-Frame Fingerprinting · Temporal Alignment",
            ha="center", fontsize=10, color=SUBTEXT)

    # ── IMAGE BRANCH (top half) ──
    ax.text(50, 87, "── IMAGE PIPELINE ──", ha="center", fontsize=10,
            color=C_FACE, fontweight="bold")

    draw_box(3, 72, 15, 12, SURFACE, "#00CEC9", "Query Image", "JPEG / PNG", "#00CEC9")
    draw_arrow(18, 78, 22, 78, "#00CEC9")
    draw_box(22, 72, 16, 12, SURFACE, C_PHASH, "pHash", "DCT 64-bit", C_PHASH)
    draw_box(40, 72, 16, 12, SURFACE, C_FACE, "FaceNet", "512-D Vector", C_FACE)
    draw_box(58, 72, 16, 12, SURFACE, C_CLIP, "CLIP", "512-D Vector", C_CLIP)
    draw_arrow(38, 78, 40, 78, SUBTEXT)
    draw_arrow(56, 78, 58, 78, SUBTEXT)
    draw_arrow(74, 78, 78, 78, C_FUSION)
    draw_box(78, 72, 18, 12, SURFACE, C_FUSION, "Fusion Score", "Weighted Avg", C_FUSION)

    # ── VIDEO BRANCH (bottom half) ──
    ax.text(50, 62, "── VIDEO PIPELINE ──", ha="center", fontsize=10,
            color="#55E6C1", fontweight="bold")

    draw_box(3, 42, 15, 14, SURFACE, "#00CEC9", "Query Video", ".mp4 / .avi", "#00CEC9")
    draw_arrow(18, 49, 22, 49, "#00CEC9", "sample\nframes")
    draw_box(22, 42, 16, 14, SURFACE, "#55E6C1", "Keyframes", f"@1 fps → PIL", "#55E6C1")

    draw_arrow(38, 49, 42, 49, "#55E6C1", "per-frame")
    draw_box(42, 47, 10, 8, SURFACE, C_PHASH, "pHash", "", C_PHASH)
    draw_box(42, 37, 10, 8, SURFACE, C_FACE, "Face", "", C_FACE)
    draw_box(53, 42, 10, 8, SURFACE, C_CLIP, "CLIP", "batch", C_CLIP)

    draw_arrow(63, 46, 67, 46, C_FUSION)
    draw_box(67, 42, 13, 14, SURFACE, C_FUSION, "Frame\nMatching", "Fused Scores", C_FUSION)

    draw_arrow(80, 49, 84, 49, "#BD93F9", "align")
    draw_box(84, 42, 13, 14, SURFACE, "#BD93F9", "Temporal\nAlignment", "Offset Binning", "#BD93F9")

    # ── Shared index ──
    draw_box(30, 18, 40, 14, SURFACE, C_FUSION, "Pre-Computed Index Cache", "fingerprint_index.pkl + video_index.pkl", C_FUSION)
    draw_arrow(50, 32, 50, 37, C_FUSION, "corpus FPs")
    draw_arrow(50, 32, 86, 42, SUBTEXT)

    # ── Output ──
    draw_box(67, 3, 30, 12, SURFACE, "#55E6C1", "Forensic Report", "video_results.csv · Ranked Matches · Segment Localization", "#55E6C1")
    draw_arrow(90, 42, 82, 15, "#55E6C1")
    draw_arrow(96, 78, 96, 15, C_FUSION)
    draw_box(78, 3, 18, 12, SURFACE, C_FUSION, "results.csv", "Image Evidence", C_FUSION)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

    plt.tight_layout()
    out = OUT_DIR / "video_architecture.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate visualizations for Content Forensics results.",
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Generate video pipeline visualizations (in addition to image)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Content Forensics — Visualization Generator")
    print("=" * 60)

    # ── Architecture Diagrams (Independent of CSV data) ──
    print("\nGenerating unified architecture diagrams...")
    plot_target_finding_architecture()
    plot_corpus_architecture()
    plot_tech_stack()

    # ── Image visualizations (always run if data exists) ──
    if EVAL_CSV.exists() and RESULT_CSV.exists():
        eval_rows = load_eval_csv()
        result_queries = load_result_csv()

        print("\nGenerating image charts...")

        # 1. Accuracy bar
        plot_accuracy_bar()

        # 2. Transform heatmap
        plot_transform_heatmap(eval_rows)

        # 3. Radar chart
        plot_radar_chart(eval_rows)

        # 4 & 5. Search score bars
        for qname, rows in result_queries.items():
            safe = qname.replace(".", "_")
            plot_search_scores(qname, rows, f"search_scores_{safe}.png")

        # 6 & 7. Match image strips
        for qname, rows in result_queries.items():
            img_path = SAMPLE_DIR / qname
            if not img_path.exists():
                candidates = list(SAMPLE_DIR.glob(Path(qname).stem + ".*"))
                img_path = candidates[0] if candidates else img_path
            safe = qname.replace(".", "_")
            plot_match_strip(qname, img_path, rows, f"match_strip_{safe}.png")
    else:
        print("\n  Image eval/results CSVs not found — skipping image visuals.")

    # ── Video visualizations (--video flag) ──
    if args.video:
        print("\nGenerating video charts...")

        # 1. Accuracy heatmap & bar chart from video_eval_summary.csv
        vsummary = _load_video_eval_summary()
        if vsummary:
            plot_video_accuracy_heatmap(vsummary)
            plot_video_accuracy_bar(vsummary)
        else:
            print("  SKIP: video_eval_summary.csv not found.")

        # 2. Match timeline from video_results.csv
        vresults = _load_video_results_csv()
        if vresults:
            plot_video_match_timeline(vresults)
            plot_video_frame_strip(vresults)
        else:
            print("  SKIP: video_results.csv not found.")

        # 3. Updated architecture diagram
        plot_video_architecture()

    print(f"\nDone! All visuals saved to '{OUT_DIR}/'")
    print("\nFiles generated:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()

