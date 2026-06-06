#!/usr/bin/env python3
"""
UNETR architecture figure — BraTS 2021.
From models/unetr.py + configs/exp03_unetr.yaml:
  ViT-B/16: L=12, K=768, 12 heads, patch=16³
  Input 128³×4 → 512 patches → linear proj → 768 dim
  Skip connections at z3, z6, z9, z12
  CNN decoder: 8³→16³→32³→64³→128³

Layout:
  - Transformer: Block 1 at TOP, Block 12 at BOTTOM
  - Decoder blocks: bottom-aligned
  - Transformer is positioned so that z3/z6/z9/z12 extraction points are
    at exactly the same y-level as the TOP of the corresponding decoder block.
  - Skip connections are therefore perfectly horizontal → no crossing.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-1.5, 16.5)
ax.set_ylim(-4.2, 7.5)

# ── Palette ───────────────────────────────────────────────────────────────────
TR_FILL    = "#EAD8FF"
TR_FILL_HI = "#C8A0FF"    # extraction blocks
TR_EDGE    = "#6600AA"
EMBED_FILL = "#D5C2FF"
DEC_FILLS  = ["#C0C0FF", "#9898EE", "#7070CC", "#4848AA", "#AADDAA"]
DEC_EDGES  = ["#111166"] * 4 + ["#226622"]
SKIP_C     = "#CC7700"
UP_C       = "#1A9922"
IN_SKIP_C  = "#2255AA"

BW = 0.18; BG = 0.06   # block/gap width
YBOT = -2.5             # bottom of all decoder blocks (bottom-aligned)

# ── Decoder block heights and derived y_tops ──────────────────────────────────
#  Stages: 8³(512ch), 16³(256ch), 32³(128ch), 64³(64ch), 128³(64ch)
DEC_INFO = [
    ("$8^3$",   512, 1.4),   # (res_label, channels, height)
    ("$16^3$",  256, 2.1),
    ("$32^3$",  128, 2.9),
    ("$64^3$",  64,  3.8),
    ("$128^3$", 64,  5.0),
]
dec_ytop = [YBOT + h for _, _, h in DEC_INFO]   # y-tops: [-1.1, -0.4, 0.4, 1.3, 2.5]

# ── Transformer block y-centers ───────────────────────────────────────────────
# Extraction blocks z3,z6,z9,z12 anchored to decoder tops:
#   Block 12 center = dec_ytop[0] = -1.1   (feeds 8³ decoder)
#   Block  9 center = dec_ytop[1] = -0.4
#   Block  6 center = dec_ytop[2] =  0.4
#   Block  3 center = dec_ytop[3] =  1.3
# Blocks between extractions are linearly interpolated.
extract_y = {12: dec_ytop[0], 9: dec_ytop[1], 6: dec_ytop[2], 3: dec_ytop[3]}

def interp_y(blk, anchors):
    """Linearly interpolate block center y between extraction anchors."""
    keys = sorted(anchors.keys())          # [3,6,9,12]
    # blocks above 3 (blocks 1,2): extrapolate upward
    sp = (anchors[6] - anchors[3]) / 3    # avg spacing in lower part
    above_3_y = {1: anchors[3] + 2*sp, 2: anchors[3] + sp}
    if blk in above_3_y:
        return above_3_y[blk]
    for i in range(len(keys)-1):
        lo, hi = keys[i], keys[i+1]
        if lo <= blk <= hi:
            t = (blk - lo) / (hi - lo)
            return anchors[lo] + t * (anchors[hi] - anchors[lo])
    return anchors[keys[-1]]

BLK_H = 0.33    # transformer block height
tr_centers = {b: interp_y(b, extract_y) for b in range(1, 13)}

# ── Transformer x-position ────────────────────────────────────────────────────
TR_X = 2.8; TR_W = 1.35

# ── Input block ───────────────────────────────────────────────────────────────
IN_X = 0.0; IN_W = 0.38; IN_H = 5.0
ax.add_patch(mpatches.FancyBboxPatch(
    (IN_X, YBOT), IN_W, IN_H,
    boxstyle="square,pad=0", lw=1.0,
    edgecolor="#2255AA", facecolor="#DDEEFF"))
ax.text(IN_X + IN_W/2, YBOT + IN_H + 0.15, "$128^3\\!\\times\\!4$",
        ha="center", va="bottom", fontsize=8)
ax.text(IN_X - 0.55, YBOT + IN_H/2, "Input",
        ha="center", va="center", fontsize=9)
ax.annotate("", xy=(IN_X, YBOT + IN_H/2),
            xytext=(IN_X - 0.2, YBOT + IN_H/2),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.1, mutation_scale=8))

# ── Patch Embedding ───────────────────────────────────────────────────────────
EMB_X = 1.0; EMB_Y = tr_centers[1] - 0.4; EMB_W = 1.2; EMB_H = 0.8
ax.add_patch(mpatches.FancyBboxPatch(
    (EMB_X, EMB_Y), EMB_W, EMB_H,
    boxstyle="round,pad=0.05", lw=1.0,
    edgecolor=TR_EDGE, facecolor=EMBED_FILL))
ax.text(EMB_X + EMB_W/2, EMB_Y + EMB_H/2, "Linear\nProjection",
        ha="center", va="center", fontsize=7.5)
ax.text(EMB_X + EMB_W/2, EMB_Y - 0.18, "$512\\times768$",
        ha="center", va="top", fontsize=7, color="#555")

# input → embedding
ax.annotate("", xy=(EMB_X, EMB_Y + EMB_H/2),
            xytext=(IN_X + IN_W, YBOT + IN_H * 0.78),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.1, mutation_scale=8,
                            connectionstyle="arc3,rad=-0.1"))
# embedding → transformer block 1
ax.annotate("", xy=(TR_X, tr_centers[1]),
            xytext=(EMB_X + EMB_W, EMB_Y + EMB_H/2),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.1, mutation_scale=8))

# ── Transformer encoder ───────────────────────────────────────────────────────
# Background box
tr_ymin = tr_centers[12] - BLK_H/2 - 0.15
tr_ymax = tr_centers[1]  + BLK_H/2 + 0.15
ax.add_patch(mpatches.FancyBboxPatch(
    (TR_X - 0.12, tr_ymin), TR_W + 0.24, tr_ymax - tr_ymin,
    boxstyle="round,pad=0.08", lw=1.2,
    edgecolor=TR_EDGE, facecolor="#F5EEFF", zorder=1))

# Blocks (1 at top, 12 at bottom)
for blk in range(1, 13):
    yc = tr_centers[blk]
    fc = TR_FILL_HI if blk in extract_y else TR_FILL
    ax.add_patch(mpatches.FancyBboxPatch(
        (TR_X, yc - BLK_H/2), TR_W, BLK_H,
        boxstyle="square,pad=0", lw=0.65,
        edgecolor=TR_EDGE, facecolor=fc, zorder=2))
    ax.text(TR_X + TR_W/2, yc, f"Block {blk}",
            ha="center", va="center", fontsize=6.2, zorder=3)

ax.text(TR_X + TR_W/2, tr_ymin - 0.35,
        "Transformer Encoder\n$L{=}12,\\ K{=}768,\\ h{=}12$",
        ha="center", va="top", fontsize=8, fontweight="bold", color=TR_EDGE)

# ── Decoder blocks (bottom-aligned at YBOT) ───────────────────────────────────
DEC_X_START = 5.5; DEC_STEP = 2.0

dec_cx = []
for j, (res, ch, h) in enumerate(DEC_INFO):
    x = DEC_X_START + j * DEC_STEP
    y = YBOT
    fc, ec = DEC_FILLS[j], DEC_EDGES[j]
    for dx in [0, BW + BG]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x + dx, y), BW, h,
            boxstyle="square,pad=0", lw=0.8,
            edgecolor=ec, facecolor=fc))
    cx = x + BW + BG/2
    dec_cx.append(cx)
    ax.text(cx, y + h + 0.15, res,   ha="center", va="bottom", fontsize=7.5)
    ax.text(cx, y + h + 0.60, f"{ch}ch", ha="center", va="bottom", fontsize=7, color="#444")

# ── Output conv ───────────────────────────────────────────────────────────────
OUT_X = DEC_X_START + 5 * DEC_STEP
out_h = 5.0
ax.add_patch(mpatches.FancyBboxPatch(
    (OUT_X, YBOT), BW * 0.7, out_h,
    boxstyle="square,pad=0", lw=0.8,
    edgecolor="#226622", facecolor="#AADDAA"))
ocx = OUT_X + BW * 0.35
ax.text(ocx, YBOT + out_h + 0.15, "$128^3$",  ha="center", va="bottom", fontsize=7.5)
ax.text(ocx, YBOT + out_h + 0.60, "4ch", ha="center", va="bottom", fontsize=7, color="#226622")
ax.text(ocx + 0.5, YBOT + out_h/2, "Output", ha="left", va="center", fontsize=9)
ax.annotate("", xy=(ocx + 0.45, YBOT + out_h/2),
            xytext=(ocx, YBOT + out_h/2),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.1, mutation_scale=8))

# ── Upsample arrows between decoder stages (at bottom) ────────────────────────
for j in range(4):
    x0 = DEC_X_START + j * DEC_STEP + 2*BW + BG
    x1 = DEC_X_START + (j+1) * DEC_STEP
    y0 = YBOT + DEC_INFO[j][2] / 2
    y1 = YBOT
    ax.annotate("", xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=UP_C, lw=1.3,
                                mutation_scale=9, connectionstyle="arc3,rad=0.0"))

# last decoder → output
ax.annotate("", xy=(OUT_X, YBOT + out_h/2),
            xytext=(DEC_X_START + 4*DEC_STEP + 2*BW + BG, YBOT + DEC_INFO[4][2]/2),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.1, mutation_scale=8))

# ── Skip connections: horizontal from transformer extraction to decoder tops ───
# z12→8³, z9→16³, z6→32³, z3→64³
skip_map = [(12, 0), (9, 1), (6, 2), (3, 3)]
for blk, dj in skip_map:
    y    = tr_centers[blk]             # extraction y = decoder top y (by construction)
    x_tr = TR_X + TR_W                 # right edge of transformer
    x_dc = DEC_X_START + dj * DEC_STEP  # left edge of decoder stage
    # horizontal line + arrowhead
    ax.annotate("", xy=(x_dc, y),
                xytext=(x_tr, y),
                arrowprops=dict(arrowstyle="-|>", color=SKIP_C, lw=1.1,
                                mutation_scale=8, linestyle="dashed"))
    # label z_i at transformer side
    ax.text(x_tr + 0.06, y + 0.12, f"$z_{{{blk}}}$",
            ha="left", va="bottom", fontsize=8, color=TR_EDGE, fontweight="bold")

# ── Input skip (z0) routed below figure ────────────────────────────────────────
# Goes: input bottom → down → right → up to 128³ decoder bottom
x_in_ctr = IN_X + IN_W/2
x_dec128  = DEC_X_START + 4 * DEC_STEP + BW  # approx centre of 128³ stage
y_route   = YBOT - 0.9

ax.plot([x_in_ctr, x_in_ctr], [YBOT, y_route],
        color=IN_SKIP_C, lw=1.0, linestyle="dashed")
ax.plot([x_in_ctr, x_dec128], [y_route, y_route],
        color=IN_SKIP_C, lw=1.0, linestyle="dashed")
ax.annotate("", xy=(x_dec128, YBOT),
            xytext=(x_dec128, y_route),
            arrowprops=dict(arrowstyle="-|>", color=IN_SKIP_C, lw=1.0,
                            mutation_scale=8))
# label
ax.text((x_in_ctr + x_dec128)/2, y_route - 0.15,
        "$z_0$  (input skip, full-resolution)",
        ha="center", va="top", fontsize=7.5, color=IN_SKIP_C)

# ── Positional embedding note ─────────────────────────────────────────────────
ax.text(EMB_X + EMB_W/2, EMB_Y + EMB_H + 0.1,
        "+  pos. emb. (learned)",
        ha="center", va="bottom", fontsize=6.5, color=TR_EDGE)

# ── Patch info ────────────────────────────────────────────────────────────────
ax.text(-1.3, -3.5,
        "Patch size: $16^3$ vox  →  $N=(128/16)^3=512$ patches  "
        "→  $16^3\\!\\times\\!4=16384\\!\\to\\!768$  via linear proj.",
        fontsize=7, color="#444")

# ── Legend ────────────────────────────────────────────────────────────────────
lx, ly = -1.3, -2.1
ax.text(lx, ly + 0.1, "Legend:", fontsize=8, fontweight="bold")

ax.add_patch(mpatches.FancyBboxPatch((lx, ly-0.55), 0.3, 0.38,
    boxstyle="square,pad=0", lw=0.7, edgecolor=TR_EDGE, facecolor=TR_FILL_HI))
ax.text(lx+0.45, ly-0.36, "Transformer block  (MSA + MLP, LayerNorm)",
        va="center", fontsize=7.5)

ax.add_patch(mpatches.FancyBboxPatch((lx, ly-1.0), 0.3, 0.38,
    boxstyle="square,pad=0", lw=0.8, edgecolor=DEC_EDGES[1], facecolor=DEC_FILLS[1]))
ax.text(lx+0.45, ly-0.81,
        "Decoder: 2× (Conv $3^3$ + InstanceNorm + ReLU, residual block)",
        va="center", fontsize=7.5)

ax.annotate("", xy=(lx+0.3, ly-1.35), xytext=(lx, ly-1.35),
            arrowprops=dict(arrowstyle="-|>", color=UP_C, lw=1.3, mutation_scale=9))
ax.text(lx+0.45, ly-1.35, "Deconv $2^3$  (upsample ×2) + concat skip",
        va="center", fontsize=7.5)

ax.plot([lx, lx+0.3], [ly-1.75, ly-1.75], color=SKIP_C, lw=1.1, linestyle="dashed")
ax.annotate("", xy=(lx+0.3, ly-1.75), xytext=(lx+0.2, ly-1.75),
            arrowprops=dict(arrowstyle="-|>", color=SKIP_C, lw=1.1, mutation_scale=7))
ax.text(lx+0.45, ly-1.75,
        "Transformer skip  (reshape + Conv $3^3$, InstanceNorm, ReLU)",
        va="center", fontsize=7.5)

ax.plot([lx, lx+0.3], [ly-2.15, ly-2.15], color=IN_SKIP_C, lw=1.0, linestyle="dashed")
ax.annotate("", xy=(lx+0.3, ly-2.15), xytext=(lx+0.2, ly-2.15),
            arrowprops=dict(arrowstyle="-|>", color=IN_SKIP_C, lw=1.0, mutation_scale=7))
ax.text(lx+0.45, ly-2.15, "Input skip  ($z_0$, full resolution)",
        va="center", fontsize=7.5)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(7.5, 7.1,
        "UNETR  (ViT-B/16 Encoder + CNN Decoder) — BraTS 2021",
        ha="center", va="center", fontsize=11, fontweight="bold")

plt.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    plt.savefig(f"unetr_architecture.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved unetr_architecture.{ext}")
