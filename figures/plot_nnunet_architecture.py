#!/usr/bin/env python3
"""
nnU-Net (PlainConvUNet, 3d_fullres) architecture figure — BraTS 2021.
From nnUNetPlans.json:
  6 encoder stages: 32->64->128->256->320->320 ch, 128³->4³ spatial
  5 decoder stages (symmetric)
  2x Conv3³ + InstanceNorm + LeakyReLU per stage
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(15, 6.5))
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-1.2, 16.5)
ax.set_ylim(-2.8, 5.5)

# ── Palette ───────────────────────────────────────────────────────────────────
ENC = ["#D0D0FF", "#AAAAFF", "#8888EE", "#6666CC", "#4444AA", "#222288"]
DEC = ["#FFE0B0", "#FFCC80", "#FFBB55", "#FFAA33", "#FF9911"]
SKIP_C = "#888888"
POOL_C = "#CC2222"
UP_C   = "#1A9922"
OUT_C  = "#AADDAA"

BW = 0.18   # block width
BG = 0.06   # gap between the two conv blocks of one stage
SP = 1.30   # spacing between stages

# Heights proportional to spatial resolution
H = [5.0, 3.9, 3.0, 2.3, 1.7, 1.2]   # enc stages 1–6
CH_ENC = [32, 64, 128, 256, 320, 320]
RES    = ["128³", "64³", "32³", "16³", "8³", "4³"]
CH_DEC = [320, 256, 128, 64, 32]       # decoder stages 1–5

# ── Draw a stage (2 rectangles) ───────────────────────────────────────────────
def stage(x, height, color, edge):
    y = -height / 2
    for dx in [0, BW + BG]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x + dx, y), BW, height,
            boxstyle="square,pad=0", lw=0.8,
            edgecolor=edge, facecolor=color))
    return x + BW + BG / 2   # centre x

# ── Encoder ───────────────────────────────────────────────────────────────────
EX = [i * SP for i in range(6)]
ecx = [stage(EX[i], H[i], ENC[i], "#111166") for i in range(6)]

# ── Decoder ───────────────────────────────────────────────────────────────────
DX = [EX[5] + (j + 1) * SP for j in range(5)]
dec_h = [H[4], H[3], H[2], H[1], H[0]]   # mirrors encoder 5→1
dcx = [stage(DX[j], dec_h[j], DEC[j], "#884400") for j in range(5)]

# ── Output 1×1×1 conv ─────────────────────────────────────────────────────────
OX = DX[4] + SP
out_h = H[0]
ax.add_patch(mpatches.FancyBboxPatch(
    (OX, -out_h/2), BW * 0.7, out_h,
    boxstyle="square,pad=0", lw=0.8,
    edgecolor="#226622", facecolor=OUT_C))
ocx = OX + BW * 0.35

# ── Pooling arrows (encoder) ──────────────────────────────────────────────────
for i in range(5):
    ax.annotate("", xy=(EX[i+1], -H[i+1]/2),
                xytext=(EX[i] + 2*BW + BG, -H[i]/2),
                arrowprops=dict(arrowstyle="-|>", color=POOL_C, lw=1.3,
                                mutation_scale=9, connectionstyle="arc3,rad=0"))

# ── Upsample arrows (decoder) ─────────────────────────────────────────────────
# bottleneck → first decoder stage
ax.annotate("", xy=(DX[0], -dec_h[0]/2),
            xytext=(EX[5] + 2*BW + BG, -H[5]/2),
            arrowprops=dict(arrowstyle="-|>", color=UP_C, lw=1.3,
                            mutation_scale=9, connectionstyle="arc3,rad=0"))
for j in range(4):
    ax.annotate("", xy=(DX[j+1], -dec_h[j+1]/2),
                xytext=(DX[j] + 2*BW + BG, -dec_h[j]/2),
                arrowprops=dict(arrowstyle="-|>", color=UP_C, lw=1.3,
                                mutation_scale=9, connectionstyle="arc3,rad=0"))

# ── Skip connections (FancyArrowPatch, arcs above figure) ─────────────────────
# Pairs (enc_i, dec_j) where dec_j mirrors enc_i:
# enc0↔dec4, enc1↔dec3, enc2↔dec2, enc3↔dec1, enc4↔dec0
# Both ends are at same y = H[i]/2, arcs nest naturally with fixed rad
skip_pairs = [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
for ei, dj in skip_pairs:
    y = H[ei] / 2
    p = FancyArrowPatch(
        posA=(ecx[ei], y), posB=(dcx[dj], y),
        arrowstyle="-|>", mutation_scale=8,
        color=SKIP_C, lw=0.9, linestyle="dashed",
        connectionstyle="arc3,rad=-0.28")
    ax.add_patch(p)

# ── Arrow to output ───────────────────────────────────────────────────────────
ax.annotate("", xy=(OX, 0), xytext=(DX[4] + 2*BW + BG, 0),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.1,
                            mutation_scale=8))

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL_Y  = 3.0    # fixed y for resolution labels (above all blocks)
LABEL_Y2 = 3.5    # channel labels above resolution labels

for i in range(6):
    ax.text(ecx[i], LABEL_Y,  RES[i],     ha="center", fontsize=7.5, color="#222222")
    ax.text(ecx[i], LABEL_Y2, f"{CH_ENC[i]}ch", ha="center", fontsize=7, color="#444488")

# Decoder: same res/ch labels
for j in range(5):
    ei_mirror = 4 - j   # which encoder stage this mirrors
    ax.text(dcx[j], LABEL_Y,  RES[ei_mirror], ha="center", fontsize=7.5, color="#222222")
    ax.text(dcx[j], LABEL_Y2, f"{CH_DEC[j]}ch",  ha="center", fontsize=7, color="#884400")

ax.text(ocx, LABEL_Y,  "128³", ha="center", fontsize=7.5, color="#222222")
ax.text(ocx, LABEL_Y2, "4ch",  ha="center", fontsize=7, color="#226622")

# Bottleneck label below
ax.text(ecx[5], -H[5]/2 - 0.18, "Bottleneck", ha="center", fontsize=7,
        color="#222288", style="italic")

# Input / Output
ax.text(EX[0] - 0.45, 0, "Input\n$128^3\\!\\times\\!4$",
        ha="center", va="center", fontsize=8.5)
ax.annotate("", xy=(EX[0], 0), xytext=(EX[0]-0.22, 0),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.1, mutation_scale=8))

ax.text(ocx + 0.45, 0, "Output\n$128^3\\!\\times\\!4$",
        ha="center", va="center", fontsize=8.5)
ax.annotate("", xy=(ocx + 0.22, 0), xytext=(ocx, 0),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.1, mutation_scale=8))

# Encoder / Decoder section labels
ax.text((ecx[0]+ecx[4])/2, -H[2]/2 - 0.45, "Encoder",
        ha="center", fontsize=9, style="italic", color="#222266",
        fontweight="bold")
ax.text((dcx[0]+dcx[4])/2, -H[2]/2 - 0.45, "Decoder",
        ha="center", fontsize=9, style="italic", color="#774400",
        fontweight="bold")

# ── Legend ────────────────────────────────────────────────────────────────────
lx, ly = 0.0, -1.8
ax.text(lx, ly + 0.1, "Legend:", fontsize=8, fontweight="bold")

ax.add_patch(mpatches.FancyBboxPatch((lx, ly-0.55), 0.3, 0.38,
    boxstyle="square,pad=0", lw=0.8, edgecolor="#111166", facecolor=ENC[2]))
ax.text(lx+0.45, ly-0.36,
        "2× (Conv $3^3$, InstanceNorm, LeakyReLU)  per stage",
        va="center", fontsize=7.5)

ax.annotate("", xy=(lx+0.3, ly-0.95), xytext=(lx, ly-0.95),
            arrowprops=dict(arrowstyle="-|>", color=POOL_C, lw=1.3, mutation_scale=9))
ax.text(lx+0.45, ly-0.95, "Stride-2 Conv $2^3$  (encoder, ÷2 per axis)", va="center", fontsize=7.5)

ax.annotate("", xy=(lx+0.3, ly-1.38), xytext=(lx, ly-1.38),
            arrowprops=dict(arrowstyle="-|>", color=UP_C, lw=1.3, mutation_scale=9))
ax.text(lx+0.45, ly-1.38, "Transposed Conv $2^3$  (decoder, ×2 per axis)", va="center", fontsize=7.5)

ax.plot([lx, lx+0.3], [ly-1.8, ly-1.8], color=SKIP_C, lw=0.9, linestyle="dashed")
ax.annotate("", xy=(lx+0.3, ly-1.8), xytext=(lx+0.2, ly-1.8),
            arrowprops=dict(arrowstyle="-|>", color=SKIP_C, lw=0.9, mutation_scale=7))
ax.text(lx+0.45, ly-1.8, "Skip connection  (concatenation before decoder stage)",
        va="center", fontsize=7.5)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(7.5, 5.1,
        "nnU-Net  (PlainConvUNet, 3D full-resolution) — BraTS 2021",
        ha="center", va="center", fontsize=11, fontweight="bold")

plt.tight_layout(pad=0.2)
for ext in ("pdf", "png"):
    plt.savefig(f"nnunet_architecture.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved nnunet_architecture.{ext}")
