"""
Generate a modalities comparison figure from the BraTS modalities strip.

Usage:
    python scripts/generate_modalities_figure.py           # default: 2x2 grid
    python scripts/generate_modalities_figure.py --layout row   # single row
    python scripts/generate_modalities_figure.py --layout grid  # 2x2 grid (default)

Options:
    --src      Path to source modalities strip image (default: samples/modalities.png)
    --out      Output file path (default: samples/modalities_2x2.png or samples/modalities_row.png)
    --gap      Gap in pixels between panels (default: 6)
    --label-height  Pixels to reserve for label text at top of each panel (default: 30)
    --panels   Number of panels to extract from the strip (default: 4)
    --layout   Output layout: 'grid' (2x2) or 'row' (1xN) (default: grid)
"""

import argparse
import os
from PIL import Image


LABELS = ["FLAIR", "T1", "T1CE", "T2"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MRI modalities figure.")
    parser.add_argument("--src", default="samples/modalities.png",
                        help="Source modalities strip image.")
    parser.add_argument("--out", default=None,
                        help="Output image path. Defaults depend on --layout.")
    parser.add_argument("--gap", type=int, default=6,
                        help="Gap in pixels between panels.")
    parser.add_argument("--label-height", type=int, default=45,
                        help="Pixels to crop from top of each panel (label area).")
    parser.add_argument("--panels", type=int, default=4,
                        help="Number of panels to extract from the strip.")
    parser.add_argument("--layout", choices=["grid", "row"], default="grid",
                        help="Output layout: 'grid' (2x2) or 'row' (1xN).")
    parser.add_argument("--order", default="0,1,2,3",
                        help="Comma-separated panel indices controlling output order. "
                             "Strip order is 0=T1,1=T1CE,2=T2,3=FLAIR. "
                             "Default '0,1,2,3' gives T1, T1CE, T2, FLAIR.")
    return parser.parse_args()


def extract_panels(src_path: str, n_panels: int, label_height: int) -> list[Image.Image]:
    """Crop the first n_panels from the horizontal strip and remove the label row."""
    src = Image.open(src_path)
    w, h = src.size
    panel_w = w // n_panels
    panels = []
    for i in range(n_panels):
        crop = src.crop((i * panel_w, label_height, (i + 1) * panel_w, h))
        panels.append(crop)
    return panels


def build_grid(panels: list[Image.Image], gap: int) -> Image.Image:
    """Arrange panels in a 2x2 grid."""
    if len(panels) != 4:
        raise ValueError(f"Grid layout requires exactly 4 panels, got {len(panels)}.")
    cell_w, cell_h = panels[0].size
    grid_w = cell_w * 2 + gap
    grid_h = cell_h * 2 + gap
    grid = Image.new("RGB", (grid_w, grid_h), color=(220, 220, 220))
    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for panel, (col, row) in zip(panels, positions):
        x = col * (cell_w + gap)
        y = row * (cell_h + gap)
        grid.paste(panel, (x, y))
    return grid


def build_row(panels: list[Image.Image], gap: int) -> Image.Image:
    """Arrange panels in a single horizontal row."""
    cell_w, cell_h = panels[0].size
    n = len(panels)
    row_w = cell_w * n + gap * (n - 1)
    row_h = cell_h
    row_img = Image.new("RGB", (row_w, row_h), color=(220, 220, 220))
    for i, panel in enumerate(panels):
        x = i * (cell_w + gap)
        row_img.paste(panel, (x, 0))
    return row_img


def main():
    args = parse_args()

    # Resolve working directory to repo root (allows running from anywhere)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    src_path = os.path.join(repo_root, args.src) if not os.path.isabs(args.src) else args.src

    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Source image not found: {src_path}")

    panels = extract_panels(src_path, args.panels, args.label_height)
    print(f"Extracted {len(panels)} panels from {src_path} "
          f"(each {panels[0].size[0]}x{panels[0].size[1]} px after label crop)")

    order = [int(i) for i in args.order.split(",")]
    if any(i >= len(panels) for i in order):
        raise ValueError(f"--order index out of range for {len(panels)} panels: {args.order}")
    panels = [panels[i] for i in order]
    print(f"Panel order: {[['FLAIR','T1','T1CE','T2'][i] for i in order]}")

    if args.layout == "grid":
        result = build_grid(panels, args.gap)
        default_out = "samples/modalities_2x2.png"
    else:
        result = build_row(panels, args.gap)
        default_out = "samples/modalities_row.png"

    out_rel = args.out if args.out is not None else default_out
    out_path = os.path.join(repo_root, out_rel) if not os.path.isabs(out_rel) else out_rel

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.save(out_path, dpi=(300, 300))
    print(f"Saved {args.layout} figure: {out_path} ({result.size[0]}x{result.size[1]} px)")


if __name__ == "__main__":
    main()
