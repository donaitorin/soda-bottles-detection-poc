"""Draw the Stage 5 training curves as a hand-made SVG (no plotting library): training loss per epoch (left) and
validation agreement per epoch (right). Numbers only, no photos, so the file can be committed.

THE VALIDATION LINES ARE AGREEMENT WITH GROUNDING DINO PSEUDO-LABELS, NOT ACCURACY.

Usage: python scripts/plot_training.py [--csv data/logs/stage5/frcnn_v1/train_log.csv] [--out reports/figures/stage_05_training.svg]
"""
import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import check_output_dir  # noqa: E402

W, H = 980, 430
PANEL = {"left": (60, 40, 440, 330), "right": (550, 40, 930, 330)}  # x0, y0, x1, y1 of the plot areas
COLORS = {"loss": "#1f6feb", "AP50": "#d1242f", "AP": "#8250df", "recall": "#1a7f37", "ref": "#6e7781"}


def scale(v, lo, hi, a, b):
    return a + (v - lo) / (hi - lo) * (b - a)


def axes(parts, box, xlim, ylim, xlabel, ylabel, yticks, fmt):
    x0, y0, x1, y1 = box
    parts.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="none" stroke="#8c959f"/>')
    for t in yticks:
        y = scale(t, ylim[0], ylim[1], y1, y0)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#d0d7de" stroke-width="0.7"/>')
        parts.append(f'<text x="{x0 - 6}" y="{y + 4:.1f}" font-size="11" text-anchor="end">{fmt(t)}</text>')
    for e in range(0, xlim[1] + 1, 5):
        x = scale(e, xlim[0], xlim[1], x0, x1)
        parts.append(f'<text x="{x:.1f}" y="{y1 + 16}" font-size="11" text-anchor="middle">{e}</text>')
    parts.append(f'<text x="{(x0 + x1) / 2}" y="{y1 + 34}" font-size="12" text-anchor="middle">{xlabel}</text>')
    title, _, subtitle = ylabel.partition("|")
    parts.append(f'<text x="{x0}" y="{y0 - 22}" font-size="13" font-weight="bold">{title}</text>')
    if subtitle:
        parts.append(f'<text x="{x0}" y="{y0 - 8}" font-size="11" fill="#cf222e">{subtitle}</text>')


def line(parts, box, xlim, ylim, xs, ys, color, width=2, dash=None):
    x0, y0, x1, y1 = box
    pts = " ".join(f"{scale(x, xlim[0], xlim[1], x0, x1):.1f},{scale(y, ylim[0], ylim[1], y1, y0):.1f}" for x, y in zip(xs, ys))
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"{d}/>')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/logs/stage5/frcnn_v1/train_log.csv")
    parser.add_argument("--out", default="reports/figures/stage_05_training.svg")
    parser.add_argument("--run-label", default="Stage 5 training curves", help="title of the figure")
    parser.add_argument("--baseline-ap50", type=float, default=0.884, help="Stage 4 Faster R-CNN COCO + filters agreement, drawn as a reference line")
    args = parser.parse_args()
    rows = list(csv.DictReader((REPO_ROOT / args.csv).open()))
    epochs = [int(r["epoch"]) for r in rows]
    n = max(epochs)
    best = [int(r["epoch"]) for r in rows if r["is_best"] == "1"][-1]
    out = REPO_ROOT / args.out
    check_output_dir(out.parent, [REPO_ROOT / "data/raw", REPO_ROOT / "data/test_images"])
    out.parent.mkdir(parents=True, exist_ok=True)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="sans-serif">',
             f"<title>{args.run_label}: training loss and validation agreement with Grounding DINO per epoch</title>",
             f'<rect width="{W}" height="{H}" fill="white"/>']
    # Left: training loss
    loss = [float(r["train_loss"]) for r in rows]
    lbox, lx, ly = PANEL["left"], (0, n), (0, 1.3)
    axes(parts, lbox, lx, ly, "epoch", "Training loss (mean per epoch)", [0, 0.25, 0.5, 0.75, 1.0, 1.25], lambda t: f"{t:g}")
    line(parts, lbox, lx, ly, epochs, loss, COLORS["loss"])
    # Right: validation agreement
    rbox, rx, ry = PANEL["right"], (0, n), (0.70, 1.0)
    axes(parts, rbox, rx, ry, "epoch", "Validation agreement with Grounding DINO|pseudo-labels: agreement, NOT accuracy", [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], lambda t: f"{t:.2f}")
    for key, label, color in (("val_recall50", "recall at IoU 0.5 (all scores)", COLORS["recall"]), ("val_AP50", "AP@0.5", COLORS["AP50"]),
                              ("val_AP", "AP@0.5:0.95", COLORS["AP"])):
        line(parts, rbox, rx, ry, epochs, [float(r[key]) for r in rows], color)
    line(parts, rbox, rx, ry, [0, n], [args.baseline_ap50] * 2, COLORS["ref"], 1.5, "6 4")
    xb = scale(best, rx[0], rx[1], rbox[0], rbox[2])
    parts.append(f'<line x1="{xb:.1f}" y1="{rbox[1]}" x2="{xb:.1f}" y2="{rbox[3]}" stroke="#bf8700" stroke-width="1.2" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{xb + 4:.1f}" y="{rbox[1] + 14}" font-size="11" fill="#bf8700">best.pt = epoch {best}</text>')
    # Legend under the right panel
    ly0 = rbox[3] + 52
    for i, (label, color, dash) in enumerate((("recall at IoU 0.5 (all scores)", COLORS["recall"], None), ("AP@0.5", COLORS["AP50"], None),
                                              ("AP@0.5:0.95", COLORS["AP"], None), ("Stage 4 Faster R-CNN COCO + filters, AP@0.5", COLORS["ref"], "6 4"))):
        x = rbox[0] + (i % 2) * 190
        y = ly0 + (i // 2) * 18
        d = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 22}" y2="{y}" stroke="{color}" stroke-width="2"{d}/>')
        parts.append(f'<text x="{x + 28}" y="{y + 4}" font-size="11">{label}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts))
    print(f"wrote {out.relative_to(REPO_ROOT)} ({len(rows)} epochs, best epoch {best})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
