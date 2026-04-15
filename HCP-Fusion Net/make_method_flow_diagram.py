import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, xy, w, h, text, fc="#f8fbff", ec="#2c3e50", fontsize=11):
    x, y = xy
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, p1, p2):
    arr = FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=13, linewidth=1.4, color="#34495e")
    ax.add_patch(arr)


def main():
    os.makedirs("../figures", exist_ok=True)
    out = "../figures/method_flow_diagram.png"

    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.03, 0.63), 0.20, 0.25, "Data & Split\nMultiModalDataset\nPatient-level 5-fold")
    box(ax, (0.28, 0.63), 0.20, 0.25, "Base Training\nCentralLNM (single task)\nTrain/Val/Test")
    box(ax, (0.53, 0.63), 0.20, 0.25, "Issue Detection\nHigh Train, Low Val/Test\nGeneralization Gap")
    box(ax, (0.78, 0.63), 0.19, 0.25, "Three-route Comparison\nClinical-only\nImage-only (B)\nClinical+DeepRad")

    box(ax, (0.14, 0.18), 0.24, 0.28, "Result Statistics\nAUC Tables\nROC Comparison\nBoxplots")
    box(ax, (0.46, 0.18), 0.24, 0.28, "Core Findings\nClinical signal dominates\nImage gain unstable")
    box(ax, (0.76, 0.18), 0.20, 0.28, "Next Steps\nFusion gating/weighting\nPatient-level MIL")

    arrow(ax, (0.23, 0.755), (0.28, 0.755))
    arrow(ax, (0.48, 0.755), (0.53, 0.755))
    arrow(ax, (0.73, 0.755), (0.78, 0.755))

    arrow(ax, (0.88, 0.63), (0.86, 0.46))
    arrow(ax, (0.30, 0.63), (0.26, 0.46))

    arrow(ax, (0.38, 0.32), (0.46, 0.32))
    arrow(ax, (0.70, 0.32), (0.76, 0.32))

    ax.text(0.5, 0.95, "CentralLNM Performance Debugging Workflow", ha="center", va="center", fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print(f"saved: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()

