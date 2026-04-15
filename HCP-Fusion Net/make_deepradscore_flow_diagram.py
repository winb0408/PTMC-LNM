import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, xy, w, h, text, fc="#f8fbff", ec="#2c3e50", fontsize=10.5):
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
    out = "../figures/deepradscore_pipeline.png"

    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    w = 0.16
    h = 0.42
    y = 0.38
    xs = [0.02, 0.21, 0.40, 0.59, 0.78]

    box(
        ax,
        (xs[0], y),
        w,
        h,
        "Step 1\nImage-only(B)\nBest model\n(fold k)",
        fc="#eef6ff",
    )
    box(
        ax,
        (xs[1], y),
        w,
        h,
        "Step 2\nExtract fused\nfeature for\nall slices",
        fc="#eefaf3",
    )
    box(
        ax,
        (xs[2], y),
        w,
        h,
        "Step 3\nPatient-level\nmean pooling\n(embedding)",
        fc="#fff8ea",
    )
    box(
        ax,
        (xs[3], y),
        w,
        h,
        "Step 4\nPCA -> 1D\nDeepRadScore\n(fit on train)",
        fc="#fff0f2",
    )
    box(
        ax,
        (xs[4], y),
        w,
        h,
        "Step 5\nConcat with\nclinical feats\n-> LogisticReg",
        fc="#f4f0ff",
    )

    arrow(ax, (xs[0] + w, 0.59), (xs[1], 0.59))
    arrow(ax, (xs[1] + w, 0.59), (xs[2], 0.59))
    arrow(ax, (xs[2] + w, 0.59), (xs[3], 0.59))
    arrow(ax, (xs[3] + w, 0.59), (xs[4], 0.59))

    ax.text(
        0.5,
        0.93,
        "DeepRadScore Generation Pipeline (Patient-level)",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.16,
        "Note: DeepRadScore here is a 1D score compressed from deep image embeddings, "
        "not handcrafted radiomics features.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#333333",
    )

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print(f"saved: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()

