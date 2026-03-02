"""
plot_results.py
Generate publication-quality charts from saved training results.
Produces the fairness convergence graph for the deck.

Run after simulate.py:
    python plot_results.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_PATH = "results/metrics.json"
OUTPUT_DIR   = "results"


def load_results(path: str) -> dict:
    if not os.path.exists(path):
        print(f"❌ Results file not found: {path}")
        print("   Run: python simulate.py first")
        exit(1)
    with open(path) as f:
        return json.load(f)


def plot_accuracy_convergence(data: dict):
    """Fairness chart: rural vs urban accuracy across rounds."""
    rounds    = data["rounds"]
    urban_acc = data["urban_acc"]
    rural_acc = data["rural_acc"]

    # Rural baseline (no FL) — flat reference line
    baseline = [65.0] * len(rounds)  # majority-class accuracy on 75%-positive rural val

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#111111")

    # Grid
    ax.grid(axis="y", color="#2A2A2A", linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

    # Baseline
    ax.plot(rounds, baseline, color="#E74C3C", linewidth=1.5,
            linestyle="--", alpha=0.7, label="Rural — No FL baseline (majority class ~65%)")

    # Urban line
    ax.plot(rounds, urban_acc, color="#3399FF", linewidth=2.5,
            marker="o", markersize=6, label="🏙️  Urban clinics")

    # Rural with FL
    ax.plot(rounds, rural_acc, color="#E8550A", linewidth=2.5,
            marker="s", markersize=6, label="🏥 Rural — With Our FL Framework")

    # Shaded convergence zone
    ax.fill_between(rounds, rural_acc, urban_acc,
                    alpha=0.1, color="#E8550A", label="Accuracy gap (shrinking)")

    # Annotations
    if len(rounds) >= 1:
        improvement = rural_acc[-1] - rural_acc[0]
        ax.annotate(
            f"  Rural +{improvement:.1f}% improvement",
            xy=(rounds[-1], rural_acc[-1]),
            xytext=(rounds[-1] - 4, rural_acc[-1] - 4),
            fontsize=10, color="#E8550A", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#E8550A", lw=1.5)
        )
        ax.annotate(
            f"  Final gap: {abs(urban_acc[-1]-rural_acc[-1]):.1f}%",
            xy=(rounds[-1], (urban_acc[-1] + rural_acc[-1]) / 2),
            fontsize=9, color="#AAAAAA"
        )

    ax.set_xlim(min(rounds) - 0.5, max(rounds) + 0.5)
    ax.set_ylim(50, 102)
    ax.set_xlabel("Training Rounds", color="white", fontsize=12)
    ax.set_ylabel("Accuracy (%)", color="white", fontsize=12)
    ax.set_title(
        "Model Accuracy by Population — Federated Rounds\n"
        "Rural clinics converge toward urban performance as federation trains",
        color="white", fontsize=13, fontweight="bold"
    )

    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#333333")

    legend = ax.legend(
        loc="lower right", facecolor="#1A1A1A", edgecolor="#444444",
        labelcolor="white", fontsize=10
    )

    # AMD branding
    fig.text(0.98, 0.02, "Team alphago | AMD Slingshot Hackathon",
             color="#666666", fontsize=8, ha="right")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "accuracy_convergence.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅ Saved: {out}")
    plt.show()


def plot_compression_ratios(data: dict):
    """Bar chart showing SVD compression ratio per round."""
    rounds = data["rounds"]
    comp   = data["compression"]

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#111111")
    ax.grid(axis="y", color="#2A2A2A", linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

    bars = ax.bar(rounds, comp, color="#E8550A", alpha=0.85, width=0.6,
                  edgecolor="#FF6B00", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, comp):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom",
                color="white", fontsize=8)

    # Reference line
    avg = np.mean(comp)
    ax.axhline(y=avg, color="#3399FF", linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Average: {avg:.1f}%")

    ax.set_xlim(min(rounds) - 0.5, max(rounds) + 0.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Training Round", color="white", fontsize=12)
    ax.set_ylabel("Payload Reduction (%)", color="white", fontsize=12)
    ax.set_title("SVD Compression Ratio per Round",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.legend(facecolor="#1A1A1A", edgecolor="#444444",
              labelcolor="white", fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "compression_ratios.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅ Saved: {out}")
    plt.show()


def print_summary(data: dict):
    rounds    = data["rounds"]
    urban_acc = data["urban_acc"]
    rural_acc = data["rural_acc"]
    comp      = data["compression"]

    print(f"\n{'='*55}")
    print(f"  📊 TRAINING SUMMARY")
    print(f"{'='*55}")
    print(f"  Rounds completed:     {len(rounds)}")
    print(f"  Final urban accuracy: {urban_acc[-1]:.2f}%")
    print(f"  Final rural accuracy: {rural_acc[-1]:.2f}%")
    print(f"  Final accuracy gap:   {urban_acc[-1]-rural_acc[-1]:.2f}%")
    print(f"  Rural improvement:    {rural_acc[-1]-rural_acc[0]:+.2f}% over training")
    print(f"  Avg compression:      {np.mean(comp):.1f}%")
    print(f"  Peak compression:     {max(comp):.1f}%")
    print(f"{'='*55}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_results(RESULTS_PATH)
    print_summary(data)
    plot_accuracy_convergence(data)
    plot_compression_ratios(data)
