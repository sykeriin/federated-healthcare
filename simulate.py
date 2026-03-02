"""
simulate.py
One-command demo: runs the full federated learning simulation in a
single process using Flower's simulation engine (no separate terminals needed).

This is what you run for the demo video:
    python simulate.py --rounds 20 --clinics 5

Shows:
  - Per-round accuracy for urban vs rural clinics
  - SVD compression ratios per clinic
  - Network simulation (latency, packet loss)
  - Live matplotlib plot updating each round
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import flwr as fl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from flwr.common import ndarrays_to_parameters



from model import HeartDiseaseModel, get_model_parameters
from client import ClinicClient
from data_utils import load_heart_disease_data, partition_data
from server import FedProxStrategy


# ─────────────────────────────────────────────────────────────────────────────
# Shared state for live plotting
# ─────────────────────────────────────────────────────────────────────────────

class TrainingTracker:
    """Collects metrics across all rounds for plotting."""

    def __init__(self):
        self.rounds:        List[int] = []
        self.urban_acc:     List[float] = []
        self.rural_acc:     List[float] = []
        self.compression:   List[float] = []

    def update(self, round_num, urban, rural, comp):
        self.rounds.append(round_num)
        self.urban_acc.append(urban)
        self.rural_acc.append(rural)
        self.compression.append(comp)

    def save(self, path="results/metrics.json"):
        os.makedirs("results", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "rounds":      self.rounds,
                "urban_acc":   self.urban_acc,
                "rural_acc":   self.rural_acc,
                "compression": self.compression,
            }, f, indent=2)
        print(f"\n✅ Metrics saved to {path}")


TRACKER = TrainingTracker()


# ─────────────────────────────────────────────────────────────────────────────
# Client factory for Flower simulation
# ─────────────────────────────────────────────────────────────────────────────

def make_client_factory(train_datasets, val_datasets, rank_ratio):

    def client_fn(cid: str):
        clinic_id = int(cid)
        X_train, y_train = train_datasets[clinic_id]
        X_val,   y_val   = val_datasets[clinic_id]

        return ClinicClient(
            clinic_id=clinic_id,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            rank_ratio=rank_ratio,
        )

    return client_fn

# ─────────────────────────────────────────────────────────────────────────────
# Instrumented strategy that feeds TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class TrackedStrategy(FedProxStrategy):

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        urban = metrics.get("urban_accuracy", 0.0)
        rural = metrics.get("rural_accuracy", 0.0)
        comp = 0.0
        if self.round_history:
            comp = self.round_history[-1].get("avg_compression", 0.0)

        TRACKER.update(server_round, urban, rural, comp)
        return loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Live plot
# ─────────────────────────────────────────────────────────────────────────────

def launch_live_plot(num_rounds: int):
    """Launch a live matplotlib chart that updates after each round."""
    matplotlib.use("Qt5Agg")  # change to "Qt5Agg" if TkAgg not available

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle("AMD Rural Healthcare FL — Live Training Dashboard",
                 fontsize=14, fontweight="bold", color="#E8550A")
    fig.patch.set_facecolor("#0D0D0D")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1A1A1A")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444444")
        ax.spines["left"].set_color("#444444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.set_xlim(0, num_rounds + 1)
    ax1.set_ylim(50, 100)
    ax1.set_ylabel("Accuracy (%)", color="white")
    ax1.set_xlabel("Training Round", color="white")
    ax1.set_title("Model Accuracy by Population", color="white")
    ax1.axhline(y=65, color="#E74C3C", linestyle="--", alpha=0.5,
                label="Rural no-FL baseline ~65% (majority class)")
    ax1.legend(loc="lower right", facecolor="#1A1A1A", labelcolor="white")

    ax2.set_xlim(0, num_rounds + 1)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Compression (%)", color="white")
    ax2.set_xlabel("Training Round", color="white")
    ax2.set_title("SVD Compression Ratio per Round", color="white")

    urban_line, = ax1.plot([], [], color="#3399FF", linewidth=2.5,
                           marker="o", markersize=5, label="🏙️  Urban")
    rural_line, = ax1.plot([], [], color="#E8550A", linewidth=2.5,
                           marker="s", markersize=5, label="🏥 Rural (our FL)")
    ax1.legend(loc="lower right", facecolor="#1A1A1A", labelcolor="white")

    comp_bar = ax2.bar([], [], color="#E8550A", alpha=0.7, width=0.6)

    def update(_frame):
        if not TRACKER.rounds:
            return

        x = TRACKER.rounds
        urban_line.set_data(x, TRACKER.urban_acc)
        rural_line.set_data(x, TRACKER.rural_acc)

        # Redraw compression bars
        ax2.cla()
        ax2.set_facecolor("#1A1A1A")
        ax2.set_xlim(0, num_rounds + 1)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Compression (%)", color="white")
        ax2.set_xlabel("Training Round", color="white")
        ax2.set_title("SVD Compression Ratio per Round", color="white")
        ax2.tick_params(colors="white")
        ax2.spines["bottom"].set_color("#444444")
        ax2.spines["left"].set_color("#444444")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.bar(x, TRACKER.compression, color="#E8550A", alpha=0.7, width=0.6)

        # Gap annotation
        if len(x) > 0:
            last_urban = TRACKER.urban_acc[-1]
            last_rural = TRACKER.rural_acc[-1]
            ax1.set_title(
                f"Model Accuracy — Round {x[-1]} | "
                f"Gap: {last_urban - last_rural:.1f}%",
                color="white"
            )

        fig.tight_layout()

    ani = animation.FuncAnimation(
        fig, update, interval=2000, cache_frame_data=False)
    plt.tight_layout()
    plt.show(block=False)
    return ani


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run full federated learning simulation"
    )
    parser.add_argument("--rounds",      type=int,   default=20)
    parser.add_argument("--clinics",     type=int,   default=5)
    parser.add_argument("--no-dp",       action="store_true")
    parser.add_argument("--rank-ratio",  type=float, default=0.1)
    parser.add_argument("--no-plot",     action="store_true",
                        help="Disable live matplotlib plot")
    parser.add_argument("--mu",          type=float, default=0.01,
                        help="FedProx mu (proximal term)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  🏥 AMD Hardware-Aware Federated Learning")
    print(f"  Built for Rural Healthcare — Team alphago")
    print(f"{'='*60}")
    print(f"  Rounds:       {args.rounds}")
    print(f"  Clinics:      {args.clinics} (1 urban + {args.clinics-1} rural)")
    print(
        f"  DP:           {'Disabled' if args.no_dp else 'Enabled (Opacus)'}")
    print(
        f"  SVD ratio:    {args.rank_ratio} ({(1-args.rank_ratio)*100:.0f}% reduction)")
    print(f"  FedProx mu:   {args.mu}")
    print(f"{'='*60}\n")

    # ── Load & partition data ──
    X, y = load_heart_disease_data()
    train_datasets, val_datasets = partition_data(
        X, y, num_clinics=args.clinics
    )

    # ── Initial model parameters ──
    init_model = HeartDiseaseModel(input_dim=train_datasets[0][0].shape[1])
    init_params = ndarrays_to_parameters(get_model_parameters(init_model))

    # ── Strategy ──
    strategy = TrackedStrategy(
        mu=args.mu,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.clinics,
        min_evaluate_clients=args.clinics,
        min_available_clients=args.clinics,
        initial_parameters=init_params,
        on_fit_config_fn=lambda rnd: {"round": rnd},
    )

    # ── Live plot ──
    ani = None
    if not args.no_plot:
        try:
            ani = launch_live_plot(args.rounds)
        except Exception as e:
            print(
                f"⚠️  Live plot unavailable ({e}). Use --no-plot to suppress.")

    # ── Client factory ──
    client_fn = make_client_factory(
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        rank_ratio=args.rank_ratio,
    )

    # ── Run simulation ──
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.clinics,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    # ── Save results ──
    TRACKER.save()
    os.makedirs("results", exist_ok=True)
    with open("results/round_history.json", "w") as f:
        json.dump(strategy.round_history, f, indent=2)

    # ── Final summary ──
    print(f"\n{'='*60}")
    print("  📊 FINAL RESULTS")
    print(f"{'='*60}")
    if TRACKER.rounds:
        print(f"  🏙️  Urban accuracy:  {TRACKER.urban_acc[-1]:.2f}%")
        print(f"  🏥  Rural accuracy:  {TRACKER.rural_acc[-1]:.2f}%")
        print(
            f"  Accuracy gap:       {TRACKER.urban_acc[-1] - TRACKER.rural_acc[-1]:.2f}%")
        print(f"  Avg compression:    {np.mean(TRACKER.compression):.1f}%")
    print(f"{'='*60}")
    print("  Results saved to results/")
    print("  Run: python plot_results.py  to generate final charts")

    if ani is not None:
        plt.show()  # Keep plot open


if __name__ == "__main__":
    main()
