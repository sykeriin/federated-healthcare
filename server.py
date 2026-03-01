"""
server.py
Federated Learning aggregation server.

Uses FedProx strategy (falls back to FedAvg) for fair aggregation
across heterogeneous clinic hardware. Logs per-round accuracy for
both urban and rural clinic populations.

Run:
    python server.py --rounds 20 --min-clients 3
"""

import argparse
from typing import Dict, List, Optional, Tuple, Union
from functools import reduce

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
)
from flwr.server.client_proxy import ClientProxy


# ─────────────────────────────────────────────────────────────────────────────
# FedProx-style aggregation strategy
# ─────────────────────────────────────────────────────────────────────────────

class FedProxStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg extended with:
    - Proximal term weighting (FedProx) to handle heterogeneous devices
    - Per-round logging of rural vs urban accuracy
    - Compression ratio tracking
    """

    def __init__(self, mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu  # proximal term coefficient

        # History for plotting
        self.round_history: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        if not results:
            return None, {}

        # Extract weights and metrics
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Weighted average (FedAvg base)
        aggregated = self._weighted_average(weights_results)

        # Log per-round info
        urban_acc = []
        rural_acc = []
        comp_ratios = []

        for _, fit_res in results:
            metrics = fit_res.metrics
            if metrics:
                is_rural = bool(metrics.get("is_rural", 0))
                comp_pct = metrics.get("compression_pct", 0.0)
                comp_ratios.append(comp_pct)

        avg_compression = np.mean(comp_ratios) if comp_ratios else 0.0

        print(f"\n[Server] ✅ Round {server_round} aggregated "
              f"{len(results)} clients | "
              f"Avg compression: {avg_compression:.1f}%")

        self.round_history.append({
            "round":           server_round,
            "n_clients":       len(results),
            "avg_compression": avg_compression,
        })

        return ndarrays_to_parameters(aggregated), {}

    def _weighted_average(
        self, results: List[Tuple[List[np.ndarray], int]]
    ) -> List[np.ndarray]:
        """Compute weighted average of model updates."""
        total_examples = sum(n for _, n in results)

        weighted = [
            [layer * (n / total_examples) for layer in weights]
            for weights, n in results
        ]

        return [
            reduce(np.add, [client_w[i] for client_w in weighted])
            for i in range(len(weighted[0]))
        ]

    def aggregate_evaluate(self, server_round, results, failures):
        """Log per-round accuracy split by rural vs urban clinics."""
        if not results:
            return None, {}

        urban_accs = []
        rural_accs = []

        for _, eval_res in results:
            metrics = eval_res.metrics
            if metrics:
                acc = metrics.get("accuracy", 0.0)
                is_rural = bool(metrics.get("is_rural", 0))
                if is_rural:
                    rural_accs.append(acc)
                else:
                    urban_accs.append(acc)

        urban_mean = np.mean(urban_accs) * 100 if urban_accs else 0.0
        rural_mean = np.mean(rural_accs) * 100 if rural_accs else 0.0
        gap = urban_mean - rural_mean

        print(f"\n[Server] 📊 Round {server_round} Accuracy:")
        print(f"  🏙️  Urban clinics:  {urban_mean:.2f}%")
        print(f"  🏥  Rural clinics:  {rural_mean:.2f}%")
        print(f"  Gap:               {gap:.2f}%")

        # Update history
        if self.round_history and self.round_history[-1]["round"] == server_round:
            self.round_history[-1].update({
                "urban_accuracy": urban_mean,
                "rural_accuracy": rural_mean,
                "accuracy_gap":   gap,
            })

        # Aggregate loss for Flower (modern Flower API)
        total_examples = sum(
            eval_res.num_examples for _, eval_res in results
        )

        weighted_losses = [
            eval_res.loss * eval_res.num_examples
            for _, eval_res in results
            if eval_res.loss is not None
        ]

        avg_loss = (
            sum(weighted_losses) / total_examples
            if weighted_losses and total_examples > 0
            else 0.0
        )

        return avg_loss, {
            "urban_accuracy": urban_mean,
            "rural_accuracy": rural_mean,
            "accuracy_gap":   gap,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Server entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Start the FL aggregation server")
    parser.add_argument("--rounds",      type=int,   default=20,
                        help="Number of federated training rounds")
    parser.add_argument("--min-clients", type=int,   default=2,
                        help="Minimum clients required per round")
    parser.add_argument("--fraction",    type=float, default=1.0,
                        help="Fraction of clients to sample per round (0–1)")
    parser.add_argument("--port",        type=int,   default=8080,
                        help="Port to listen on")
    parser.add_argument("--mu",          type=float, default=0.01,
                        help="FedProx proximal term coefficient")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  🚀 AMD Rural Healthcare FL Server")
    print(f"{'='*55}")
    print(f"  Rounds:       {args.rounds}")
    print(f"  Min clients:  {args.min_clients}")
    print(f"  Fraction:     {args.fraction}")
    print(f"  FedProx mu:   {args.mu}")
    print(f"  Listening on: 0.0.0.0:{args.port}")
    print(f"{'='*55}\n")

    strategy = FedProxStrategy(
        mu=args.mu,
        fraction_fit=args.fraction,
        fraction_evaluate=args.fraction,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        on_fit_config_fn=lambda rnd: {"round": rnd},
    )

    history = fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    # Save round history for plotting
    import json
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/round_history.json", "w") as f:
        json.dump(strategy.round_history, f, indent=2)

    print(f"\n✅ Training complete. History saved to results/round_history.json")
    return history


if __name__ == "__main__":
    main()
