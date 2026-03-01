"""
client.py
Federated Learning client — simulates a rural/urban clinic node.

Each client:
  1. Receives global model weights from the server
  2. Trains locally on its private patient data
  3. Adds differential privacy noise (Opacus)
  4. Compresses the update via SVD
  5. Sends compressed update back to the server

Run standalone (for testing a single clinic):
    python client.py --clinic-id 0 --partition 0
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import flwr as fl

from model import HeartDiseaseModel, get_model_parameters, set_model_parameters
from compression import (
    compress_model_update,
    add_dp_noise,
    detect_hardware,
    print_compression_stats,
)
from data_utils import load_heart_disease_data, partition_data, make_dataloader
from network_sim import simulate_rural_network

# Try to import Opacus for formal DP — fall back to manual noise if unavailable
try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print("⚠️  Opacus not installed — using manual DP noise. "
          "Install with: pip install opacus")


# ─────────────────────────────────────────────────────────────────────────────
# Clinic Client
# ─────────────────────────────────────────────────────────────────────────────

class ClinicClient(fl.client.NumPyClient):
    """
    Federated Learning client representing one clinic node.

    Args:
        clinic_id:       Integer ID (0 = urban, 1+ = rural)
        X_train, y_train: Local private training data
        X_test, y_test:  Global test set (for local evaluation)
        use_dp:          Enable differential privacy
        rank_ratio:      SVD compression aggressiveness (0.0–1.0)
        noise_multiplier: DP noise scale
    """

    def __init__(
        self,
        clinic_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        use_dp:  bool = True,
        rank_ratio: float = None,
        noise_multiplier: float = 0.5,
    ):
        self.clinic_id        = clinic_id
        self.is_rural         = clinic_id > 0
        self.use_dp           = use_dp
        self.noise_multiplier = noise_multiplier

        # Hardware-aware settings
        hw = detect_hardware()
        self.device     = torch.device(hw["device"])
        self.epochs     = hw["local_epochs"]
        self.rank_ratio = rank_ratio if rank_ratio is not None else hw["rank_ratio"]

        clinic_type = "🏥 Rural " if self.is_rural else "🏙️  Urban"
        print(f"\n[Clinic {clinic_id}] {clinic_type} | "
              f"Device: {hw['device'].upper()} | "
              f"GPU: {hw['gpu_name'] or 'None'} | "
              f"Epochs: {self.epochs} | "
              f"Rank ratio: {self.rank_ratio}")

        # Build data loaders
        batch_size = min(16, len(y_train))
        self.trainloader = make_dataloader(X_train, y_train, batch_size=batch_size)
        self.testloader  = make_dataloader(X_test,  y_test,  batch_size=32, shuffle=False)
        self.n_train = len(y_train)

        # Model
        self.model = HeartDiseaseModel(input_dim=X_train.shape[1]).to(self.device)

        # Optimizer — will be wrapped by Opacus if DP enabled
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        # Opacus privacy engine
        self.privacy_engine = None
        if self.use_dp and OPACUS_AVAILABLE and len(y_train) > batch_size:
            self._attach_privacy_engine()

        # Track accuracy history for fairness plot
        self.accuracy_history = []

    def _attach_privacy_engine(self):
        """Wrap model + optimizer with Opacus for formal DP guarantees."""
        try:
            self.privacy_engine = PrivacyEngine()
            (
                self.model,
                self.optimizer,
                self.trainloader,
            ) = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.trainloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=1.0,
            )
            print(f"  [Clinic {self.clinic_id}] 🔐 Opacus DP enabled "
                  f"(noise={self.noise_multiplier})")
        except Exception as e:
            print(f"  [Clinic {self.clinic_id}] ⚠️  Opacus failed ({e}) — "
                  f"using manual DP noise")
            self.privacy_engine = None

    # ── Flower interface ───────────────────────────────────────────────────

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        """
        Receive global model → train locally → compress → return update.
        """
        round_num = config.get("round", "?")
        print(f"\n{'─'*55}")
        print(f"[Clinic {self.clinic_id}] 🔄 Round {round_num} — Starting local training")

        # Load global model weights
        set_model_parameters(self.model, parameters)

        # Simulate rural network latency before training starts
        if self.is_rural:
            simulate_rural_network(clinic_id=self.clinic_id)

        # ── Local training loop ──
        self.model.train()
        t_train = time.time()

        for epoch in range(self.epochs):
            running_loss = 0.0
            for X_batch, y_batch in self.trainloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(self.trainloader)
            print(f"  [Clinic {self.clinic_id}] Epoch {epoch+1}/{self.epochs} "
                  f"| Loss: {avg_loss:.4f}")

        train_time = time.time() - t_train
        print(f"  [Clinic {self.clinic_id}] ⏱️  Training time: {train_time:.2f}s")

        # ── Get updated parameters ──
        updated_params = get_model_parameters(self.model)

        # ── Manual DP noise (fallback if Opacus not used) ──
        if self.use_dp and self.privacy_engine is None:
            updated_params = add_dp_noise(
                updated_params,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=1.0,
            )
            print(f"  [Clinic {self.clinic_id}] 🔐 Manual DP noise applied")

        # ── SVD Compression ──
        compressed, stats = compress_model_update(
            updated_params,
            rank_ratio=self.rank_ratio,
        )
        print_compression_stats(stats, self.clinic_id)

        # Decompress before sending to server (Flower expects plain numpy arrays)
        # In a real system you'd send compressed bytes; here we simulate the ratio
        from compression import decompress_model_update
        final_params = decompress_model_update(compressed)

        # Simulate upload with bandwidth constraint
        if self.is_rural:
            simulate_rural_network(clinic_id=self.clinic_id, upload=True,
                                   payload_kb=stats["compressed_kb"])

        return final_params, self.n_train, {
            "clinic_id":       self.clinic_id,
            "is_rural":        int(self.is_rural),
            "compression_pct": stats["reduction_pct"],
            "train_loss":      avg_loss,
        }

    def evaluate(self, parameters, config):
        """Evaluate global model on local test set."""
        set_model_parameters(self.model, parameters)

        self.model.eval()
        correct = 0
        total   = 0
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in self.testloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total   += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(self.testloader)

        self.accuracy_history.append(accuracy)

        clinic_type = "Rural " if self.is_rural else "Urban "
        print(f"  [Clinic {self.clinic_id}] 📊 {clinic_type} Accuracy: "
              f"{accuracy*100:.2f}% | Loss: {avg_loss:.4f}")

        return avg_loss, total, {
            "accuracy": accuracy,
            "clinic_id": self.clinic_id,
            "is_rural": int(self.is_rural),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Start a clinic FL client")
    parser.add_argument("--clinic-id",  type=int, default=0,
                        help="Clinic ID (0=urban, 1+=rural)")
    parser.add_argument("--num-clinics", type=int, default=5,
                        help="Total number of clinics in federation")
    parser.add_argument("--server",     type=str, default="127.0.0.1:8080",
                        help="Server address host:port")
    parser.add_argument("--no-dp",      action="store_true",
                        help="Disable differential privacy")
    parser.add_argument("--rank-ratio", type=float, default=None,
                        help="SVD rank ratio (0.05–0.3)")
    args = parser.parse_args()

    # Load and partition data
    X, y = load_heart_disease_data()
    clinic_datasets, (X_test, y_test) = partition_data(X, y, num_clinics=args.num_clinics)

    X_train, y_train = clinic_datasets[args.clinic_id]

    client = ClinicClient(
        clinic_id  = args.clinic_id,
        X_train    = X_train,
        y_train    = y_train,
        X_test     = X_test,
        y_test     = y_test,
        use_dp     = not args.no_dp,
        rank_ratio = args.rank_ratio,
    )

    fl.client.start_numpy_client(
        server_address=args.server,
        client=client,
    )


if __name__ == "__main__":
    main()
