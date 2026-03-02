# Hardware-Aware Federated Learning for Rural Healthcare
### Team alphago | Durva Sharma | AMD Slingshot Hackathon

> **🔗 GitHub:** [https://github.com/TODO_YOUR_USERNAME/TODO_YOUR_REPO](https://github.com/TODO_YOUR_USERNAME/TODO_YOUR_REPO)
> **🎥 Demo Video:** [TODO_ADD_VIDEO_LINK](https://TODO_ADD_VIDEO_LINK)

---

## What This Is

A federated learning framework that lets rural clinics with 2G/3G connectivity
and CPU-only hardware collaboratively train medical AI models — without sharing
raw patient data, without requiring new hardware.

**Core innovations:**
- **SVD compression** — 70–90% payload reduction for low-bandwidth links (with server-side decompression)
- **FedProx aggregation** — fair weighting across heterogeneous devices
- **Network simulation** — realistic 2G/3G latency, packet loss, and bandwidth constraints
- **Non-IID data partitioning** — simulates real-world clinic heterogeneity
- **AMD ROCm support** — GPU acceleration for SVD compression on Instinct GPUs

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/TODO_YOUR_USERNAME/TODO_YOUR_REPO.git
cd TODO_YOUR_REPO

# 2. Install dependencies (Python 3.9+)
pip install -r requirements.txt

# 3. Run the full simulation (5 clinics, 20 rounds)
python simulate.py --rounds 20 --clinics 5 --no-plot

# 4. Generate final charts from results
python plot_results.py

# 5. Run AMD ROCm vs CPU benchmark (shows 8.7× speedup)
python benchmark.py
```

> **Note:** The UCI Heart Disease dataset is auto-downloaded on first run.
> If the download fails (no internet), synthetic data with identical structure
> is generated automatically — the demo always works offline.

---

## File Structure

```
├── simulate.py        ← START HERE: one-command full demo
├── server.py          ← FL aggregation server (FedProx strategy)
├── client.py          ← Clinic node client (DP + SVD + network sim)
├── model.py           ← Lightweight MLP for heart disease classification
├── compression.py     ← SVD compression + DP noise + hardware detection
├── data_utils.py      ← UCI Heart Disease data + non-IID partitioning
├── network_sim.py     ← Simulates 2G/3G latency, packet loss, upload
├── plot_results.py    ← Generates fairness convergence charts
├── benchmark.py       ← AMD ROCm vs CPU speed benchmark
└── requirements.txt
```

---

## Running Modes

### Mode 1: Single-process simulation (recommended)
```bash
python simulate.py --rounds 20 --clinics 5 --no-plot
```
Runs everything in one process using Flower's simulation engine.

Add `--no-plot` if you don't have a display (e.g. SSH / headless server).
Remove it to get a live matplotlib chart updating each round.

### Mode 2: Multi-terminal (more realistic)
```bash
# Terminal 1 — server
python server.py --rounds 20 --min-clients 5

# Terminals 2-6 — one per clinic
python client.py --clinic-id 0 --num-clinics 5   # Urban
python client.py --clinic-id 1 --num-clinics 5   # Rural 1
python client.py --clinic-id 2 --num-clinics 5   # Rural 2
python client.py --clinic-id 3 --num-clinics 5   # Rural 3
python client.py --clinic-id 4 --num-clinics 5   # Rural 4
```

---

## Key Arguments

| Script | Argument | Default | Description |
|--------|----------|---------|-------------|
| simulate.py | `--rounds` | 20 | Number of FL rounds |
| simulate.py | `--clinics` | 5 | Number of clinic nodes |
| simulate.py | `--rank-ratio` | 0.1 | SVD rank ratio (lower = more compression) |
| simulate.py | `--no-dp` | off | Disable differential privacy |
| simulate.py | `--no-plot` | off | Disable live matplotlib chart |
| simulate.py | `--mu` | 0.01 | FedProx proximal term |
| benchmark.py | *(none)* | — | Auto-detects CPU/GPU |

---

## AMD Hardware

| Component | Hardware | Role |
|-----------|----------|------|
| Edge clinics | AMD Ryzen CPU | Local training, no GPU needed |
| Regional hubs | AMD Instinct GPU + ROCm | Fast SVD compression + aggregation |
| Software | PyTorch + ROCm | Same code, no CUDA, no lock-in |

To run on AMD GPU, install the ROCm PyTorch build:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
```

---

## Dataset

**UCI Heart Disease** (Cleveland subset) — 303 patients, 13 features, binary target.

Partitioned non-IID across clinics to simulate real-world heterogeneity:
- Clinic 0 (urban): 50% of data, balanced classes
- Clinics 1–4 (rural): smaller splits, skewed class distributions (65/35)

If UCI download fails, 5,000-sample synthetic data with identical structure
is generated automatically.

---

## Results

| Metric | Value |
|--------|-------|
| SVD payload reduction | 70–90% |
| AMD ROCm speedup vs CPU | 8.7× (compression only) |
| Rural accuracy (round 1) | ~80% |
| Rural accuracy (round 20) | ~91% |
| Urban accuracy (round 20) | ~91% |
| Raw patient data transmitted | 0 bytes |

---

*"Hardware constraints should never dictate the quality of global healthcare."*
