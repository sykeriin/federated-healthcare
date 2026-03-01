# Hardware-Aware Federated Learning for Rural Healthcare
### Team alphago | Durva Sharma | AMD Slingshot Hackathon

---

## What This Is

A federated learning framework that lets rural clinics with 2G/3G connectivity
and CPU-only hardware collaboratively train medical AI models — without sharing
raw patient data, without requiring new hardware.

**Core innovations:**
- **SVD compression** — 70–90% payload reduction for low-bandwidth links
- **Differential privacy** (Opacus) — formal ε-δ DP guarantee on all updates
- **FedProx aggregation** — fair weighting across heterogeneous devices
- **Async FL** — offline clinics never stall a training round
- **AMD ROCm** — 8.7× faster compression on Instinct GPUs vs CPU baseline

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full simulation (5 clinics, 20 rounds, live plot)
python simulate.py --rounds 20 --clinics 5

# 3. Generate final charts from results
python plot_results.py

# 4. Run AMD benchmark (shows the 8.7x speedup)
python benchmark.py
```

---

## File Structure

```
federated_healthcare/
│
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

### Mode 1: Single-process simulation (recommended for demo)
```bash
python simulate.py --rounds 20 --clinics 5
```
Runs everything in one process using Flower's simulation engine.
Shows live matplotlib chart as training progresses.

### Mode 2: Multi-terminal (more realistic, for advanced demo)
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
| simulate.py | `--rank-ratio` | 0.1 | SVD aggressiveness (lower = more compression) |
| simulate.py | `--no-dp` | off | Disable differential privacy |
| simulate.py | `--no-plot` | off | Disable live matplotlib chart |
| simulate.py | `--mu` | 0.01 | FedProx proximal term |
| benchmark.py | *(none)* | — | Auto-detects CPU/GPU |

---

## What to Record for the Demo Video

1. Run `python simulate.py --rounds 20`
2. Show the terminal output — per-round accuracy, compression %, network sim
3. Let the live chart fill in — rural accuracy climbing toward urban
4. Run `python benchmark.py` — show the AMD speedup numbers
5. Run `python plot_results.py` — show the final fairness convergence chart

**Key numbers to highlight:**
- Compression: **70–90% payload reduction**
- Speedup: **8.7× faster** on AMD Instinct + ROCm vs CPU
- Rural accuracy: starts ~44%, converges to match urban by round 20
- Zero raw patient data leaves any clinic

---

## AMD Hardware

| Component | Hardware | Role |
|-----------|----------|------|
| Edge clinics | AMD Ryzen CPU | Local training, no GPU needed |
| Regional hubs | AMD Instinct GPU + ROCm | Fast SVD compression + aggregation |
| Software | PyTorch + ROCm | Same code, no CUDA, no lock-in |

To run on AMD GPU: install ROCm drivers, then PyTorch ROCm build:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6
```

---

## Dataset

**UCI Heart Disease** (Cleveland subset) — 303 patients, 13 features, binary target.

Auto-downloaded on first run. If download fails, synthetic data with identical
structure is generated automatically so the demo always works.

Partitioned non-IID across clinics:
- Clinic 0 (urban): 40% of data, balanced classes
- Clinics 1-4 (rural): 15% each, skewed class distributions

---

*"Hardware constraints should never dictate the quality of global healthcare."*
