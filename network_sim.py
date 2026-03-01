"""
network_sim.py
Simulates real-world rural network conditions: latency, packet loss,
and bandwidth-limited upload. Used to make the demo realistic.
"""

import time
import random
import sys


# ─────────────────────────────────────────────────────────────────────────────
# Network profiles (maps clinic_id → typical rural network conditions)
# ─────────────────────────────────────────────────────────────────────────────

NETWORK_PROFILES = {
    0: {  # Urban — fast, stable
        "name":         "4G LTE (Urban)",
        "latency_ms":   50,
        "jitter_ms":    10,
        "packet_loss":  0.0,
        "bandwidth_kbps": 5000,
    },
    1: {  # Rural — slow 3G
        "name":         "3G (Rural)",
        "latency_ms":   450,
        "jitter_ms":    120,
        "packet_loss":  0.05,
        "bandwidth_kbps": 200,
    },
    2: {  # Rural — very slow 2G
        "name":         "2G EDGE (Remote Rural)",
        "latency_ms":   800,
        "jitter_ms":    200,
        "packet_loss":  0.12,
        "bandwidth_kbps": 64,
    },
    3: {  # Rural — intermittent (worst case)
        "name":         "Intermittent 2G",
        "latency_ms":   1200,
        "jitter_ms":    400,
        "packet_loss":  0.20,
        "bandwidth_kbps": 32,
    },
    4: {  # Rural — moderate 3G
        "name":         "3G (Semi-Rural)",
        "latency_ms":   350,
        "jitter_ms":    80,
        "packet_loss":  0.08,
        "bandwidth_kbps": 150,
    },
}

# Default profile for clinic IDs beyond the defined range
DEFAULT_RURAL_PROFILE = NETWORK_PROFILES[1]
MAX_RETRIES = 3


def get_profile(clinic_id: int) -> dict:
    return NETWORK_PROFILES.get(clinic_id, DEFAULT_RURAL_PROFILE)


def simulate_rural_network(
    clinic_id: int,
    upload: bool = False,
    payload_kb: float = 0.0,
) -> bool:
    """
    Simulate network conditions for a given clinic.

    Args:
        clinic_id:  Clinic index (0 = urban, 1+ = rural)
        upload:     If True, simulate upload time based on payload size.
        payload_kb: Size of payload in KB (used for upload simulation).

    Returns:
        True if transmission succeeded, False if dropped (packet loss).
    """
    profile = get_profile(clinic_id)

    # Check packet loss
    if random.random() < profile["packet_loss"]:
        print(f"  [Clinic {clinic_id}] ⚠️  Packet loss on {profile['name']} "
              f"(simulated {profile['packet_loss']*100:.0f}% loss rate)")

        # Retry logic
        for attempt in range(1, MAX_RETRIES + 1):
            time.sleep(0.3)
            if random.random() >= profile["packet_loss"]:
                print(f"  [Clinic {clinic_id}] 🔁 Retry {attempt} succeeded")
                break
            if attempt == MAX_RETRIES:
                print(f"  [Clinic {clinic_id}] ❌ Max retries exceeded — skipping round")
                return False

    # Simulate latency
    latency_s = (profile["latency_ms"] + random.uniform(
        -profile["jitter_ms"], profile["jitter_ms"]
    )) / 1000.0
    latency_s = max(0.01, latency_s)

    if not upload:
        # Download: just latency
        print(f"  [Clinic {clinic_id}] 🌐 Network: {profile['name']} | "
              f"RTT: {latency_s*1000:.0f}ms")
        time.sleep(min(latency_s, 1.5))  # cap at 1.5s for demo speed
    else:
        # Upload: latency + transfer time
        bw_kbps = profile["bandwidth_kbps"]
        transfer_s = (payload_kb * 8) / bw_kbps  # KB -> kbits / kbps

        print(f"  [Clinic {clinic_id}] 📤 Upload: {payload_kb:.1f} KB @ "
              f"{bw_kbps} kbps → {transfer_s:.2f}s transfer "
              f"+ {latency_s*1000:.0f}ms RTT")

        # Animate upload progress for demo
        _progress_bar(transfer_s, label=f"Clinic {clinic_id} uploading")
        time.sleep(min(latency_s * 0.5, 0.5))

    return True


def _progress_bar(duration_s: float, label: str = "Uploading", width: int = 30):
    """Simple ASCII progress bar for demo visual."""
    steps = width
    step_sleep = min(duration_s / steps, 0.05)

    sys.stdout.write(f"  [{label}] [")
    sys.stdout.flush()

    for i in range(steps):
        time.sleep(step_sleep)
        sys.stdout.write("█")
        sys.stdout.flush()

    sys.stdout.write("] ✓\n")
    sys.stdout.flush()
