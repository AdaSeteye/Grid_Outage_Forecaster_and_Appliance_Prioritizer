"""Synthetic ``grid_history.csv`` + JSON fixtures. ``python generate_data.py``."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
ROOT = Path(__file__).resolve().parent


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_grid_history(n_days: int = 180) -> pd.DataFrame:
    n = n_days * 24
    t = np.arange(n, dtype=np.float64)
    hour = (t % 24).astype(np.int32)
    dow = ((t // 24) % 7).astype(np.int32)

    base = 120.0
    daily = 25.0 * np.exp(-0.5 * ((hour - 8) / 3.0) ** 2) + 20.0 * np.exp(
        -0.5 * ((hour - 19) / 3.5) ** 2
    )
    weekly = 6.0 * np.sin(2 * math.pi * dow / 7.0)
    rainy_season = 4.0 * np.sin(2 * math.pi * t / (90 * 24))
    noise = RNG.normal(0, 2.5, size=n)

    rain_mm = np.clip(RNG.gamma(shape=1.2, scale=0.8, size=n), 0, 40)
    load_mw = base + daily + weekly + rainy_season + 0.08 * rain_mm + noise
    load_mw = np.clip(load_mw, 60, 220)

    temp_c = 22 + 6 * np.sin(2 * math.pi * t / (365 * 24)) + RNG.normal(0, 1.5, n)
    humidity = np.clip(45 + 0.35 * rain_mm + RNG.normal(0, 6, n), 15, 99)
    wind_ms = np.clip(RNG.lognormal(mean=1.0, sigma=0.35, size=n), 0.5, 25)

    load_lag1 = np.roll(load_mw, 1)
    load_lag1[0] = load_mw[0]

    a0, a1, a2, a3 = -4.35, 0.008, 0.028, -0.05
    z = a0 + a1 * load_lag1 + a2 * rain_mm + a3 * (hour - 12)
    p_out = _sigmoid(z)
    outage = RNG.binomial(1, p_out)

    mu_log = math.log(90) - 0.5 * (0.6**2)
    dur = RNG.lognormal(mean=mu_log, sigma=0.6, size=n)
    dur = np.clip(dur, 5, 600)
    duration_min = np.where(outage.astype(bool), dur, 0.0)

    start = pd.Timestamp("2024-01-01T00:00:00Z")
    ts = start + pd.to_timedelta(t, unit="h")

    return pd.DataFrame(
        {
            "timestamp": ts,
            "load_mw": load_mw.astype(np.float32),
            "temp_c": temp_c.astype(np.float32),
            "humidity": humidity.astype(np.float32),
            "wind_ms": wind_ms.astype(np.float32),
            "rain_mm": rain_mm.astype(np.float32),
            "outage": outage.astype(np.int8),
            "duration_min": duration_min.astype(np.float32),
        }
    )


def main() -> None:
    df = generate_grid_history(180)
    df.to_csv(ROOT / "grid_history.csv", index=False)

    appliances = [
        {
            "id": "hair_dryer",
            "name": "Hair dryer bank",
            "category": "luxury",
            "watts_avg": 2200,
            "start_up_spike_w": 2600,
            "revenue_if_running_rwf_per_h": 12000,
        },
        {
            "id": "music",
            "name": "Bluetooth speaker",
            "category": "luxury",
            "watts_avg": 40,
            "start_up_spike_w": 60,
            "revenue_if_running_rwf_per_h": 2500,
        },
        {
            "id": "ac_unit",
            "name": "Split AC",
            "category": "comfort",
            "watts_avg": 1800,
            "start_up_spike_w": 3200,
            "revenue_if_running_rwf_per_h": 18000,
        },
        {
            "id": "water_heater",
            "name": "Water heater",
            "category": "comfort",
            "watts_avg": 2000,
            "start_up_spike_w": 2000,
            "revenue_if_running_rwf_per_h": 8000,
        },
        {
            "id": "lights",
            "name": "LED lighting (full shop)",
            "category": "comfort",
            "watts_avg": 350,
            "start_up_spike_w": 400,
            "revenue_if_running_rwf_per_h": 14000,
        },
        {
            "id": "fridge_display",
            "name": "Display fridge",
            "category": "critical",
            "watts_avg": 250,
            "start_up_spike_w": 800,
            "revenue_if_running_rwf_per_h": 22000,
        },
        {
            "id": "cold_room_comp",
            "name": "Cold-room compressor",
            "category": "critical",
            "watts_avg": 3500,
            "start_up_spike_w": 7000,
            "revenue_if_running_rwf_per_h": 45000,
        },
        {
            "id": "sewing_motor",
            "name": "Industrial sewing motor",
            "category": "critical",
            "watts_avg": 400,
            "start_up_spike_w": 900,
            "revenue_if_running_rwf_per_h": 28000,
        },
        {
            "id": "pos",
            "name": "POS + router",
            "category": "critical",
            "watts_avg": 80,
            "start_up_spike_w": 120,
            "revenue_if_running_rwf_per_h": 9000,
        },
        {
            "id": "security",
            "name": "CCTV + alarm hub",
            "category": "critical",
            "watts_avg": 60,
            "start_up_spike_w": 90,
            "revenue_if_running_rwf_per_h": 5000,
        },
    ]

    businesses = [
        {
            "id": "salon",
            "name": "Urban salon",
            "generator_watts": 6500,
            "appliance_ids": [
                "hair_dryer",
                "music",
                "ac_unit",
                "water_heater",
                "lights",
                "fridge_display",
                "pos",
            ],
        },
        {
            "id": "cold_room",
            "name": "Produce cold room",
            "generator_watts": 9000,
            "appliance_ids": ["cold_room_comp", "lights", "security", "pos", "water_heater"],
        },
        {
            "id": "tailor",
            "name": "Neighborhood tailor",
            "generator_watts": 3500,
            "appliance_ids": ["sewing_motor", "lights", "music", "ac_unit", "pos"],
        },
    ]

    (ROOT / "appliances.json").write_text(
        json.dumps(appliances, indent=2), encoding="utf-8"
    )
    (ROOT / "businesses.json").write_text(
        json.dumps(businesses, indent=2), encoding="utf-8"
    )

    mean_p = df["outage"].mean()
    print(f"Wrote grid_history.csv ({len(df)} rows). Empirical outage rate: {mean_p:.3f}")


if __name__ == "__main__":
    main()
