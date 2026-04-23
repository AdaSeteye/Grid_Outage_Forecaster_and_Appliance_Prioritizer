"""Write ``lite_ui.html`` from ``lite_ui.template.html`` (salon forecast + plan)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from forecaster import Forecaster, load_grid_csv
from prioritizer import load_plan_inputs, plan

ROOT = Path(__file__).resolve().parent


def main() -> None:
    df = load_grid_csv(ROOT / "grid_history.csv")
    origin = len(df) - 25
    if origin < 200:
        raise RuntimeError("grid_history.csv too short for a 24h demo window")

    model = Forecaster()
    model.fit(df.iloc[: origin + 1], max_label_index=origin)

    forecast = model.predict_next_24h(df.iloc[: origin + 25], origin_idx=origin)
    businesses = json.loads((ROOT / "businesses.json").read_text(encoding="utf-8"))
    salon = next(b for b in businesses if b["id"] == "salon")
    apps, gen_w = load_plan_inputs(salon)
    schedule = plan(forecast, apps, gen_w)

    truth = []
    for h in range(1, 25):
        row = df.iloc[origin + h]
        truth.append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "outage": int(row["outage"]),
                "duration_min": float(row["duration_min"]),
            }
        )

    payload = {
        "business": salon,
        "forecast": forecast,
        "plan": schedule,
        "truth": truth,
        "latency_ms": model.latency_ms(),
        "appliance_category": {a["id"]: a["category"] for a in apps},
    }

    template = (ROOT / "lite_ui.template.html").read_text(encoding="utf-8")
    marker_begin = "/*__UI_JSON_BEGIN__*/"
    marker_end = "/*__UI_JSON_END__*/"
    if marker_begin not in template or marker_end not in template:
        raise RuntimeError("lite_ui.template.html missing JSON markers")

    before, rest = template.split(marker_begin, 1)
    mid, after = rest.split(marker_end, 1)
    json_txt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    out_html = before + marker_begin + json_txt + marker_end + after
    (ROOT / "lite_ui.html").write_text(out_html, encoding="utf-8")
    print(f"Wrote lite_ui.html ({len(out_html) / 1024:.1f} KB). predict latency: {model.latency_ms():.2f} ms")


if __name__ == "__main__":
    main()
