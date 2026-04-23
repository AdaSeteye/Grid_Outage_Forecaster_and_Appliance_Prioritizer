"""24h ON/OFF plan: shed luxury→comfort→critical (revenue/W ties); maximize expected revenue among ON sets that fit the generator (no overload on backup)."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Sequence, Tuple

Category = Literal["luxury", "comfort", "critical"]

_CAT_RANK: Dict[str, int] = {"luxury": 0, "comfort": 1, "critical": 2}


def _rev_per_w(app: Mapping[str, Any]) -> float:
    w = float(app["watts_avg"])
    if w <= 0:
        return 0.0
    return float(app["revenue_if_running_rwf_per_h"]) / w


def _shed_sort_key(app: Mapping[str, Any]) -> Tuple[int, float, str]:
    """Shed order: category rank, then ascending revenue/W (then id)."""
    cat = str(app["category"])
    rpw = _rev_per_w(app)
    return (_CAT_RANK.get(cat, 99), rpw, str(app.get("id", app.get("name", ""))))


def plan(
    forecast: Sequence[Mapping[str, Any]],
    appliances: Sequence[Mapping[str, Any]],
    generator_watts: float,
) -> Dict[str, Any]:
    """Return dict with keys hours (length 24), shed_order, generator_watts. Each forecast step must include p_outage."""
    if len(forecast) != 24:
        raise ValueError("forecast must contain exactly 24 hourly steps")

    apps = list(appliances)
    shed_order = sorted(apps, key=_shed_sort_key)
    order_ids = [str(a.get("id", a["name"])) for a in shed_order]
    n = len(shed_order)

    hours_out: List[Dict[str, Any]] = []
    for fh in forecast:
        p = float(fh["p_outage"])
        best_k = n
        best_score = -1e30

        for k in range(0, n + 1):
            kept = shed_order[k:]
            on_rev = sum(float(a["revenue_if_running_rwf_per_h"]) for a in kept)
            on_w = sum(float(a["watts_avg"]) for a in kept)
            if on_w > generator_watts:
                continue
            e = (1.0 - p) * on_rev + p * on_rev
            tie = 1e-9 * (n - k)
            score = e + tie
            if score > best_score:
                best_score = score
                best_k = k

        kept_set = {str(a.get("id", a["name"])) for a in shed_order[best_k:]}
        ap_states = {str(a.get("id", a["name"])): ("ON" if str(a.get("id", a["name"])) in kept_set else "OFF") for a in apps}

        row: Dict[str, Any] = {
            "timestamp": fh.get("timestamp"),
            "p_outage": p,
            "appliances": ap_states,
        }
        hours_out.append(row)

    return {
        "hours": hours_out,
        "shed_order": order_ids,
        "generator_watts": float(generator_watts),
    }


def load_plan_inputs(
    business: Mapping[str, Any],
    appliances_path: str = "appliances.json",
) -> Tuple[List[Dict[str, Any]], float]:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent
    all_apps = json.loads((root / appliances_path).read_text(encoding="utf-8"))
    id_set = set(business["appliance_ids"])
    chosen = [a for a in all_apps if a["id"] in id_set]
    if len(chosen) != len(id_set):
        missing = id_set - {a["id"] for a in chosen}
        raise ValueError(f"Missing appliance ids: {missing}")
    return chosen, float(business["generator_watts"])
