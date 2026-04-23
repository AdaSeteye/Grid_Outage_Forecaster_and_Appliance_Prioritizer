# Process log - T2.3

## Hour-by-hour timeline (4 h cap)

| Window | Minutes | Work |
|--------|--------:|------|
| **0:00–0:35** | 35 | **Problem understanding:** read the write-up end-to-end; list deliverables and metrics; sketch how forecast feeds `plan()`; choose stack (CPU, LightGBM + calibration, static UI). |
| **0:35–1:00** | 25 | **Data:** repo layout; `generate_data.py`; run → `grid_history.csv`, `appliances.json`, `businesses.json`; quick plots / counts on outages. |
| **1:00–1:55** | 55 | **Forecaster:** `forecaster.py` features + labels, train, `predict_next_24h`; verify CPU inference latency. |
| **1:55–2:35** | 40 | **Prioritizer:** `prioritizer.py` `plan()`, shed order, generator watts; wire to a sample 24h forecast (salon). |
| **2:35–3:20** | 45 | **Evaluation:** `eval.ipynb` rolling 30-day window; Brier, duration MAE, lead time; fix leakage or bottlenecks if needed. |
| **3:20–3:45** | 25 | **Deployment:** `lite_ui.template.html`, `export_ui.py`, `minimal_run.py`; regenerate `lite_ui.html`; open locally and sanity-check chart + table. |
| **3:45–4:00** | 15 | **Submission:** `digest_spec.md`; `README.md` + `LICENSE`; `SIGNED.md`; this `process_log.md`; `git init` / commit / push to public remote and confirm two-command repro on a clean folder or Colab. |
| **Total** | **240** | |

## Tools / LLM use

| Tool | Why |
|------|-----|
| Cursor
| Local Python 3 | Run generator, export UI, sanity-check metrics |

## Three sample prompts actually used (paraphrase OK)

1. “Complete this data generating script.”
2. “This training loop is slow; here’s my feature code — fix it without changing the logic.”
3. “Check my `plan()` function: luxury off before critical, same generator rule.”

## One prompt discarded

**Discarded:** “Add XGBoost and a Docker API.” **Why:** overkill for the time limit; I kept **LightGBM** on CPU and a static **`lite_ui.html`**.

## Hardest decision (one paragraph)

The challenge allows several forecast formulations; the hardest choice was how to represent **multi-step** targets without leaking future weather. I trained on **(origin × horizon)** rows with **calendar-hour profiles** for rain/load built only from history **strictly before** the origin index, then used the same feature builder at inference. This trades a full weather feed for a causal feature set that fits the low-bandwidth story. I used **LightGBM** with **isotonic calibration** and **quantile** heads for bands so rolling re-fits stay under the **10-minute** CPU budget and **24h inference stays under 300 ms** on a laptop.
