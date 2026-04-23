# T2.3 — Grid outage forecaster + appliance prioritizer

CPU-only project for **24-hour-ahead** grid **outage risk** and **appliance load-shed plans**.

- **Data:** synthetic hourly `grid_history.csv` (plus `appliances.json`, `businesses.json`) from `generate_data.py`.
- **Forecast:** `forecaster.py` — **LightGBM** classifier with **isotonic calibration**, **quantile** models for probability bands, and **LightGBM** regressors for duration (point + quantiles).
- **Plan:** `prioritizer.py` — `plan()` outputs per-hour **ON/OFF** per appliance under a generator limit and luxury → comfort → critical shed order.
- **UI:** static **`lite_ui.html`** (built by `export_ui.py` from `lite_ui.template.html`).

**Repository:** [github.com/AdaSeteye/Grid_Outage_Forecaster_and_Appliance_Prioritizer](https://github.com/AdaSeteye/Grid_Outage_Forecaster_and_Appliance_Prioritizer)

## Quick start (≤2 commands, CPU)

```bash
pip install -r requirements.txt
python minimal_run.py
```

`minimal_run.py` runs `generate_data.py`, then `export_ui.py`. Open **`lite_ui.html`** in a browser (local file is fine).

In **Google Colab**, run the same two lines in a code cell (optionally prefix with `!`).

### Other commands

```bash
python generate_data.py   # refresh CSV + JSON only
python export_ui.py       # rebuild lite_ui.html (needs existing grid_history.csv)
```

## Evaluation

Open **`eval.ipynb`** in Jupyter, run all cells: rolling **30-day** **Brier** (probability), **MAE** on duration where outages occur, and **lead time** to outage starts (see notebook for definitions).

## Main files

| File | Role |
|------|------|
| `forecaster.py` | `Forecaster.fit`, `predict_next_24h` |
| `prioritizer.py` | `plan`, `load_plan_inputs` |
| `eval.ipynb` | Held-out metrics |
| `lite_ui.html` | Demo page (forecast + plan) |
| `digest_spec.md` | Product / SMS / offline notes |
| `process_log.md` | Timeline + tool declaration |

## Model checkpoint (Hugging Face Hub)

**Model + model card:** [huggingface.co/AddisuSeteye/Grid_Outage_Forecaster_and_Appliance_Prioritizer](https://huggingface.co/AddisuSeteye/Grid_Outage_Forecaster_and_Appliance_Prioritizer) — download **`forecaster.joblib`** from **Files and versions**; usage and metadata are on the Hub **README** (model card).

Loading still needs this **GitHub** repo on `PYTHONPATH` for the `forecaster` module. Example:

```python
import joblib
from forecaster import load_grid_csv

m = joblib.load("forecaster.joblib")
df = load_grid_csv("grid_history.csv")
o = len(df) - 25
out = m.predict_next_24h(df.iloc[: o + 25], origin_idx=o)
```

## License

See `LICENSE`.
