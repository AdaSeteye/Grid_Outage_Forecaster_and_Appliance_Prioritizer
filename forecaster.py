"""24h grid outage forecaster: calibrated LightGBM classifier, quantile P-bands, duration GBMs. Causal features (no leakage past forecast origin)."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent

_LGBM_COMMON = dict(
    n_estimators=100,
    learning_rate=0.07,
    num_leaves=40,
    min_child_samples=70,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
)

_LGBM_Q_PROB = dict(
    n_estimators=70,
    learning_rate=0.08,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=43,
    verbosity=-1,
    n_jobs=-1,
)

_LGBM_DUR = dict(
    n_estimators=80,
    learning_rate=0.06,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=44,
    verbosity=-1,
    n_jobs=-1,
)


def _ensure_utc_ts(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, utc=True)
    if getattr(out.dt, "tz", None) is None:
        out = out.dt.tz_localize("UTC")
    return out


def _slot_ids(d: pd.DataFrame) -> np.ndarray:
    return (d["dow"].to_numpy(dtype=np.int32) * 24 + d["hour"].to_numpy(dtype=np.int32)).astype(
        np.int32
    )


def _order_prob_band(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.clip(lo.astype(np.float64), 0.0, 1.0)
    hi = np.clip(hi.astype(np.float64), 0.0, 1.0)
    a = np.minimum(lo, hi)
    b = np.maximum(lo, hi)
    a = np.minimum(a, p)
    b = np.maximum(b, p)
    return a, b


def _order_duration_band(mid: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.minimum(lo.astype(np.float64), hi.astype(np.float64))
    hi = np.maximum(lo.astype(np.float64), hi.astype(np.float64))
    lo = np.minimum(lo, mid)
    hi = np.maximum(hi, mid)
    return np.maximum(lo, 5.0), np.maximum(hi, 5.0)


class Forecaster:
    """Train on history rows; predict 24h ahead with probabilities and duration estimates."""

    def __init__(self, min_train_rows: int = 800) -> None:
        self.min_train_rows = min_train_rows
        self._clf: Optional[Any] = None
        self._p_q_lo: Optional[lgb.LGBMRegressor] = None
        self._p_q_hi: Optional[lgb.LGBMRegressor] = None
        self._dur: Optional[lgb.LGBMRegressor] = None
        self._dur_q_lo: Optional[lgb.LGBMRegressor] = None
        self._dur_q_hi: Optional[lgb.LGBMRegressor] = None
        self._meta: Dict[str, Any] = {}

    @staticmethod
    def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        ts = _ensure_utc_ts(x["timestamp"])
        x["hour"] = ts.dt.hour.astype(np.int16)
        x["dow"] = ts.dt.dayofweek.astype(np.int16)
        x["month"] = ts.dt.month.astype(np.int16)
        x["hour_sin"] = np.sin(2 * math.pi * x["hour"] / 24.0)
        x["hour_cos"] = np.cos(2 * math.pi * x["hour"] / 24.0)
        x["dow_sin"] = np.sin(2 * math.pi * x["dow"] / 7.0)
        x["dow_cos"] = np.cos(2 * math.pi * x["dow"] / 7.0)
        return x

    def _build_supervised(
        self, df: pd.DataFrame, max_label_index: Optional[int] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        d = self._add_calendar(df).reset_index(drop=True)
        n = len(d)
        if max_label_index is None:
            max_label_index = n - 1
        t_max = int(max_label_index) - 24
        rows: List[Dict[str, float]] = []
        y_out: List[int] = []
        y_dur: List[float] = []

        outage = d["outage"].to_numpy(dtype=np.int8)
        duration = d["duration_min"].to_numpy(dtype=np.float64)
        rain = d["rain_mm"].to_numpy(dtype=np.float64)
        load = d["load_mw"].to_numpy(dtype=np.float64)
        slot = _slot_ids(d)
        sum_l = np.zeros(7 * 24, dtype=np.float64)
        sum_r = np.zeros(7 * 24, dtype=np.float64)
        cnt = np.zeros(7 * 24, dtype=np.float64)
        g_l = float(load.mean()) if n else 0.0
        g_r = float(rain.mean()) if n else 0.0

        for idx in range(168):
            s = int(slot[idx])
            sum_l[s] += load[idx]
            sum_r[s] += rain[idx]
            cnt[s] += 1.0

        for t in range(168, max(169, t_max + 1)):
            last_load = float(d.at[t, "load_mw"])
            last_rain = float(d.at[t, "rain_mm"])
            roll_out = float(outage[max(0, t - 24) : t].mean()) if t > 0 else 0.0
            roll_rain = float(rain[max(0, t - 72) : t].mean()) if t > 0 else 0.0
            cnt_safe = np.maximum(cnt, 1.0)
            means_l = np.where(cnt > 0, sum_l / cnt_safe, g_l)
            means_r = np.where(cnt > 0, sum_r / cnt_safe, g_r)

            for h in range(1, 25):
                j = t + h
                if j > max_label_index or j >= n:
                    break
                sj = int(slot[j])
                rows.append(
                    {
                        "h": float(h),
                        "hour_sin": float(d.at[j, "hour_sin"]),
                        "hour_cos": float(d.at[j, "hour_cos"]),
                        "dow_sin": float(d.at[j, "dow_sin"]),
                        "dow_cos": float(d.at[j, "dow_cos"]),
                        "prof_load": float(means_l[sj]),
                        "prof_rain": float(means_r[sj]),
                        "last_load": last_load,
                        "last_rain": last_rain,
                        "roll_out_24": roll_out,
                        "roll_rain_72": roll_rain,
                    }
                )
                y_out.append(int(outage[j]))
                y_dur.append(float(duration[j]) if outage[j] else 0.0)

            s_t = int(slot[t])
            sum_l[s_t] += load[t]
            sum_r[s_t] += rain[t]
            cnt[s_t] += 1.0

        X = pd.DataFrame(rows)
        return X, np.asarray(y_out, dtype=np.int8), np.asarray(y_dur, dtype=np.float64)

    def fit(self, df: pd.DataFrame, max_label_index: Optional[int] = None) -> "Forecaster":
        X, y_out, y_dur = self._build_supervised(df, max_label_index=max_label_index)
        if len(X) < self.min_train_rows:
            raise ValueError(f"Need >= {self.min_train_rows} training rows, got {len(X)}")

        pos = int(y_out.sum())
        neg = int(len(y_out) - pos)
        spw = float(neg / max(pos, 1))

        base_clf = lgb.LGBMClassifier(scale_pos_weight=spw, **_LGBM_COMMON)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        self._clf = CalibratedClassifierCV(estimator=base_clf, method="isotonic", cv=cv)
        self._clf.fit(X, y_out)

        y_f = y_out.astype(np.float64)
        self._p_q_lo = lgb.LGBMRegressor(objective="quantile", alpha=0.05, **_LGBM_Q_PROB)
        self._p_q_hi = lgb.LGBMRegressor(objective="quantile", alpha=0.95, **_LGBM_Q_PROB)
        self._p_q_lo.fit(X, y_f)
        self._p_q_hi.fit(X, y_f)

        mask = y_out.astype(bool)
        if mask.sum() < 80:
            self._dur = None
            self._dur_q_lo = None
            self._dur_q_hi = None
            self._meta["duration_fallback"] = float(np.mean(y_dur[mask]) if mask.any() else 90.0)
        else:
            y_log = np.log1p(y_dur[mask])
            Xm = X.loc[mask]
            self._dur = lgb.LGBMRegressor(objective="regression", **_LGBM_DUR)
            self._dur.fit(Xm, y_log)
            self._dur_q_lo = lgb.LGBMRegressor(objective="quantile", alpha=0.1, **_LGBM_DUR)
            self._dur_q_hi = lgb.LGBMRegressor(objective="quantile", alpha=0.9, **_LGBM_DUR)
            self._dur_q_lo.fit(Xm, y_log)
            self._dur_q_hi.fit(Xm, y_log)
            self._meta["duration_fallback"] = float(np.mean(y_dur[mask]))

        self._meta["feature_names"] = list(X.columns)
        return self

    def _features_for_horizon_block(self, df: pd.DataFrame, origin_idx: int) -> pd.DataFrame:
        d = self._add_calendar(df).reset_index(drop=True)
        n = len(d)
        t = int(origin_idx)
        load = d["load_mw"].to_numpy(dtype=np.float64)
        rain = d["rain_mm"].to_numpy(dtype=np.float64)
        slot = _slot_ids(d)
        outage = d["outage"].to_numpy(dtype=np.int8)
        S = 7 * 24
        sum_l = np.zeros(S, dtype=np.float64)
        sum_r = np.zeros(S, dtype=np.float64)
        cnt = np.zeros(S, dtype=np.float64)
        g_l = float(load[: max(1, t)].mean()) if t > 0 else float(load.mean())
        g_r = float(rain[: max(1, t)].mean()) if t > 0 else float(rain.mean())
        for idx in range(0, t):
            s = int(slot[idx])
            sum_l[s] += load[idx]
            sum_r[s] += rain[idx]
            cnt[s] += 1.0
        cnt_safe = np.maximum(cnt, 1.0)
        means_l = np.where(cnt > 0, sum_l / cnt_safe, g_l)
        means_r = np.where(cnt > 0, sum_r / cnt_safe, g_r)

        last_load = float(d.at[t, "load_mw"])
        last_rain = float(d.at[t, "rain_mm"])
        roll_out = float(outage[max(0, t - 24) : t].mean()) if t > 0 else 0.0
        roll_rain = float(rain[max(0, t - 72) : t].mean()) if t > 0 else 0.0

        rows: List[Dict[str, float]] = []
        for h in range(1, 25):
            j = t + h
            if j >= n:
                break
            sj = int(slot[j])
            rows.append(
                {
                    "h": float(h),
                    "hour_sin": float(d.at[j, "hour_sin"]),
                    "hour_cos": float(d.at[j, "hour_cos"]),
                    "dow_sin": float(d.at[j, "dow_sin"]),
                    "dow_cos": float(d.at[j, "dow_cos"]),
                    "prof_load": float(means_l[sj]),
                    "prof_rain": float(means_r[sj]),
                    "last_load": last_load,
                    "last_rain": last_rain,
                    "roll_out_24": roll_out,
                    "roll_rain_72": roll_rain,
                }
            )
        return pd.DataFrame(rows)

    def predict_next_24h(self, df: pd.DataFrame, origin_idx: int) -> List[Dict[str, Any]]:
        if self._clf is None:
            raise RuntimeError("Call fit() before predict_next_24h().")

        t0 = time.perf_counter()
        Xb = self._features_for_horizon_block(df, origin_idx)
        p = self._clf.predict_proba(Xb)[:, 1].astype(np.float64)
        pq_lo = self._p_q_lo.predict(Xb).astype(np.float64)
        pq_hi = self._p_q_hi.predict(Xb).astype(np.float64)
        p_lo, p_hi = _order_prob_band(p, pq_lo, pq_hi)

        if self._dur is not None:
            log_mid = self._dur.predict(Xb).astype(np.float64)
            log_lo = self._dur_q_lo.predict(Xb).astype(np.float64)
            log_hi = self._dur_q_hi.predict(Xb).astype(np.float64)
            edur = np.expm1(np.clip(log_mid, 0.0, 8.0))
            ed_lo = np.expm1(np.clip(log_lo, 0.0, 8.0))
            ed_hi = np.expm1(np.clip(log_hi, 0.0, 8.0))
            ed_lo, ed_hi = _order_duration_band(edur, ed_lo, ed_hi)
        else:
            fb = float(self._meta["duration_fallback"])
            edur = np.full(len(Xb), fb, dtype=np.float64)
            ed_lo = ed_hi = edur.copy()

        d = self._add_calendar(df).reset_index(drop=True)
        out: List[Dict[str, Any]] = []
        for k, h in enumerate(range(1, len(Xb) + 1)):
            ts = d.at[origin_idx + h, "timestamp"]
            pm = float(np.clip(p[k], 1e-6, 1.0 - 1e-6))
            out.append(
                {
                    "timestamp": pd.Timestamp(ts).isoformat(),
                    "p_outage": pm,
                    "p_outage_low": float(np.clip(p_lo[k], 0.0, 1.0)),
                    "p_outage_high": float(np.clip(p_hi[k], 0.0, 1.0)),
                    "e_duration_min": float(max(5.0, edur[k])),
                    "e_duration_low_min": float(ed_lo[k]),
                    "e_duration_high_min": float(ed_hi[k]),
                }
            )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._meta["last_predict_ms"] = dt_ms
        return out

    def latency_ms(self) -> float:
        return float(self._meta.get("last_predict_ms", 0.0))


def load_grid_csv(path: Path | str = ROOT / "grid_history.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = _ensure_utc_ts(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)
