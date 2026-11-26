from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize(window: pd.Series) -> np.ndarray:
    base = window.iloc[0]
    if base == 0:
        return np.zeros(len(window))
    return ((window - base) / base).values


def find_similar_days(df: pd.DataFrame, lookback: int = 10, top_n: int = 5) -> list[dict]:
    closes = df["Close"].astype(float)
    if len(closes) < lookback + 2:
        return []

    current_window = _normalize(closes.iloc[-(lookback + 1) : -1])

    candidates = []
    # Search in the past 2 years (504 trading days), excluding recent 30 days
    search_end = len(closes) - lookback - 30
    search_start = max(0, len(closes) - 504 - lookback)

    for i in range(search_start, search_end):
        window = closes.iloc[i : i + lookback]
        if len(window) < lookback:
            continue
        norm = _normalize(window)
        dist = float(np.sqrt(np.mean((norm - current_window) ** 2)))
        candidates.append((i + lookback, dist))

    candidates.sort(key=lambda x: x[1])
    results = []

    for idx, dist in candidates[:top_n]:
        if idx >= len(closes) - 1:
            continue
        date_of_match = str(closes.index[idx].date())
        next_close = float(closes.iloc[idx + 1]) if idx + 1 < len(closes) else None
        match_close = float(closes.iloc[idx])
        if next_close is not None and match_close > 0:
            next_day_return = round((next_close - match_close) / match_close * 100, 2)
        else:
            next_day_return = None
        results.append({
            "date": date_of_match,
            "similarity": round(1 - min(dist, 1), 2),
            "next_day_return_pct": next_day_return,
            "close_at_match": round(match_close, 4),
        })

    return results
