"""Backtesting module — test het voorspellingsysteem op historische data."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import fetch_ohlcv
from evaluation import DIRECTION_THRESHOLD, FORECAST_HORIZON, classify_direction
from indicators import compute_all, detect_market_regime
from predictor import predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEST_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "JPM", "V", "JNJ", "XOM", "TSLA",
]

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "backtest_results.json")

MIN_WINDOW = 210          # Minimaal 210 dagen nodig voor MA200 + buffer


def _actual_direction(change_pct: float) -> str:
    return classify_direction(change_pct)


def _extract_ind_signals(indicators: dict) -> dict:
    """Haal de richtingssignalen per indicator uit het indicators dict."""
    current = indicators.get("current_price", 0) or 0
    ma20    = indicators.get("ma20")
    ma50    = indicators.get("ma50")
    ma200   = indicators.get("ma200")
    rsi     = indicators.get("rsi") or 50
    macd    = indicators.get("macd") or {}
    mom5    = indicators.get("momentum_5d") or 0
    mom10   = indicators.get("momentum_10d") or 0
    ema_sig = indicators.get("ema_cross_signal")
    obv     = indicators.get("obv_trend")
    obv_c   = indicators.get("obv_confirmation")
    bb_pos  = indicators.get("bb_position")
    adx_s   = indicators.get("adx_strong_trend", False)
    adx_b   = indicators.get("adx_bullish", False)

    return {
        # MA signalen
        "ma20_bull":   bool(ma20 and current > ma20),
        "ma20_bear":   bool(ma20 and current < ma20),
        "ma50_bull":   bool(ma50 and current > ma50),
        "ma50_bear":   bool(ma50 and current < ma50),
        "ma200_bull":  bool(ma200 and current > ma200),
        "ma200_bear":  bool(ma200 and current < ma200),
        # RSI
        "rsi_oversold":   rsi < 30,
        "rsi_overbought": rsi > 70,
        # MACD
        "macd_bull": (macd.get("histogram") or 0) > 0,
        "macd_bear": (macd.get("histogram") or 0) < 0,
        # Momentum
        "mom5_bull": mom5 > 1,
        "mom5_bear": mom5 < -1,
        "mom10_bull": mom10 > 2,
        "mom10_bear": mom10 < -2,
        # EMA crossover
        "ema_bull": ema_sig == "bullish",
        "ema_bear": ema_sig == "bearish",
        # OBV
        "obv_bull": obv == "stijgend" and obv_c is True,
        "obv_bear": obv == "dalend"   and obv_c is True,
        # Bollinger
        "bb_oversold":   bb_pos is not None and bb_pos < 20,
        "bb_overbought": bb_pos is not None and bb_pos > 80,
        # ADX trend
        "adx_strong_bull": adx_s and adx_b,
        "adx_strong_bear": adx_s and not adx_b,
    }


def backtest_ticker(ticker: str, df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> list[dict]:
    """Backtest één ticker zonder toekomstdata.

    Voor elke dag i (>=MIN_WINDOW):
      - Window = df[0 .. i-1]  (data t/m dag i-1, slotkoers dag i NIET bekend)
      - Predicted: richting van dag i+4 t.o.v. dag i-1 slotkoers (5-daags)
      - Actual:    (dag i+4 close - dag i-1 close) / dag i-1 close
    """
    results = []
    total_days = len(df)

    if total_days < MIN_WINDOW + FORECAST_HORIZON + 1:
        logger.warning("%s: onvoldoende data (%d dagen)", ticker, total_days)
        return results

    for i in range(MIN_WINDOW, total_days - FORECAST_HORIZON):
        # Window bevat GEEN data van dag i of later — geen toekomst-lek
        window_df     = df.iloc[:i]
        current_close = float(df["Close"].iloc[i - 1])           # dag i-1: laatste bekende koers
        next_close    = float(df["Close"].iloc[i + FORECAST_HORIZON - 1])  # dag i+4: 5 dagen later

        # Regime: gebruik SPY data t/m dag i (geen toekomst-lek)
        regime = "neutral"
        if spy_df is not None:
            day_date = df.index[i]
            spy_window = spy_df[spy_df.index <= day_date]
            if len(spy_window) >= 200:
                regime = detect_market_regime(spy_window)

        try:
            indicators = compute_all(window_df)
            pred       = predict(indicators, window_df, regime=regime)
        except Exception as e:
            logger.debug("Voorspelling mislukt dag %d %s: %s", i, ticker, e)
            continue

        change_pct = (next_close - current_close) / current_close * 100
        actual     = _actual_direction(change_pct)
        correct    = pred["direction"] == actual

        results.append({
            "ticker":     ticker,
            "date":       str(df.index[i].date()),
            "predicted":  pred["direction"],
            "actual":     actual,
            "correct":    correct,
            "confidence": pred["confidence"],
            "change_pct": round(change_pct, 3),
            "score":      round(pred.get("score", 0.0), 2),
            "regime":     regime,
            "ind":        _extract_ind_signals(indicators),
        })

    logger.info("%s: %d dagen getest", ticker, len(results))
    return results


def _pct(correct: int, total: int) -> float:
    return round(correct / total * 100, 1) if total else 0.0


# Per indicator: (dict-sleutel in "ind", welke actual-richting is "correct")
_SIGNAL_DEFS: list[tuple[str, str, str]] = [
    # (label, ind_key, verwachte actual-richting bij dit signaal)
    ("MA20 bull → Up",         "ma20_bull",       "Up"),
    ("MA20 bear → Down",       "ma20_bear",       "Down"),
    ("MA50 bull → Up",         "ma50_bull",       "Up"),
    ("MA50 bear → Down",       "ma50_bear",       "Down"),
    ("MA200 bull → Up",        "ma200_bull",      "Up"),
    ("MA200 bear → Down",      "ma200_bear",      "Down"),
    ("RSI oversold → Up",      "rsi_oversold",    "Up"),
    ("RSI overbought → Down",  "rsi_overbought",  "Down"),
    ("MACD bull → Up",         "macd_bull",       "Up"),
    ("MACD bear → Down",       "macd_bear",       "Down"),
    ("Mom5 bull → Up",         "mom5_bull",       "Up"),
    ("Mom5 bear → Down",       "mom5_bear",       "Down"),
    ("EMA bull → Up",          "ema_bull",        "Up"),
    ("EMA bear → Down",        "ema_bear",        "Down"),
    ("OBV bull → Up",          "obv_bull",        "Up"),
    ("OBV bear → Down",        "obv_bear",        "Down"),
    ("BB oversold → Up",       "bb_oversold",     "Up"),
    ("BB overbought → Down",   "bb_overbought",   "Down"),
    ("ADX bull → Up",          "adx_strong_bull", "Up"),
    ("ADX bear → Down",        "adx_strong_bear", "Down"),
]


def analyze_results(all_results: list[dict]) -> dict:
    """Bereken alle accuracy metrics uit de ruwe backtest resultaten."""
    total = len(all_results)
    if total == 0:
        return {"error": "Geen resultaten beschikbaar"}

    correct_all = sum(1 for r in all_results if r["correct"])

    # Per richting
    per_direction: dict[str, dict] = {}
    for direction in ("Up", "Down", "Flat"):
        subset = [r for r in all_results if r["predicted"] == direction]
        c = sum(1 for r in subset if r["correct"])
        per_direction[direction] = {
            "total":        len(subset),
            "correct":      c,
            "accuracy_pct": _pct(c, len(subset)),
        }

    # Per confidence niveau
    buckets = {
        "laag":   [r for r in all_results if r["confidence"] < 55],
        "middel": [r for r in all_results if 55 <= r["confidence"] <= 65],
        "hoog":   [r for r in all_results if r["confidence"] > 65],
    }
    per_confidence: dict[str, dict] = {}
    labels = {"laag": "<55%", "middel": "55–65%", "hoog": ">65%"}
    for key, subset in buckets.items():
        c = sum(1 for r in subset if r["correct"])
        per_confidence[key] = {
            "range":        labels[key],
            "total":        len(subset),
            "correct":      c,
            "accuracy_pct": _pct(c, len(subset)),
        }

    # Per ticker
    per_ticker: dict[str, dict] = {}
    for t in sorted({r["ticker"] for r in all_results}):
        subset = [r for r in all_results if r["ticker"] == t]
        c = sum(1 for r in subset if r["correct"])
        per_ticker[t] = {
            "total":        len(subset),
            "correct":      c,
            "accuracy_pct": _pct(c, len(subset)),
        }

    # Indicator analyse — per signaal: hoe vaak klopte de markbeweging?
    indicator_analysis: dict[str, dict] = {}
    for label, ind_key, expected_dir in _SIGNAL_DEFS:
        subset = [r for r in all_results if r.get("ind", {}).get(ind_key) is True]
        if not subset:
            continue
        c = sum(1 for r in subset if r["actual"] == expected_dir)
        indicator_analysis[label] = {
            "appearances":  len(subset),
            "correct":      c,
            "accuracy_pct": _pct(c, len(subset)),
        }

    sorted_inds = sorted(
        indicator_analysis.items(),
        key=lambda x: x[1]["accuracy_pct"],
        reverse=True,
    )
    best_indicators  = [{"indicator": k, **v} for k, v in sorted_inds[:5]]
    worst_indicators = [{"indicator": k, **v} for k, v in sorted_inds[-5:]]

    # Per marktregime
    per_regime: dict[str, dict] = {}
    for reg in ("bull", "neutral", "bear"):
        subset = [r for r in all_results if r.get("regime") == reg]
        c = sum(1 for r in subset if r["correct"])
        per_regime[reg] = {
            "total":        len(subset),
            "correct":      c,
            "accuracy_pct": _pct(c, len(subset)),
        }

    return {
        "meta": {
            "tickers":            sorted({r["ticker"] for r in all_results}),
            "run_date":           datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total_predictions":  total,
            "threshold_pct":      DIRECTION_THRESHOLD,
        },
        "overall": {
            "total":        total,
            "correct":      correct_all,
            "accuracy_pct": _pct(correct_all, total),
        },
        "per_direction":    per_direction,
        "per_confidence":   per_confidence,
        "per_ticker":       per_ticker,
        "per_regime":       per_regime,
        "best_indicators":  best_indicators,
        "worst_indicators": worst_indicators,
        "indicator_detail": indicator_analysis,
    }


def run_backtest(tickers: list[str] = TEST_TICKERS) -> dict:
    """Haal data op en voer backtest uit voor alle tickers."""
    print("Drempel: ±3.0% over 5 handelsdagen")
    print("Horizon: 5 dagen vooruit")
    all_results: list[dict] = []

    # SPY data eenmalig ophalen voor regime detectie
    spy_df: pd.DataFrame | None = None
    try:
        spy_df = fetch_ohlcv("SPY", period="5y")
        logger.info("SPY data opgehaald: %d dagen", len(spy_df))
    except Exception as e:
        logger.warning("SPY data ophalen mislukt, regime detectie uitgeschakeld: %s", e)

    for ticker in tickers:
        logger.info("Bezig met %s...", ticker)
        try:
            df             = fetch_ohlcv(ticker, period="5y")
            ticker_results = backtest_ticker(ticker, df, spy_df=spy_df)
            all_results.extend(ticker_results)
        except Exception as e:
            logger.error("Fout bij %s: %s", ticker, e)

    results = analyze_results(all_results)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Resultaten opgeslagen: %s", RESULTS_PATH)

    return results


def print_results(results: dict) -> None:
    """Print backtest resultaten naar stdout."""
    meta    = results.get("meta", {})
    overall = results.get("overall", {})

    print("\n" + "=" * 60)
    print("--- 5-DAAGSE PERFORMANCE ---")
    print("BACKTEST RESULTATEN")
    print("=" * 60)
    print(f"Tickers:      {', '.join(meta.get('tickers', []))}")
    print(f"Datum:        {meta.get('run_date', '—')}")
    print(f"Drempel:      ±{meta.get('threshold_pct', '?')}%  (over {FORECAST_HORIZON} handelsdagen)")
    print(f"Horizon:      {FORECAST_HORIZON} dagen vooruit")
    print(f"Totaal:       {overall.get('total', 0)} voorspellingen")
    print(f"\nOVERALL ACCURACY: {overall.get('accuracy_pct', 0)}%")
    print(f"  Correct: {overall.get('correct', 0)} / {overall.get('total', 0)}")

    print("\n--- PER RICHTING ---")
    for d, v in results.get("per_direction", {}).items():
        print(f"  {d:5s}: {v['accuracy_pct']:5.1f}%  ({v['correct']}/{v['total']})")

    print("\n--- PER CONFIDENCE NIVEAU ---")
    for k, v in results.get("per_confidence", {}).items():
        print(f"  {k:7s} ({v['range']:8s}): {v['accuracy_pct']:5.1f}%  ({v['correct']}/{v['total']})")

    print("\n--- PER MARKTREGIME ---")
    for reg, v in results.get("per_regime", {}).items():
        print(f"  {reg:8s}: {v['accuracy_pct']:5.1f}%  ({v['correct']}/{v['total']})")

    print("\n--- PER TICKER ---")
    for t, v in sorted(results.get("per_ticker", {}).items(), key=lambda x: -x[1]["accuracy_pct"]):
        print(f"  {t:6s}: {v['accuracy_pct']:5.1f}%  ({v['correct']}/{v['total']})")

    print("\n--- BESTE INDICATOR SIGNALEN ---")
    for ind in results.get("best_indicators", []):
        print(f"  {ind['indicator']:25s}: {ind['accuracy_pct']:5.1f}%  ({ind['appearances']} signalen)")

    print("\n--- SLECHTSTE INDICATOR SIGNALEN ---")
    for ind in results.get("worst_indicators", []):
        print(f"  {ind['indicator']:25s}: {ind['accuracy_pct']:5.1f}%  ({ind['appearances']} signalen)")

    print("=" * 60)


if __name__ == "__main__":
    results = run_backtest(TEST_TICKERS)
    print_results(results)
