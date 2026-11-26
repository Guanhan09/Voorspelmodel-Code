"""
Experiment 2, stap A: handelssimulatie over 2025 met €10.000 fictief
startkapitaal per aandeel en vijf strategieën:
  - Voorspelmodel (Ensemble C): koop bij voorspelling Up, verkoop bij Down
  - RSI (koop bij kruising onder 30, verkoop bij kruising boven 70)
  - MACD (koop/verkoop bij kruising van de nullijn)
  - Stochastic Oscillator (%K kruist %D onder 20 = koop, boven 80 = verkoop)
  - Buy-and-hold (koop op de eerste, verkoop op de laatste handelsdag van 2025)

Gebruikt pws_exp2_testdata.pkl uit excel_exp1_voorspellingen.py: daarin staan
per aandeel de koersen en signalen van 2025 en de kansen van Ensemble C, dat
is getraind op data t/m 31-12-2024. Er wordt hier niets getraind.
Long-only, geen transactiekosten; een open positie wordt op de laatste
handelsdag van 2025 gesloten.

Output: pws_exp2_transacties.pkl -> excel_exp2_schrijven.py
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pws_experiment_2025 as E  # noqa: E402

TESTDATA_PATH = os.path.join(HERE, "pws_exp2_testdata.pkl")
OUT_XLSX = os.path.join(HERE, "..", "experiment2_resultaten.xlsx")
CACHE = os.path.join(HERE, "pws_exp2_transacties.pkl")
START_BEDRAG = 10_000.0

STRAT_NL = {"ML_Ensemble": "Voorspelmodel (Ensemble C)", "RSI": "RSI", "MACD": "MACD",
            "Stochastic": "Stochastic Oscillator", "BuyAndHold": "Buy-and-hold"}
STRAT_VOLGORDE = ["ML_Ensemble", "RSI", "MACD", "Stochastic", "BuyAndHold"]


def simuleer() -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    with open(TESTDATA_PATH, "rb") as f:
        d = pickle.load(f)
    test, top3, skipped = d["test"], d["top3"], d["skipped"]
    print(f"{len(test)} aandelen | Ensemble C = {top3}", flush=True)

    trades_rows, detail_rows = [], []
    for ticker, tf in test.items():
        dates, close = tf["dates"], tf["close"]
        pred = pd.Series(np.argmax(tf["proba_c"], axis=1), index=dates)
        per_strat = {
            "ML_Ensemble": E.simulate_trades(dates, close, pred == 2, pred == 0),
            "RSI": E.simulate_trades(dates, close, tf["buy_rsi"], tf["sell_rsi"]),
            "MACD": E.simulate_trades(dates, close, tf["buy_macd"], tf["sell_macd"]),
            "Stochastic": E.simulate_trades(dates, close, tf["buy_stoch"], tf["sell_stoch"]),
            "BuyAndHold": [{"entry_date": dates[0], "exit_date": dates[-1],
                            "entry_price": float(close.iloc[0]), "exit_price": float(close.iloc[-1]),
                            "return_pct": (float(close.iloc[-1]) - float(close.iloc[0])) / float(close.iloc[0]) * 100,
                            "forced_close": False}],
        }
        for strat, trades in per_strat.items():
            st = E.trade_stats(trades)
            detail_rows.append({"ticker": ticker, "strategie": strat, **st,
                                "eindwaarde": START_BEDRAG * (1 + st["total_return_pct"] / 100)})
            waarde = START_BEDRAG
            for nr, t in enumerate(trades, 1):
                waarde *= (1 + t["return_pct"] / 100)   # hele kapitaal gaat elke trade mee
                trades_rows.append({"strategie": strat, "ticker": ticker, "nr": nr, **t, "waarde_na": waarde})
    return pd.DataFrame(trades_rows), pd.DataFrame(detail_rows), skipped, top3


def summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in STRAT_VOLGORDE:
        sub = detail[detail["strategie"] == s]
        rows.append({"Strategie": STRAT_NL[s], "Aantal aandelen": len(sub),
                     "Gemiddelde eindwaarde (€)": sub["eindwaarde"].mean(),
                     "Gemiddeld rendement (%)": sub["total_return_pct"].mean(),
                     "Mediaan (%)": sub["total_return_pct"].median(),
                     "Percentage positief (%)": (sub["total_return_pct"] > 0).mean() * 100,
                     "Gemiddeld aantal trades": sub["n_trades"].mean(),
                     "Gem. winnende trades": sub["wins"].mean(),
                     "Gem. verliezende trades": sub["losses"].mean()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.time()
    trades, detail, skipped, top3 = simuleer()
    with open(CACHE, "wb") as f:
        pickle.dump({"trades": trades, "detail": detail, "skipped": skipped, "top3": top3}, f)
    print(f"Klaar ({time.time() - t0:.0f}s). Overgeslagen: {skipped}", flush=True)
    print(summary(detail).round(2).to_string(index=False))
