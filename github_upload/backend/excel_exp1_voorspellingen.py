"""
Experiment 1, stap A: voorspellingen van de vier modellen en de drie ensembles
op de S&P 500.

Methode:
- Koersdata van 01-01-2015 t/m 31-12-2024 om te trainen, 2025 om te testen.
  De splitsing gaat op datum. Een trainingsrij telt alleen mee als ook zijn
  label (koers 5 handelsdagen later) op of vóór 31-12-2024 ligt, zodat het
  model niets uit de testperiode kent.
- Alle features zijn causaal (pws_experiment_2025.build_feature_frame): een
  waarde op datum X gebruikt alleen koersdata t/m X. Ook indicator 41,
  sectormomentum (sector_relative_strength_20d), gebruikt het 20-daagse momentum
  van de sector-ETF op die dag, niet de huidige live waarde.
- Label: 0 = Down (< -2%), 1 = Flat, 2 = Up (> +2%) over 5 handelsdagen.
- Ensemble A = gemiddelde kansen RF + GB, Ensemble B = gemiddelde van alle 4,
  Ensemble C = gemiddelde van de 3 modellen met de hoogste top-20%-accuracy.

Output (in deze map):
- pws_exp1_voorspellingen.npz : kansen per model per testrij -> excel_exp1_schrijven.py
- pws_exp2_testdata.pkl       : 2025-koersen, signalen en Ensemble C-kansen per aandeel
                                -> excel_exp2_transacties.py
- pws_experiment_models/      : de getrainde modellen

Gebruik: python excel_exp1_voorspellingen.py [--limit N]   (--limit alleen om te testen)
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pws_experiment_2025 as E  # noqa: E402
from data import SECTOR_ETF_MAP  # noqa: E402

NPZ_PATH = os.path.join(HERE, "pws_exp1_voorspellingen.npz")
TESTDATA_PATH = os.path.join(HERE, "pws_exp2_testdata.pkl")
BASE = ["LogisticRegression", "RandomForest", "GradientBoosting", "LightGBM"]


def top20_acc(proba: np.ndarray, y: np.ndarray) -> float:
    """Accuracy over de 20% voorspellingen met de hoogste zekerheid (max. kans)."""
    conf = proba.max(axis=1)
    hc = conf >= np.quantile(conf, 0.80)
    return float((proba.argmax(axis=1)[hc] == y[hc]).mean())


def marktdata(tickers: list[str]):
    spy_df = E.batch_download(["SPY"], E.FETCH_START, E.FETCH_END, chunk_size=1).get("SPY")
    if spy_df is None or spy_df.empty:
        sys.exit("FOUT: SPY-data kon niet worden opgehaald.")
    spy_close = spy_df["Close"]
    spy_ma50, spy_ma200 = spy_close.rolling(50).mean(), spy_close.rolling(200).mean()
    regime = pd.Series("neutral", index=spy_close.index)
    regime[(spy_close > spy_ma200) & (spy_ma50 > spy_ma200)] = "bull"
    regime[(spy_close < spy_ma200) & (spy_ma50 < spy_ma200)] = "bear"
    spy_aux = {"mom5": spy_close.pct_change(5) * 100, "mom20": spy_close.pct_change(20) * 100, "regime": regime}

    # Indicator 41: historische 20-daagse koersverandering van de sector-ETF, per dag.
    sector_etfs = sorted(set(SECTOR_ETF_MAP.values()))
    sector_data = E.batch_download(sector_etfs, E.FETCH_START, E.FETCH_END, chunk_size=len(sector_etfs))
    sector_mom20 = {etf: d["Close"].pct_change(20) * 100 for etf, d in sector_data.items()}
    sector_series_map = {t: sector_mom20.get(SECTOR_ETF_MAP.get(t)) for t in tickers}

    macro = E.batch_download(["^VIX", "^TNX"], E.FETCH_START, E.FETCH_END, chunk_size=2)
    vix_s = macro["^VIX"]["Close"] if "^VIX" in macro else None
    tnx_s = macro["^TNX"]["Close"] if "^TNX" in macro else None
    dxy_df = E.batch_download(["DX-Y.NYB"], E.FETCH_START, E.FETCH_END, chunk_size=1).get("DX-Y.NYB")
    if dxy_df is None:
        dxy_df = E.batch_download(["DX=F"], E.FETCH_START, E.FETCH_END, chunk_size=1).get("DX=F")
    dxy_chg_s = (dxy_df["Close"].pct_change(5) * 100) if dxy_df is not None else None
    return spy_aux, sector_series_map, vix_s, tnx_s, dxy_chg_s


def signalen(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Koop/verkoop-signalen van RSI, MACD en Stochastic Oscillator (experiment 2)."""
    closes = df["Close"]
    rsi = E.compute_rsi_series(closes)
    low14, high14 = df["Low"].rolling(14).min(), df["High"].rolling(14).max()
    stoch_k = (closes - low14) / (high14 - low14).replace(0, np.nan) * 100
    stoch_d = stoch_k.rolling(3).mean()
    cross_up = (stoch_k.shift(1) < stoch_d.shift(1)) & (stoch_k >= stoch_d)
    cross_down = (stoch_k.shift(1) > stoch_d.shift(1)) & (stoch_k <= stoch_d)
    macd_line, _, _ = E.compute_macd_series(closes)
    return {
        "buy_rsi": ((rsi < 30) & (rsi.shift(1) >= 30)).fillna(False),
        "sell_rsi": ((rsi > 70) & (rsi.shift(1) <= 70)).fillna(False),
        "buy_macd": ((macd_line > 0) & (macd_line.shift(1) <= 0)).fillna(False),
        "sell_macd": ((macd_line < 0) & (macd_line.shift(1) >= 0)).fillna(False),
        "buy_stoch": (cross_up & (stoch_k < 20)).fillna(False),
        "sell_stoch": (cross_down & (stoch_k > 80)).fillna(False),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="alleen de eerste N tickers (testrun)")
    args = ap.parse_args()
    t0 = time.time()
    print(f"Training: {E.FETCH_START} t/m {E.TRAIN_CUTOFF.date()} | test: {E.TEST_START.date()} t/m {E.TEST_END.date()}")

    tickers = E.fetch_sp500_tickers()
    if args.limit:
        tickers = tickers[:args.limit]
    print(f"{len(tickers)} S&P 500-tickers", flush=True)

    spy_aux, sector_map, vix_s, tnx_s, dxy_s = marktdata(tickers)
    price_data = E.batch_download(tickers, E.FETCH_START, E.FETCH_END, chunk_size=E.CHUNK_SIZE)

    H = E.FORECAST_HORIZON
    train_rows: list[pd.DataFrame] = []
    test: dict[str, dict] = {}
    skipped: list[tuple[str, str]] = []
    for n, ticker in enumerate(tickers, 1):
        df = price_data.pop(ticker, None)
        if df is None or df.empty:
            skipped.append((ticker, "geen koersdata")); continue
        df = df.dropna(subset=["Close"]).sort_index()
        if len(df) < E.MIN_WINDOW + H + 5:
            skipped.append((ticker, f"te weinig historische data ({len(df)} rijen)")); continue
        if df.index.max() < E.TEST_START:
            skipped.append((ticker, "geen koersdata in 2025")); continue

        feats_raw, reden = None, "earnings-timeout (3x)"
        for _ in range(3):
            try:
                feats_raw = E.build_feature_frame(df, ticker, spy_aux, sector_map, vix_s, tnx_s, dxy_s)
                break
            except E.EarningsTimeout:
                time.sleep(2)
            except Exception as e:  # noqa: BLE001
                reden = f"featureberekening mislukt: {e}"
                break
        if feats_raw is None:
            skipped.append((ticker, reden)); continue

        feats = feats_raw.shift(1)   # features op datum X = bekend bij de slotkoers van de dag ervoor
        closes = df["Close"]
        price_now = closes.shift(1)
        fwd = (closes.shift(-(H - 1)) - price_now) / price_now * 100
        target = pd.Series(1, index=df.index)
        target[fwd > E.TARGET_THRESHOLD] = 2
        target[fwd < -E.TARGET_THRESHOLD] = 0
        pos = np.arange(len(df))
        label_date = pd.Series(df.index, index=df.index).shift(-(H - 1))

        train_mask = ((pos >= E.MIN_WINDOW) & label_date.notna() & (label_date <= E.TRAIN_CUTOFF)
                      & feats.notna().all(axis=1) & fwd.notna())
        if train_mask.any():
            tr = feats.loc[train_mask].copy()
            tr["target"] = target.loc[train_mask].values
            train_rows.append(tr)

        test_mask = (df.index >= E.TEST_START) & (df.index <= E.TEST_END) & (pos >= E.MIN_WINDOW)
        if not test_mask.any():
            skipped.append((ticker, "geen bruikbare testrijen in 2025")); continue
        sig = signalen(df)
        # Label alleen bekend als de koers 5 handelsdagen later nog in 2025 ligt.
        heeft_label = (fwd.notna() & (label_date <= E.TEST_END)).loc[test_mask].values
        test[ticker] = {
            "dates": df.index[test_mask],
            "close": closes.loc[test_mask],
            "X": feats.loc[test_mask].fillna(0.0).values,
            "y": np.where(heeft_label, target.loc[test_mask].values, -1),
            **{k: v.loc[test_mask] for k, v in sig.items()},
        }
        if n % 50 == 0 or n == len(tickers):
            print(f"  {n}/{len(tickers)} tickers verwerkt", flush=True)

    del price_data
    train_df = pd.concat(train_rows, ignore_index=True)
    del train_rows
    gc.collect()
    print(f"\nTraining: {len(train_df):,} rijen (alle labels t/m {E.TRAIN_CUTOFF.date()}) | "
          f"test: {len(test)} aandelen | overgeslagen: {len(skipped)} {skipped}")
    for cls, naam in [(0, "Down"), (1, "Flat"), (2, "Up")]:
        print(f"  {naam}: {(train_df['target'] == cls).mean() * 100:.1f}%")

    fitted, scaler, clip_low, clip_high = E.train_ensemble(train_df)
    del train_df
    gc.collect()
    if list(fitted) != BASE:
        sys.exit(f"FOUT: verwachtte de modellen {BASE}, kreeg {list(fitted)} (is lightgbm geïnstalleerd?)")

    # ── Voorspellen op alle 2025-rijen ──
    order = list(test)
    X = np.vstack([test[t]["X"] for t in order])
    y = np.concatenate([test[t]["y"] for t in order])
    Xc = np.clip(X, clip_low, clip_high)
    Xs = scaler.transform(Xc)
    probas = {name: m.predict_proba(Xs if use_scaled else Xc) for name, (m, use_scaled) in fitted.items()}
    del X, Xc, Xs

    lab = y >= 0
    hc = {name: top20_acc(probas[name][lab], y[lab]) for name in BASE}
    top3 = sorted(BASE, key=lambda nm: hc[nm], reverse=True)[:3]
    probas["Ensemble_A (RF+GB)"] = 0.5 * probas["RandomForest"] + 0.5 * probas["GradientBoosting"]
    probas["Ensemble_B (alle)"] = sum(probas[nm] for nm in BASE) / 4
    probas["Ensemble_C (top-3)"] = sum(probas[nm] for nm in top3) / 3
    print(f"\nEnsemble C = {top3}")
    for name, p in probas.items():
        acc = (p[lab].argmax(1) == y[lab]).mean() * 100
        print(f"  {name:20s} overall {acc:5.2f}% | top-20% {top20_acc(p[lab], y[lab]) * 100:5.2f}%")

    # ── Experiment 1: alleen testrijen mét label ──
    tick = np.concatenate([[t] * len(test[t]["y"]) for t in order])
    dat = np.concatenate([test[t]["dates"].strftime("%Y-%m-%d").values for t in order])
    np.savez_compressed(
        NPZ_PATH,
        model_names=np.array(list(probas)),
        probas=np.stack([probas[nm][lab] for nm in probas]),
        y_test=y[lab], test_tickers=tick[lab].astype(str), test_dates=dat[lab].astype(str),
        top3=np.array(top3), skipped=np.array([f"{t}: {r}" for t, r in skipped], dtype=str),
    )

    # ── Experiment 2: Ensemble C-kansen voor álle 2025-handelsdagen per aandeel ──
    pc = probas["Ensemble_C (top-3)"]
    i = 0
    for t in order:
        k = len(test[t]["y"])
        test[t]["proba_c"] = pc[i:i + k]
        del test[t]["X"]
        i += k
    with open(TESTDATA_PATH, "wb") as f:
        pickle.dump({"test": test, "top3": top3, "skipped": skipped}, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nOpgeslagen: {NPZ_PATH}\n            {TESTDATA_PATH}\nLooptijd: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
