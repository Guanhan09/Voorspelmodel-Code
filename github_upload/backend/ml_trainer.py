"""ML trainer — bouw en evalueer ML-modellen op historische indicator-data.

Wijzigt de productie-app NIET. Alleen bedoeld om te meten of ML significant
beter scoort dan het huidige handmatige systeem (baseline ~34%).
"""
from __future__ import annotations

import io
import json
import logging
import os
import requests
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import compute_earnings_features, fetch_ohlcv, get_macro_history, get_spy_data
from indicators import compute_all, detect_market_regime

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def _read_wiki_tables(url: str):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def get_all_tickers() -> list[str]:
    tickers = set()

    # S&P 500
    try:
        tables = _read_wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp500 = tables[0]
        sp500_tickers = sp500["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        tickers.update(sp500_tickers)
        print(f"S&P 500: {len(sp500_tickers)} tickers toegevoegd")
    except Exception as e:
        print(f"S&P 500 ophalen mislukt: {e}")

    # NASDAQ 100
    try:
        tables = _read_wiki_tables("https://en.wikipedia.org/wiki/Nasdaq-100")
        nasdaq = None
        for t in tables:
            if "Ticker" in t.columns:
                nasdaq = t
                break
        if nasdaq is not None:
            nasdaq_tickers = nasdaq["Ticker"].astype(str).tolist()
            tickers.update(nasdaq_tickers)
            print(f"NASDAQ 100: {len(nasdaq_tickers)} tickers toegevoegd")
    except Exception as e:
        print(f"NASDAQ 100 ophalen mislukt: {e}")

    # AEX
    try:
        tables = _read_wiki_tables("https://en.wikipedia.org/wiki/AEX_index")
        aex = None
        for t in tables:
            if "Ticker" in t.columns:
                aex = t
                break
        if aex is not None:
            aex_tickers = [str(x) + ".AS" for x in aex["Ticker"].tolist()]
            tickers.update(aex_tickers)
            print(f"AEX: {len(aex_tickers)} tickers toegevoegd")
    except Exception as e:
        print(f"AEX ophalen mislukt: {e}")

    # DAX
    try:
        tables = _read_wiki_tables("https://en.wikipedia.org/wiki/DAX")
        dax = None
        for t in tables:
            if "Ticker" in t.columns:
                dax = t
                break
        if dax is not None:
            dax_tickers = [str(x) + ".DE" for x in dax["Ticker"].tolist()]
            tickers.update(dax_tickers)
            print(f"DAX: {len(dax_tickers)} tickers toegevoegd")
    except Exception as e:
        print(f"DAX ophalen mislukt: {e}")

    # CAC 40
    try:
        tables = _read_wiki_tables("https://en.wikipedia.org/wiki/CAC_40")
        cac = None
        for t in tables:
            if "Ticker" in t.columns:
                cac = t
                break
        if cac is not None:
            cac_tickers = [str(x) + ".PA" for x in cac["Ticker"].tolist()]
            tickers.update(cac_tickers)
            print(f"CAC 40: {len(cac_tickers)} tickers toegevoegd")
    except Exception as e:
        print(f"CAC 40 ophalen mislukt: {e}")

    # FTSE 100
    try:
        tables = _read_wiki_tables("https://en.wikipedia.org/wiki/FTSE_100_Index")
        ftse = None
        for t in tables:
            if "Ticker" in t.columns:
                ftse = t
                break
        if ftse is not None:
            ftse_tickers = [str(x) + ".L" for x in ftse["Ticker"].tolist()]
            tickers.update(ftse_tickers)
            print(f"FTSE 100: {len(ftse_tickers)} tickers toegevoegd")
    except Exception as e:
        print(f"FTSE 100 ophalen mislukt: {e}")

    # Verwijder dubbele suffixen (bv. "AKZA.AS.AS" -> "AKZA.AS") die ontstaan als
    # de Wikipedia-tabel de suffix al bevat én de code hem nog eens toevoegt.
    def _dedup_suffix(t: str) -> str:
        for suf in (".AS", ".DE", ".PA", ".L"):
            while t.endswith(suf + suf):
                t = t[: -len(suf)]
        return t

    result = sorted({_dedup_suffix(t) for t in tickers})
    print(f"\nTotaal unieke tickers: {len(result)}")
    return result

TEST_TICKERS = get_all_tickers()

FORECAST_HORIZON = 5     # 5 handelsdagen vooruit
TARGET_THRESHOLD = 2.0   # % drempel voor Up/Down classificatie
MIN_WINDOW = 210         # Minimaal voor MA200

DATASET_PATH  = os.path.join(os.path.dirname(__file__), "ml_dataset.csv")
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "ml_model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "ml_features.json")

FEATURE_NAMES = [
    "rsi_14",
    "rsi_oversold",
    "rsi_overbought",
    "macd_diff",
    "macd_bull",
    "macd_bear",
    "ma20_distance",
    "ma50_distance",
    "ma200_distance",
    "ma_alignment",
    "mom_5d",
    "mom_10d",
    "mom_20d",
    "volume_ratio",
    "volume_trend",
    "obv_slope",
    "adx_14",
    "adx_strong",
    "bb_position",
    "stoch_rsi_k",
    "atr_pct",
    "volatility_20d",
    "distance_to_52w_high",
    "distance_to_52w_low",
    "market_regime_bull",
    "market_regime_bear",
    "gap_pct",
    "candlestick_bullish",
    "candlestick_bearish",
    "relative_strength_5d",
    "relative_strength_20d",
    "sector_relative_strength_20d",
    "price_acceleration",
    "volume_anomaly",
    "volatility_regime",
    "trend_consistency",
    "rsi_trend",
    "macd_acceleration",
    "vix_level",
    "tnx_level",
    "dxy_change_5d",
    "days_since_last_earnings",
    "days_until_next_earnings",
    "earnings_zone",
]


def _macro_at(series: pd.Series | None, date) -> float:
    """Laatste beschikbare waarde van een macro-series op of vóór `date`.
    Return 0.0 als de series None is of er geen waarde beschikbaar is.
    """
    if series is None or len(series) == 0:
        return 0.0
    try:
        idx = series.index
        ts = pd.Timestamp(date)
        if getattr(idx, "tz", None) is not None:
            ts = ts.tz_localize(idx.tz) if ts.tzinfo is None else ts.tz_convert(idx.tz)
        mask = idx <= ts
        if not mask.any():
            return 0.0
        val = float(series[mask].iloc[-1])
        if np.isnan(val):
            return 0.0
        return val
    except Exception:
        return 0.0


def _dxy_change_5d_at(series: pd.Series | None, date) -> float:
    """5-daagse procentuele verandering van de dollar-index op/voor `date`."""
    if series is None or len(series) < 6:
        return 0.0
    try:
        idx = series.index
        ts = pd.Timestamp(date)
        if getattr(idx, "tz", None) is not None:
            ts = ts.tz_localize(idx.tz) if ts.tzinfo is None else ts.tz_convert(idx.tz)
        window = series[idx <= ts]
        if len(window) < 6:
            return 0.0
        start = float(window.iloc[-6])
        end = float(window.iloc[-1])
        if start == 0 or np.isnan(start) or np.isnan(end):
            return 0.0
        return round((end - start) / start * 100, 4)
    except Exception:
        return 0.0


def _obv_slope(closes: pd.Series, volumes: pd.Series, window: int = 10) -> float:
    """Lineaire helling van OBV over de laatste `window` dagen."""
    if len(closes) < window + 1:
        return 0.0
    direction = np.sign(closes.diff().fillna(0))
    obv = (direction * volumes).cumsum()
    obv_tail = obv.iloc[-window:].values.astype(float)
    x = np.arange(window, dtype=float)
    if obv_tail.std() == 0:
        return 0.0
    slope = float(np.polyfit(x, obv_tail, 1)[0])
    # Normaliseer door de gemiddelde absolute OBV-waarde te delen
    mean_abs = float(np.abs(obv_tail).mean()) or 1.0
    return round(slope / mean_abs, 6)


def _extract_features(indicators: dict, spy_regime: str) -> dict:
    """Zet een indicators-dict om naar een plat feature-dict."""
    ind = indicators
    current = ind.get("current_price", 1) or 1

    ma20  = ind.get("ma20")
    ma50  = ind.get("ma50")
    ma200 = ind.get("ma200")

    ma20_dist  = ((current - ma20)  / ma20  * 100) if ma20  else 0.0
    ma50_dist  = ((current - ma50)  / ma50  * 100) if ma50  else 0.0
    ma200_dist = ((current - ma200) / ma200 * 100) if ma200 else 0.0

    # MA alignment
    if ma20 and ma50 and ma200:
        if ma20 > ma50 > ma200:
            ma_align = 1
        elif ma20 < ma50 < ma200:
            ma_align = -1
        else:
            ma_align = 0
    else:
        ma_align = 0

    macd = ind.get("macd") or {}
    macd_diff = (macd.get("macd", 0) or 0) - (macd.get("signal", 0) or 0)
    hist = macd.get("histogram", 0) or 0

    rsi = ind.get("rsi", 50) or 50
    atr = ind.get("atr") or 0
    atr_pct = (atr / current * 100) if current else 0.0

    bb_pos = ind.get("bb_position")
    # Normaliseer bb_position naar 0-1 bereik
    bb_pos_norm = (bb_pos / 100.0) if bb_pos is not None else 0.5

    candle_patterns = ind.get("candle_patterns") or []
    bullish_pats = {"hammer", "bullish_engulfing", "marubozu_bullish"}
    bearish_pats = {"shooting_star", "bearish_engulfing", "marubozu_bearish"}

    vol_trend_ratio = ind.get("volume_trend_ratio", 1.0) or 1.0

    # ── DEEL 2: nieuwe features ──
    mom_5d  = ind.get("momentum_5d", 0) or 0
    mom_10d = ind.get("momentum_10d", 0) or 0
    mom_20d = ind.get("momentum_20d", 0) or 0
    spy_5   = ind.get("spy_momentum_5d", 0) or 0
    spy_20  = ind.get("spy_momentum_20d", 0) or 0
    vol_20  = ind.get("volatility_20d", 0) or 0
    vol_60  = ind.get("volatility_60d", 0) or 0
    vol_ratio = ind.get("volume_ratio", 1.0) or 1.0

    rel_str_5d        = mom_5d - spy_5
    rel_str_20d       = mom_20d - spy_20
    price_accel       = mom_5d - mom_10d
    volume_anomaly    = float(np.clip(vol_ratio - 1.0, -2, 2))
    volatility_regime = vol_20 / max(vol_60, 0.001)

    sector_mom_20 = ind.get("sector_momentum_20d", 0) or 0
    mom_20d_for_sector = ind.get("momentum_20d", 0) or 0
    sector_relative_strength_20d = mom_20d_for_sector - sector_mom_20

    vix_level     = float(ind.get("vix_level", 0.0) or 0.0)
    tnx_level     = float(ind.get("tnx_level", 0.0) or 0.0)
    dxy_change_5d = float(ind.get("dxy_change_5d", 0.0) or 0.0)

    tc = ind.get("trend_consistency")
    trend_consistency = 0.5 if tc is None else float(tc)
    rsi_trend         = float(ind.get("rsi_trend", 0.0) or 0.0)
    macd_accel        = float(ind.get("macd_acceleration", 0.0) or 0.0)

    return {
        "rsi_14":               rsi,
        "rsi_oversold":         float(rsi < 30),
        "rsi_overbought":       float(rsi > 70),
        "macd_diff":            macd_diff,
        "macd_bull":            float(hist > 0),
        "macd_bear":            float(hist < 0),
        "ma20_distance":        ma20_dist,
        "ma50_distance":        ma50_dist,
        "ma200_distance":       ma200_dist,
        "ma_alignment":         float(ma_align),
        "mom_5d":               ind.get("momentum_5d", 0) or 0,
        "mom_10d":              ind.get("momentum_10d", 0) or 0,
        "mom_20d":              ind.get("momentum_20d", 0) or 0,
        "volume_ratio":         ind.get("volume_ratio", 1.0) or 1.0,
        "volume_trend":         vol_trend_ratio,
        "obv_slope":            0.0,  # wordt apart ingevuld
        "adx_14":               ind.get("adx", 0) or 0,
        "adx_strong":           float(ind.get("adx_strong_trend", False)),
        "bb_position":          bb_pos_norm,
        "stoch_rsi_k":          (ind.get("stoch_rsi_k") or 50) / 100.0,
        "atr_pct":              atr_pct,
        "volatility_20d":       ind.get("volatility_20d", 0) or 0,
        "distance_to_52w_high": ind.get("pct_from_52w_high", 0) or 0,
        "distance_to_52w_low":  ind.get("pct_from_52w_low", 0) or 0,
        "market_regime_bull":   float(spy_regime == "bull"),
        "market_regime_bear":   float(spy_regime == "bear"),
        "gap_pct":              ind.get("gap_pct", 0) or 0,
        "candlestick_bullish":  float(any(p in bullish_pats for p in candle_patterns)),
        "candlestick_bearish":  float(any(p in bearish_pats for p in candle_patterns)),
        "relative_strength_5d":  rel_str_5d,
        "relative_strength_20d": rel_str_20d,
        "sector_relative_strength_20d": sector_relative_strength_20d,
        "price_acceleration":    price_accel,
        "volume_anomaly":        volume_anomaly,
        "volatility_regime":     volatility_regime,
        "trend_consistency":     trend_consistency,
        "rsi_trend":             rsi_trend,
        "macd_acceleration":     macd_accel,
        "vix_level":             vix_level,
        "tnx_level":             tnx_level,
        "dxy_change_5d":         dxy_change_5d,
    }


def _target_class(forward_return_pct: float) -> int:
    """2 = Up, 0 = Down, 1 = Flat."""
    if forward_return_pct > TARGET_THRESHOLD:
        return 2
    if forward_return_pct < -TARGET_THRESHOLD:
        return 0
    return 1


def build_dataset(tickers: list[str] = TEST_TICKERS) -> pd.DataFrame:
    """Bouw de ML dataset van alle tickers en sla op als CSV."""
    spy_df = get_spy_data()
    macro = get_macro_history() or {}
    vix_series = macro.get("vix")
    tnx_series = macro.get("tnx")
    dxy_series = macro.get("dxy")
    rows: list[dict] = []

    for ticker in tickers:
        logger.info("Features berekenen voor %s...", ticker)
        try:
            df = fetch_ohlcv(ticker, period="5y")
        except Exception as e:
            logger.error("Data ophalen mislukt voor %s: %s", ticker, e)
            continue

        total = len(df)
        if total < MIN_WINDOW + FORECAST_HORIZON + 1:
            logger.warning("%s: onvoldoende data (%d rijen)", ticker, total)
            continue

        closes  = df["Close"]
        volumes = df["Volume"]

        for i in range(MIN_WINDOW, total - FORECAST_HORIZON):
            window_df = df.iloc[:i]

            try:
                # Bewuste vereenvoudiging: de sector-ETF momentum dat hieruit volgt
                # is het meest recente (live) 20d-momentum van de sector-ETF, niet
                # de historische waarde op dag i. Voor dataset-opbouw is dit een
                # benadering — exacte historische sector-prijzen zouden vergen dat
                # _get_sector_etf_data ook een datum-cutoff ondersteunt.
                indicators = compute_all(window_df, ticker=ticker)
            except Exception:
                continue

            # OBV slope apart berekenen
            if len(window_df) >= 11:
                obv_sl = _obv_slope(closes.iloc[:i], volumes.iloc[:i])
            else:
                obv_sl = 0.0

            # Marktregime op basis van SPY tot dag i
            regime = "neutral"
            if spy_df is not None:
                spy_window = spy_df[spy_df.index <= df.index[i]]
                if len(spy_window) >= 200:
                    regime = detect_market_regime(spy_window)

            feats = _extract_features(indicators, regime)
            feats["obv_slope"] = obv_sl

            # Historisch correcte macro-waarden op df.index[i]: laatste beschikbare
            # waarde op of vóór die datum (ffill).
            sample_date = df.index[i]
            feats["vix_level"]     = _macro_at(vix_series, sample_date)
            feats["tnx_level"]     = _macro_at(tnx_series, sample_date)
            feats["dxy_change_5d"] = _dxy_change_5d_at(dxy_series, sample_date)

            # Earnings-nabijheid t.o.v. de sample-datum. De kalender wordt per
            # ticker eenmalig gecachet (zie data.get_earnings_calendar), dus dit
            # is per rij slechts een lookup. Gaten-bestendig: nooit NaN.
            feats.update(compute_earnings_features(ticker, sample_date))

            # Target: 5-daags forward return
            price_now  = float(closes.iloc[i - 1])
            price_then = float(closes.iloc[i + FORECAST_HORIZON - 1])
            fwd_return = (price_then - price_now) / price_now * 100
            target     = _target_class(fwd_return)

            feats["ticker"]     = ticker
            feats["date"]       = str(df.index[i].date())
            feats["fwd_return"] = round(fwd_return, 3)
            feats["target"]     = target
            rows.append(feats)

        logger.info("%s: %d samples", ticker, sum(1 for r in rows if r["ticker"] == ticker))

    dataset = pd.DataFrame(rows)
    dataset.to_csv(DATASET_PATH, index=False)
    logger.info("Dataset opgeslagen: %s (%d rijen)", DATASET_PATH, len(dataset))

    # Directe backup zodra de (dure) dataset er is, zodat een latere crash of een
    # REUSE_DATASET=0-run hem niet als enige kopie wist.
    try:
        backup_path = DATASET_PATH + ".bak_earnings"
        dataset.to_csv(backup_path, index=False)
        logger.info("Dataset-backup geschreven: %s", backup_path)
    except Exception as e:
        logger.warning("Dataset-backup schrijven mislukt: %s", e)

    # Statistieken
    print(f"\nDataset: {len(dataset)} samples")
    label_map = {0: "Down", 1: "Flat", 2: "Up"}
    for cls, name in label_map.items():
        n = (dataset["target"] == cls).sum()
        print(f"  {name}: {n} ({n / len(dataset) * 100:.1f}%)")

    return dataset


def train_and_evaluate(tickers: list[str] = TEST_TICKERS) -> None:
    """Bouw dataset, train 3 modellen, evalueer en sla het beste op."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.frozen import FrozenEstimator
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    import joblib

    # ── Verwijder oude gecachede dataset ─────────────────────────────────
    csv_path = Path(__file__).parent / "ml_dataset.csv"
    if csv_path.exists() and os.environ.get("REUSE_DATASET") != "1":
        os.remove(csv_path)
        print("Oude dataset verwijderd, opnieuw berekenen...")

    # ── Dataset bouwen of laden ──────────────────────────────────────────
    if os.path.exists(DATASET_PATH):
        print(f"\nBestaande dataset gevonden: {DATASET_PATH}")
        print("Laden... (verwijder ml_dataset.csv om opnieuw te bouwen)")
        dataset = pd.read_csv(DATASET_PATH)
    else:
        print("\nDataset bouwen (dit duurt 15-30 minuten)...")
        dataset = build_dataset(TEST_TICKERS)

    if len(dataset) < 100:
        print("Te weinig data om te trainen.")
        return

    # ── Features en target ───────────────────────────────────────────────
    feature_cols = [c for c in FEATURE_NAMES if c in dataset.columns]
    X = dataset[feature_cols].fillna(0).values
    y = dataset["target"].values
    dates = pd.to_datetime(dataset["date"])

    # ── Tijdgebaseerde train/calibratie/test split (geen shuffle) ──
    # Calibratie krijgt een eigen hold-out slice, disjunct van de train-fold
    # waarop het model zelf gefit wordt — voorkomt dat CalibratedClassifierCV
    # kalibreert op (een deel van) dezelfde data die het model al gezien heeft.
    split_train = dates.quantile(0.60)
    split_calib = dates.quantile(0.80)
    train_mask = dates <= split_train
    calib_mask = (dates > split_train) & (dates <= split_calib)
    test_mask  = dates >  split_calib

    X_train, y_train = X[train_mask], y[train_mask]
    X_calib, y_calib = X[calib_mask], y[calib_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"\nTrain: {len(X_train)} samples  |  Calibratie: {len(X_calib)} samples  |  Test: {len(X_test)} samples")
    print(f"Split train/calibratie: {split_train.date()}  |  Split calibratie/test: {split_calib.date()}")

    # ── Clip alle features op ±5 standaarddeviaties (op basis van train) ──
    feat_mean = X_train.mean(axis=0)
    feat_std  = X_train.std(axis=0)
    clip_low  = feat_mean - 5 * feat_std
    clip_high = feat_mean + 5 * feat_std
    X_train = np.clip(X_train, clip_low, clip_high)
    X_calib = np.clip(X_calib, clip_low, clip_high)
    X_test  = np.clip(X_test,  clip_low, clip_high)

    # ── Normaliseer features ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_calib_s = scaler.transform(X_calib)
    X_test_s  = scaler.transform(X_test)

    # ── Baseline: altijd "Flat" (class 1) voorspellen ────────────────────
    baseline_preds = np.ones(len(y_test), dtype=int)
    baseline_acc   = accuracy_score(y_test, baseline_preds)
    label_map = {0: "Down", 1: "Flat", 2: "Up"}
    print(f"\n{'=' * 60}")
    print(f"BASELINE (altijd Flat): {baseline_acc * 100:.1f}%")
    print(f"Een model moet >{baseline_acc * 100:.1f}% halen om beter dan baseline te zijn")
    print("De TOP-20% CONFIDENCE accuracy is het belangrijkste getal "
          "— als dat boven 65% komt heeft het model echte voorspelkracht")

    # ── Modellen ─────────────────────────────────────────────────────────
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=42,
        ),
    }

    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            subsample=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        print("WAARSCHUWING: lightgbm niet beschikbaar — sla LightGBM-model over "
              "(installeer met: venv\\Scripts\\pip install lightgbm)")

    results: dict[str, dict] = {}

    for name, model in models.items():
        print(f"\n{'─' * 60}")
        print(f"Model: {name}")

        # Train
        use_scaled = name == "LogisticRegression"
        Xtr  = X_train_s if use_scaled else X_train
        Xcal = X_calib_s if use_scaled else X_calib
        Xte  = X_test_s  if use_scaled else X_test

        logger.info("Trainen %s...", name)
        model.fit(Xtr, y_train)

        # Predict
        preds = model.predict(Xte)
        proba = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, preds)
        print(f"Overall accuracy: {acc * 100:.1f}%  (baseline: {baseline_acc * 100:.1f}%)")

        # Per class accuracy
        print("Per klasse:")
        for cls, lbl in label_map.items():
            mask = y_test == cls
            if mask.sum() == 0:
                continue
            cls_acc = accuracy_score(y_test[mask], preds[mask])
            print(f"  {lbl:5s}: {cls_acc * 100:.1f}%  (n={mask.sum()})")

        # Confidence calibratie — top 20% meest zekere voorspellingen
        if proba is not None:
            max_conf = proba.max(axis=1)
            threshold_conf = np.quantile(max_conf, 0.80)
            high_conf_mask = max_conf >= threshold_conf
            if high_conf_mask.sum() > 0:
                hc_acc = accuracy_score(y_test[high_conf_mask], preds[high_conf_mask])
                print(f"Top-20% confidence accuracy: {hc_acc * 100:.1f}%  (n={high_conf_mask.sum()})")
            else:
                hc_acc = acc
        else:
            hc_acc = acc

        # Bewaar het ongekalibreerde model voor feature-importance (de
        # CalibratedClassifierCV-wrapper heeft geen feature_importances_/coef_).
        base_model = model

        # ── Kalibreer ALLE modellen die predict_proba ondersteunen, behalve
        #    LogisticRegression (die is al gekalibreerd). Gebruikt de losse
        #    calibratie-hold-out (Xcal/y_calib) — niet Xtr/y_train — zodat
        #    de isotonic-fit nooit data ziet waarop het model zelf getraind is. ──
        hc_acc_voor = hc_acc
        if name != "LogisticRegression" and hasattr(model, "predict_proba"):
            calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
            calibrated.fit(Xcal, y_calib)
            cal_preds = calibrated.predict(Xte)
            cal_proba = calibrated.predict_proba(Xte)
            cal_max_conf = cal_proba.max(axis=1)
            cal_threshold = np.quantile(cal_max_conf, 0.80)
            cal_hc_mask = cal_max_conf >= cal_threshold
            if cal_hc_mask.sum() > 0:
                hc_acc = accuracy_score(y_test[cal_hc_mask], cal_preds[cal_hc_mask])
            print(f"  Top-20% conf VOOR calibratie: {hc_acc_voor * 100:.1f}%  NA calibratie: {hc_acc * 100:.1f}%")
            model = calibrated

        # Finale (eventueel gekalibreerde) voorspellingen + kansen op de testset
        final_preds = model.predict(Xte)
        final_proba = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None

        # Feature importance — gebruik het ongekalibreerde basismodel, met
        # hasattr-checks zodat dit nooit crasht op gekalibreerde modellen.
        if hasattr(base_model, "feature_importances_"):
            importances = base_model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            print("Top-10 features:")
            for rank, idx in enumerate(top_idx, 1):
                fname = feature_cols[idx] if idx < len(feature_cols) else f"feat_{idx}"
                print(f"  {rank:2d}. {fname:25s}: {importances[idx] * 100:.1f}%")
        elif hasattr(base_model, "coef_"):
            # LogisticRegression: gemiddelde absolute coëfficiënten
            coef_abs = np.abs(base_model.coef_).mean(axis=0)
            top_idx = np.argsort(coef_abs)[::-1][:10]
            print("Top-10 features (coëfficiënten):")
            for rank, idx in enumerate(top_idx, 1):
                fname = feature_cols[idx] if idx < len(feature_cols) else f"feat_{idx}"
                print(f"  {rank:2d}. {fname:25s}: {coef_abs[idx]:.4f}")

        results[name] = {
            "model":       model,
            "acc":         acc,
            "hc_acc":      hc_acc,
            "use_scaled":  use_scaled,
            "final_preds": final_preds,
            "final_proba": final_proba,
        }

    # ── Beste model kiezen ───────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["acc"])
    best      = results[best_name]

    print(f"\n{'=' * 60}")
    print("EINDRAPPORT")
    print(f"{'=' * 60}")
    print(f"Huidig systeem (handmatig):  ~34.4%")
    print(f"Baseline (altijd Flat):       {baseline_acc * 100:.1f}%")
    print(f"Drempel om zinvol te zijn:    >{baseline_acc * 100:.1f}% overall  OF  >65% top-20% conf")
    print()
    for name, r in sorted(results.items(), key=lambda x: -x[1]["acc"]):
        marker = " ← BESTE" if name == best_name else ""
        beats = "✓ beter dan baseline" if r["acc"] > baseline_acc else "✗ slechter dan baseline"
        print(f"  {name:25s}: {r['acc'] * 100:.1f}%  top-20%conf: {r['hc_acc'] * 100:.1f}%  {beats}{marker}")
    print()
    if best["hc_acc"] > 0.65:
        verdict = "POSITIEF — top-20% confidence >65%. Model heeft echte voorspelkracht. Overweeg integratie."
    elif best["acc"] > baseline_acc:
        verdict = "MATIG — beter dan baseline maar top-20% conf <65%. Meer features of data nodig."
    else:
        verdict = "NEGATIEF — geen enkel model verslaat de baseline. Altijd Flat voorspellen wint."
    print(f"Conclusie: {verdict}")

    # ── Eerlijke top-20% check voor het BESTE model ──────────────────────
    best_proba = best.get("final_proba")
    best_preds = best.get("final_preds")
    if best_proba is not None and best_preds is not None:
        print(f"\n{'=' * 60}")
        print(f"TOP-20% EERLIJKHEIDSCHECK — {best_name}")
        print(f"{'=' * 60}")
        max_conf = best_proba.max(axis=1)
        thr = np.quantile(max_conf, 0.80)
        hc_mask = max_conf >= thr
        n_hc = int(hc_mask.sum())
        if n_hc > 0:
            hc_acc = accuracy_score(y_test[hc_mask], best_preds[hc_mask])
            # Top-20% baseline = grootste-klasse-percentage BINNEN deze subset
            counts = np.bincount(y_test[hc_mask].astype(int), minlength=3)
            biggest = int(counts.argmax())
            top20_baseline = counts.max() / n_hc
            edge = (hc_acc - top20_baseline) * 100
            print(f"Top-20% confidence accuracy : {hc_acc * 100:.1f}%  (n={n_hc})")
            print(f"Top-20% baseline (grootste klasse = {label_map[biggest]}): "
                  f"{top20_baseline * 100:.1f}%")
            print(f"Echte edge (acc - baseline) : {edge:+.1f} ppt  "
                  f"{'→ voegt iets toe' if edge > 0 else '→ voegt NIETS toe'}")

        # ── Kalibratie-check: voorspelde kans vs werkelijke accuracy ─────
        print(f"\nKALIBRATIE-CHECK ({best_name}) — klopt het percentage?")
        print(f"  {'bucket':10s} {'n':>7s} {'gem. voorspeld':>15s} {'werkelijke acc':>16s}")
        correct = (best_preds == y_test)
        buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]
        for lo, hi in buckets:
            bmask = (max_conf >= lo) & (max_conf < hi)
            n_b = int(bmask.sum())
            label = "70%+" if hi > 1.0 else f"{int(lo*100)}-{int(hi*100)}%"
            if n_b == 0:
                print(f"  {label:10s} {n_b:>7d} {'—':>15s} {'—':>16s}")
                continue
            avg_pred = max_conf[bmask].mean() * 100
            actual_acc = correct[bmask].mean() * 100
            print(f"  {label:10s} {n_b:>7d} {avg_pred:>14.1f}% {actual_acc:>15.1f}%")

    # ── Beste model opslaan ──────────────────────────────────────────────
    joblib.dump(
        {"model": best["model"], "scaler": scaler if best["use_scaled"] else None},
        MODEL_PATH,
    )
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"\nModel opgeslagen: {MODEL_PATH}")
    print(f"Features opgeslagen: {FEATURES_PATH}")

    # Sla alle modellen apart op voor ensemble voting
    model_key_map = [
        ("gb", "GradientBoosting"),
        ("rf", "RandomForest"),
        ("lr", "LogisticRegression"),
        ("lgbm", "LightGBM"),
    ]
    for short_name, model_name in model_key_map:
        if model_name not in results:
            continue
        r = results[model_name]
        model_path = Path(__file__).parent / f"ml_model_{short_name}.pkl"
        joblib.dump(
            {"model": r["model"], "scaler": scaler if r["use_scaled"] else None},
            model_path,
        )
        print(f"Ensemble model opgeslagen: {model_path}")


if __name__ == "__main__":
    train_and_evaluate()
