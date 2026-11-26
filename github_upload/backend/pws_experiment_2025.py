"""
PWS-experimenten: gedeelde functies voor experiment 1 en 2.

Datumgrenzen, koersdata ophalen, de 44 causale features, het trainen van de
vier modellen en de handelssimulatie. De experimenten zelf draai je met:
  excel_exp1_voorspellingen.py -> excel_exp1_schrijven.py   (experiment 1)
  excel_exp2_transacties.py    -> excel_exp2_schrijven.py   (experiment 2)

Training op koersdata 01-01-2015 t/m 31-12-2024 (geen enkele 2025-observatie
in features of labels), test op 2025. Hergebruikt alleen
read-only imports uit data.py (SECTOR_ETF_MAP, get_earnings_calendar).

BELANGRIJKE CORRECTIE t.o.v. de bestaande pijplijn (ml_trainer.py/indicators.py):
compute_all() daar berekent spy_momentum_5d/20d en sector_momentum_20d altijd
met de LIVE (huidige) marktdata,ongeacht de historische datum waarvoor de rij
bedoeld is — dat is een impliciet datalek. Dit script herberekent alle 43
features zelf, volledig gevectoriseerd EN causaal (elke waarde op datum X
gebruikt uitsluitend koersdata t/m X), en past daarna uniform een shift(1)
toe zodat een "sample" op datum X altijd de indicatoren van de vorige
slotkoers gebruikt (net als in ml_trainer.py's build_dataset, waar
window_df = df.iloc[:i] wordt gebruikt).
"""
from __future__ import annotations

import argparse
import gc
import io
import logging
import os
import pickle
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from data import SECTOR_ETF_MAP, get_earnings_calendar  # read-only hergebruik

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = os.path.dirname(__file__)

EARNINGS_TIMEOUT_S = 10   # harde bovengrens per earnings-oproep (vorige run hing ~2,9 uur op CVNA)
CHECKPOINT_EVERY = 25     # tussenstand wegschrijven na elke N gesimuleerde tickers

# Geheugenbeperking (vorige run werd gestopt door geheugentekort bij RF/GB op 1,1M samples)
MAX_HEAVY_SAMPLES = 400_000   # RF + GB trainen op max. zoveel willekeurig getrokken samples
HEAVY_SEED = 42
RF_TREES = 300                # was 500
RF_JOBS = 4                   # was -1 (alle 8 cores); minder parallelle bootstraps tegelijk
MODEL_DIR = os.path.join(OUT_DIR, "pws_experiment_models")   # nieuw; raakt ml_model*.pkl niet aan


class EarningsTimeout(Exception):
    """Earnings-oproep duurde langer dan EARNINGS_TIMEOUT_S."""


def get_earnings_calendar_safe(ticker: str) -> list | None:
    """get_earnings_calendar met een harde timeout. De oproep draait in een
    daemon-thread; blijft die hangen dan laten we hem achter en gooien we
    EarningsTimeout, zodat de aanroeper de ticker kan overslaan."""
    box: dict = {}

    def _worker():
        try:
            box["result"] = get_earnings_calendar(ticker)
        except Exception as e:  # noqa: BLE001 — doorgeven aan de aanroeper
            box["error"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(EARNINGS_TIMEOUT_S)
    if t.is_alive():
        raise EarningsTimeout(f"earnings-oproep duurde langer dan {EARNINGS_TIMEOUT_S}s")
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _atomic_to_csv(df: pd.DataFrame, path: str) -> None:
    """Schrijf via tijdelijk bestand + os.replace, zodat een crash halverwege
    nooit een half CSV-bestand achterlaat."""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

# ── Harde datumgrenzen — het hele punt van dit experiment ──────────────────
TRAIN_CUTOFF = pd.Timestamp("2024-12-31")   # laatste dag die het model mag "zien"
TEST_START   = pd.Timestamp("2025-01-01")
TEST_END     = pd.Timestamp("2025-12-31")
FETCH_START  = "2015-01-01"                 # ruime warmup vóór de trainingsperiode
FETCH_END    = "2026-01-02"                 # pakt 31-12-2025 nog mee

FORECAST_HORIZON = 5
TARGET_THRESHOLD = 2.0
MIN_WINDOW = 210
SIDEWAYS_PCT = 0.1  # |rendement| < dit = "zijwaarts" (nul of vrijwel geen verandering)
CHUNK_SIZE = 60

FEATURE_NAMES = [
    "rsi_14", "rsi_oversold", "rsi_overbought",
    "macd_diff", "macd_bull", "macd_bear",
    "ma20_distance", "ma50_distance", "ma200_distance", "ma_alignment",
    "mom_5d", "mom_10d", "mom_20d",
    "volume_ratio", "volume_trend", "obv_slope",
    "adx_14", "adx_strong", "bb_position", "stoch_rsi_k",
    "atr_pct", "volatility_20d",
    "distance_to_52w_high", "distance_to_52w_low",
    "market_regime_bull", "market_regime_bear",
    "gap_pct", "candlestick_bullish", "candlestick_bearish",
    "relative_strength_5d", "relative_strength_20d",
    "sector_relative_strength_20d",
    "price_acceleration", "volume_anomaly", "volatility_regime",
    "trend_consistency", "rsi_trend", "macd_acceleration",
    "vix_level", "tnx_level", "dxy_change_5d",
    "days_since_last_earnings", "days_until_next_earnings", "earnings_zone",
]

STRATEGY_ORDER = ["ML_Ensemble", "RSI", "Stochastic", "MACD", "BuyAndHold"]
STRATEGY_LABELS = {
    "ML_Ensemble": "1. Mijn ML-model (ensemble)",
    "RSI": "2. RSI (30/70 kruising)",
    "Stochastic": "3. Stochastic Oscillator (%K/%D, 20/80)",
    "MACD": "4. MACD (nullijn-kruising)",
    "BuyAndHold": "5. Buy & Hold",
}


# ── Tickers & data ophalen ──────────────────────────────────────────────────

def fetch_sp500_tickers() -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
    resp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                         headers=headers, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    sp500 = tables[0]
    tickers = sp500["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return sorted(set(tickers))


def batch_download(tickers: list[str], start: str, end: str, chunk_size: int = CHUNK_SIZE) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    n_chunks = (len(tickers) - 1) // chunk_size + 1 if tickers else 0
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        data = None
        for attempt in range(3):
            try:
                data = yf.download(chunk, start=start, end=end, auto_adjust=True,
                                    group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:
                logger.warning("Batch download fout (poging %d/3): %s", attempt + 1, e)
                time.sleep(3 * (attempt + 1))
        if data is None or data.empty:
            print(f"  Batch {i // chunk_size + 1}/{n_chunks}: mislukt, overgeslagen")
            continue
        if isinstance(data.columns, pd.MultiIndex):
            top = set(data.columns.get_level_values(0))
            for t in chunk:
                if t in top:
                    sub = data[t].dropna(how="all")
                    if not sub.empty:
                        out[t] = sub
        elif len(chunk) == 1:
            out[chunk[0]] = data
        got = sum(1 for t in chunk if t in out)
        print(f"  Batch {i // chunk_size + 1}/{n_chunks}: {len(chunk)} opgevraagd, {got} met data")
    return out


# ── Gevectoriseerde, causale indicatoren ────────────────────────────────────

def compute_rsi_series(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def compute_macd_series(closes: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


def rolling_slope_normalized(series: pd.Series, window: int) -> pd.Series:
    """Genormaliseerde rollende regressiehelling (zoals _obv_slope in ml_trainer.py),
    gevectoriseerd via de gesloten-vorm covariantieformule (geen rolling.apply nodig)."""
    t = pd.Series(np.arange(len(series)), index=series.index, dtype=float)
    ty = t * series
    mean_y = series.rolling(window).mean()
    mean_t = t.rolling(window).mean()
    mean_ty = ty.rolling(window).mean()
    cov_ty = mean_ty - mean_t * mean_y
    var_t = (window ** 2 - 1) / 12.0
    slope = cov_ty / var_t
    mean_abs = series.abs().rolling(window).mean().replace(0, np.nan)
    return (slope / mean_abs).fillna(0.0)


def compute_earnings_features_vectorized(dates: pd.DatetimeIndex, calendar: list | None) -> pd.DataFrame:
    n = len(dates)
    if not calendar:
        return pd.DataFrame({
            "days_since_last_earnings": np.full(n, 45.0),
            "days_until_next_earnings": np.full(n, 45.0),
            "earnings_zone": np.zeros(n),
        }, index=dates)
    cal_ord = np.array(sorted(pd.Timestamp(d).toordinal() for d in calendar))
    date_ord = np.array([pd.Timestamp(d).toordinal() for d in dates])
    idx = np.searchsorted(cal_ord, date_ord, side="right")
    days_since = np.full(n, 45.0)
    days_until = np.full(n, 45.0)
    has_past = idx > 0
    days_since[has_past] = (date_ord[has_past] - cal_ord[idx[has_past] - 1]).astype(float)
    has_future = idx < len(cal_ord)
    days_until[has_future] = (cal_ord[idx[has_future]] - date_ord[has_future]).astype(float)
    zone = ((days_since < 5) | (days_until < 10)).astype(float)
    return pd.DataFrame({
        "days_since_last_earnings": days_since,
        "days_until_next_earnings": days_until,
        "earnings_zone": zone,
    }, index=dates)


def _align(series: pd.Series | None, target_index: pd.DatetimeIndex, fill) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(fill, index=target_index)
    combined = series.reindex(target_index.union(series.index)).sort_index().ffill()
    return combined.reindex(target_index).fillna(fill)


def build_feature_frame(df: pd.DataFrame, ticker: str, spy_aux: dict,
                         sector_series_map: dict, vix_s, tnx_s, dxy_chg_s) -> pd.DataFrame:
    """Bouwt alle 43 features, causaal en gevectoriseerd (nog GEEN shift(1) —
    dat past de aanroeper toe, zodat deze functie ook los te testen is)."""
    closes, opens, highs, lows, volumes = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    idx = df.index

    rsi_series = compute_rsi_series(closes)
    macd_line, signal_line, histogram = compute_macd_series(closes)
    atr_series = compute_atr_series(df)
    adx_series = compute_adx_series(df)

    ma20, ma50, ma200 = closes.rolling(20).mean(), closes.rolling(50).mean(), closes.rolling(200).mean()

    std20 = closes.rolling(20).std()
    upper_bb, lower_bb = ma20 + 2 * std20, ma20 - 2 * std20
    band_width = upper_bb - lower_bb
    bb_position = ((closes - lower_bb) / band_width * 100).where(band_width > 0, 50.0)

    rsi_min14, rsi_max14 = rsi_series.rolling(14).min(), rsi_series.rolling(14).max()
    rsi_range = (rsi_max14 - rsi_min14).replace(0, np.nan)
    stoch_rsi_k = (rsi_series - rsi_min14) / rsi_range * 100

    direction = np.sign(closes.diff()).fillna(0)
    obv = (direction * volumes).cumsum()
    obv_slope = rolling_slope_normalized(obv, 10)

    high_52 = closes.rolling(252, min_periods=50).max()
    low_52 = closes.rolling(252, min_periods=50).min()
    pct_from_52w_high = (closes - high_52) / high_52 * 100
    pct_from_52w_low = (closes - low_52) / low_52 * 100

    prev_close = closes.shift(1)
    gap_pct = (opens - prev_close) / prev_close * 100

    body = (closes - opens).abs()
    total_range = (highs - lows).replace(0, np.nan)
    hi_oc = pd.concat([opens, closes], axis=1).max(axis=1)
    lo_oc = pd.concat([opens, closes], axis=1).min(axis=1)
    upper_wick, lower_wick = highs - hi_oc, lo_oc - lows
    body_ratio = (body / total_range).fillna(0.0)
    prev_c, prev2_c, prev_o = closes.shift(1), closes.shift(2), opens.shift(1)
    hammer = (lower_wick > body * 2) & (upper_wick < body * 0.5) & (closes > opens) & (prev_c < prev2_c)
    shooting_star = (upper_wick > body * 2) & (lower_wick < body * 0.5) & (closes < opens) & (prev_c > prev2_c)
    bull_engulf = (closes > opens) & (prev_c < prev_o) & (closes > prev_o) & (opens < prev_c)
    bear_engulf = (closes < opens) & (prev_c > prev_o) & (closes < prev_o) & (opens > prev_c)
    marubozu_bull = (closes > opens) & (body_ratio > 0.85)
    marubozu_bear = (closes < opens) & (body_ratio > 0.85)
    candlestick_bullish = (hammer | bull_engulf | marubozu_bull).fillna(False).astype(float)
    candlestick_bearish = (shooting_star | bear_engulf | marubozu_bear).fillna(False).astype(float)

    mom_5d, mom_10d, mom_20d = closes.pct_change(5) * 100, closes.pct_change(10) * 100, closes.pct_change(20) * 100

    ret = closes.pct_change()
    vol20 = ret.rolling(20).std() * np.sqrt(252) * 100
    vol60 = ret.rolling(60, min_periods=60).std() * np.sqrt(252) * 100
    vol60 = vol60.where(vol60 > 0, vol20).fillna(vol20)

    avg_vol_prev20 = volumes.shift(1).rolling(20).mean()
    volume_ratio = (volumes / avg_vol_prev20).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    avg5, avg20 = volumes.rolling(5).mean(), volumes.rolling(20).mean()
    volume_trend = (avg5 / avg20).replace([np.inf, -np.inf], 1.0).fillna(1.0)

    trend_consistency = (closes.diff() > 0).astype(float).rolling(10).mean().fillna(0.5)
    rsi_trend = (rsi_series - rsi_series.shift(5)).fillna(0.0)
    macd_acceleration = (histogram - histogram.shift(3)).fillna(0.0)

    ma20_distance = (closes - ma20) / ma20 * 100
    ma50_distance = (closes - ma50) / ma50 * 100
    ma200_distance = (closes - ma200) / ma200 * 100
    ma_alignment = pd.Series(0.0, index=idx)
    valid_ma = ma20.notna() & ma50.notna() & ma200.notna()
    ma_alignment[valid_ma & (ma20 > ma50) & (ma50 > ma200)] = 1.0
    ma_alignment[valid_ma & (ma20 < ma50) & (ma50 < ma200)] = -1.0

    macd_bull = (histogram > 0).astype(float)
    macd_bear = (histogram < 0).astype(float)
    rsi_oversold = (rsi_series < 30).astype(float)
    rsi_overbought = (rsi_series > 70).astype(float)
    adx_strong = (adx_series > 25).astype(float)
    atr_pct = atr_series / closes * 100

    spy_mom5_a = _align(spy_aux.get("mom5"), idx, 0.0)
    spy_mom20_a = _align(spy_aux.get("mom20"), idx, 0.0)
    regime_a = _align(spy_aux.get("regime"), idx, "neutral")
    market_regime_bull = (regime_a == "bull").astype(float)
    market_regime_bear = (regime_a == "bear").astype(float)

    sector_series = sector_series_map.get(ticker)
    sector_mom20_a = _align(sector_series, idx, np.nan)
    sector_mom20_a = sector_mom20_a.fillna(spy_mom20_a)

    vix_a = _align(vix_s, idx, 0.0)
    tnx_a = _align(tnx_s, idx, 0.0)
    dxy_a = _align(dxy_chg_s, idx, 0.0)

    relative_strength_5d = mom_5d - spy_mom5_a
    relative_strength_20d = mom_20d - spy_mom20_a
    sector_relative_strength_20d = mom_20d - sector_mom20_a
    price_acceleration = mom_5d - mom_10d
    volume_anomaly = (volume_ratio - 1.0).clip(-2, 2)
    volatility_regime = vol20 / vol60.clip(lower=0.001)

    cal = get_earnings_calendar_safe(ticker)
    earnings_df = compute_earnings_features_vectorized(idx, cal)

    feats = pd.DataFrame({
        "rsi_14": rsi_series, "rsi_oversold": rsi_oversold, "rsi_overbought": rsi_overbought,
        "macd_diff": histogram, "macd_bull": macd_bull, "macd_bear": macd_bear,
        "ma20_distance": ma20_distance, "ma50_distance": ma50_distance, "ma200_distance": ma200_distance,
        "ma_alignment": ma_alignment,
        "mom_5d": mom_5d, "mom_10d": mom_10d, "mom_20d": mom_20d,
        "volume_ratio": volume_ratio, "volume_trend": volume_trend, "obv_slope": obv_slope,
        "adx_14": adx_series, "adx_strong": adx_strong,
        "bb_position": bb_position / 100.0, "stoch_rsi_k": stoch_rsi_k / 100.0,
        "atr_pct": atr_pct, "volatility_20d": vol20,
        "distance_to_52w_high": pct_from_52w_high, "distance_to_52w_low": pct_from_52w_low,
        "market_regime_bull": market_regime_bull, "market_regime_bear": market_regime_bear,
        "gap_pct": gap_pct,
        "candlestick_bullish": candlestick_bullish, "candlestick_bearish": candlestick_bearish,
        "relative_strength_5d": relative_strength_5d, "relative_strength_20d": relative_strength_20d,
        "sector_relative_strength_20d": sector_relative_strength_20d,
        "price_acceleration": price_acceleration, "volume_anomaly": volume_anomaly,
        "volatility_regime": volatility_regime,
        "trend_consistency": trend_consistency, "rsi_trend": rsi_trend,
        "macd_acceleration": macd_acceleration,
        "vix_level": vix_a, "tnx_level": tnx_a, "dxy_change_5d": dxy_a,
        "days_since_last_earnings": earnings_df["days_since_last_earnings"],
        "days_until_next_earnings": earnings_df["days_until_next_earnings"],
        "earnings_zone": earnings_df["earnings_zone"],
    }, index=idx)

    return feats[FEATURE_NAMES]


# ── Model trainen ────────────────────────────────────────────────────────────

def train_ensemble(train_df: pd.DataFrame):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    try:
        from lightgbm import LGBMClassifier
        has_lgbm = True
    except ImportError:
        has_lgbm = False
        print("  [LET OP] LightGBM niet geïnstalleerd — ensemble gebruikt 3 i.p.v. 4 modellen.")

    X64 = train_df[FEATURE_NAMES].fillna(0).to_numpy(dtype=np.float64, copy=True)   # schrijfbaar (pandas CoW geeft anders read-only)
    y = train_df["target"].to_numpy()

    feat_mean, feat_std = X64.mean(axis=0), X64.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    clip_low, clip_high = feat_mean - 5 * feat_std, feat_mean + 5 * feat_std
    np.clip(X64, clip_low, clip_high, out=X64)          # in-place: geen extra kopie

    scaler = StandardScaler()
    scaler.fit(X64)
    # float32 halveert het geheugen; boommodellen binnen sklearn/LightGBM rekenen sowieso op float32.
    X_scaled = scaler.transform(X64).astype(np.float32)
    X = X64.astype(np.float32)
    del X64
    gc.collect()

    # Zware modellen (RF, GB) trainen op een vaste, willekeurige subsample.
    if len(X) > MAX_HEAVY_SAMPLES:
        heavy_idx = np.sort(np.random.RandomState(HEAVY_SEED).choice(len(X), MAX_HEAVY_SAMPLES, replace=False))
    else:
        heavy_idx = None

    # (model, gebruikt geschaalde data, subsample gebruiken)
    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=42), True, False),
        "RandomForest": (RandomForestClassifier(n_estimators=RF_TREES, max_depth=12, min_samples_leaf=10,
                                                  max_features="sqrt", random_state=42, n_jobs=RF_JOBS), False, True),
        "GradientBoosting": (GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=4,
                                                          subsample=0.8, min_samples_leaf=20, random_state=42), False, True),
    }
    if has_lgbm:
        models["LightGBM"] = (LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=8,
                                              num_leaves=31, subsample=0.8, random_state=42,
                                              n_jobs=-1, verbose=-1), False, False)

    _save_pickle("scaler_en_clip", {"scaler": scaler, "clip_low": clip_low, "clip_high": clip_high,
                                    "feature_names": FEATURE_NAMES})

    fitted = {}
    for name, (model, use_scaled, heavy) in models.items():
        Xm = X_scaled if use_scaled else X
        ym = y
        if heavy and heavy_idx is not None:
            Xm, ym = Xm[heavy_idx], y[heavy_idx]
        print(f"  Trainen {name} op {len(Xm):,} samples...", end=" ", flush=True)
        t0 = time.time()
        model.fit(Xm, ym)
        print(f"klaar ({time.time() - t0:.0f}s).")
        fitted[name] = (model, use_scaled)
        _save_pickle(f"model_{name}", {"model": model, "use_scaled": use_scaled, "n_train": len(Xm)})

        del Xm, ym
        if name == "LogisticRegression":
            del X_scaled           # geschaalde kopie is verder niet meer nodig
        gc.collect()

    return fitted, scaler, clip_low, clip_high


def _save_pickle(name: str, obj) -> None:
    """Sla een getraind onderdeel direct op; een fout hierbij mag de run niet stoppen."""
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, f"pws_experiment_{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"    opgeslagen: {os.path.relpath(path, OUT_DIR)}")
    except Exception as e:
        print(f"    [waarschuwing] opslaan van {name} mislukt, run gaat door: {e}")


# ── Trade-simulatie ──────────────────────────────────────────────────────────

def simulate_trades(dates, closes: pd.Series, buy_signal: pd.Series, sell_signal: pd.Series) -> list[dict]:
    """Long-only state machine: koop op buy_signal, verkoop op sell_signal.
    Negeert signalen die niet bij de huidige staat passen (geen shorts, geen
    dubbele buys). Sluit een eventuele open positie geforceerd af op de
    laatste testdag, zodat elke strategie een eerlijk vergelijkbare set
    VOLTOOIDE trades oplevert."""
    trades = []
    in_position = False
    entry_price = entry_date = None
    for date in dates:
        price = float(closes.loc[date])
        if not in_position and bool(buy_signal.loc[date]):
            in_position, entry_price, entry_date = True, price, date
        elif in_position and bool(sell_signal.loc[date]):
            trades.append({"entry_date": entry_date, "exit_date": date,
                            "entry_price": entry_price, "exit_price": price,
                            "return_pct": (price - entry_price) / entry_price * 100,
                            "forced_close": False})
            in_position = False
    if in_position:
        last_date = dates[-1]
        price = float(closes.loc[last_date])
        trades.append({"entry_date": entry_date, "exit_date": last_date,
                        "entry_price": entry_price, "exit_price": price,
                        "return_pct": (price - entry_price) / entry_price * 100,
                        "forced_close": True})
    return trades


def trade_stats(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "wins": 0, "losses": 0, "sideways": 0, "total_return_pct": 0.0}
    wins = sum(1 for t in trades if t["return_pct"] > SIDEWAYS_PCT)
    losses = sum(1 for t in trades if t["return_pct"] < -SIDEWAYS_PCT)
    sideways = n - wins - losses
    compounded = 1.0
    for t in trades:
        compounded *= (1 + t["return_pct"] / 100)
    return {"n_trades": n, "wins": wins, "losses": losses, "sideways": sideways,
            "total_return_pct": (compounded - 1) * 100}


# ── Grafiek ──────────────────────────────────────────────────────────────────

def plot_ticker_chart(ticker: str, dates, close: pd.Series, ml_trades: list[dict], out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(dates, close.values, color="#1f77b4", linewidth=1.4, label=f"{ticker} slotkoers")

    buy_dates = [t["entry_date"] for t in ml_trades]
    buy_prices = [t["entry_price"] for t in ml_trades]
    sell_dates = [t["exit_date"] for t in ml_trades if not t["forced_close"]]
    sell_prices = [t["exit_price"] for t in ml_trades if not t["forced_close"]]

    if buy_dates:
        ax.scatter(buy_dates, buy_prices, color="#2ca02c", marker="^", s=110, zorder=5,
                   label=f"ML koopt ({len(buy_dates)}x)", edgecolors="black", linewidths=0.5)
    if sell_dates:
        ax.scatter(sell_dates, sell_prices, color="#d62728", marker="v", s=110, zorder=5,
                   label=f"ML verkoopt ({len(sell_dates)}x)", edgecolors="black", linewidths=0.5)

    ax.set_title(f"PWS Experiment 2025 — {ticker}: koers + ML-ensemble koop/verkoop-signalen\n"
                 f"(model getraind uitsluitend op data t/m {TRAIN_CUTOFF.date()})", fontsize=13)
    ax.set_xlabel("Datum", fontsize=11)
    ax.set_ylabel("Koers ($)", fontsize=11)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    sys.exit("Dit bestand bevat alleen gedeelde functies. Draai excel_exp1_voorspellingen.py, "
             "excel_exp1_schrijven.py, excel_exp2_transacties.py en excel_exp2_schrijven.py.")
