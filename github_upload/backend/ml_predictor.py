"""Productie ML predictor — ensemble voting over GradientBoosting, RandomForest en LogisticRegression.

Valt terug op het legacy scoring systeem als geen ensemble modellen beschikbaar zijn.
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
# Onderdruk sklearn joblib warnings specifiek
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

logger = logging.getLogger(__name__)

MODEL_PATH    = Path(__file__).parent / "ml_model.pkl"
FEATURES_PATH = Path(__file__).parent / "ml_features.json"

_models: dict = {}          # {"gb": model_data, "rf": model_data, "lr": model_data}
_features: list[str] | None = None
_load_attempted = False

CONFIDENCE_THRESHOLD = 0.50

_FEATURE_EXPLANATIONS = {
    "atr_pct":               "Dagelijkse beweeglijkheid",
    "distance_to_52w_low":   "Afstand tot jaarlaagtepunt",
    "distance_to_52w_high":  "Afstand tot jaarhoogtepunt",
    "volatility_20d":        "Volatiliteit afgelopen 20 dagen",
    "macd_diff":             "MACD momentum",
    "ma20_distance":         "Afstand tot 20-daags gemiddelde",
    "ma50_distance":         "Afstand tot 50-daags gemiddelde",
    "ma200_distance":        "Afstand tot 200-daags gemiddelde",
    "rsi_14":                "RSI waarde",
    "mom_5d":                "5-daags momentum",
    "mom_10d":               "10-daags momentum",
    "mom_20d":               "20-daags momentum",
    "obv_slope":             "Volume-prijs trend (OBV)",
    "adx_14":                "Trendsterkte (ADX)",
    "bb_position":           "Positie binnen Bollinger Bands",
    "volume_ratio":          "Volume vs gemiddelde",
    "volume_trend":          "Recente volume trend",
    "market_regime_bull":    "Marktregime bullish",
    "market_regime_bear":    "Marktregime bearish",
    "ma_alignment":          "MA uitlijning",
    "rsi_oversold":          "RSI oververkocht",
    "rsi_overbought":        "RSI overkocht",
    "macd_bull":             "MACD bullish signaal",
    "macd_bear":             "MACD bearish signaal",
    "gap_pct":               "Opening gap",
    "adx_strong":            "Sterke trend (ADX>25)",
    "stoch_rsi_k":           "Stochastic RSI",
    "candlestick_bullish":   "Bullish kaarspatroon",
    "candlestick_bearish":   "Bearish kaarspatroon",
    "relative_strength_5d":  "Relatieve sterkte vs markt (5d)",
    "relative_strength_20d": "Relatieve sterkte vs markt (20d)",
    "sector_relative_strength_20d": "Relatieve sterkte vs sector (20d)",
    "price_acceleration":    "Prijsversnelling",
    "volume_anomaly":        "Volume-afwijking",
    "volatility_regime":     "Volatiliteitsregime",
    "trend_consistency":     "Trendconsistentie",
    "rsi_trend":             "RSI-trend",
    "macd_acceleration":     "MACD-versnelling",
    "vix_level":             "VIX (verwachte marktvolatiliteit)",
    "tnx_level":             "10-jaars US-staatsrente",
    "dxy_change_5d":         "Dollar-index 5-daagse verandering",
}


def _load_models() -> bool:
    """Laad alle ensemble modellen eenmalig. Geeft True als minstens 1 model geladen is."""
    global _models, _features, _load_attempted
    if _load_attempted:
        return len(_models) > 0
    _load_attempted = True

    try:
        import joblib
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
    except ImportError:
        logger.error("joblib niet beschikbaar — fallback naar legacy systeem")
        return False

    model_files = {
        "gb": "ml_model_gb.pkl",
        "rf": "ml_model_rf.pkl",
        "lr": "ml_model_lr.pkl",
    }
    for name, filename in model_files.items():
        path = Path(__file__).parent / filename
        if path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _models[name] = joblib.load(path)
                logger.info("Ensemble model geladen: %s (%s)", name, filename)
            except Exception as e:
                logger.error("Ensemble model %s laden mislukt: %s", name, e)

    if not _models and MODEL_PATH.exists():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _models["gb"] = joblib.load(MODEL_PATH)
            logger.warning("Ensemble modellen niet gevonden, fallback naar ml_model.pkl")
        except Exception as e:
            logger.error("Fallback model laden mislukt: %s — legacy systeem actief", e)

    if not _models:
        logger.warning("WAARSCHUWING: geen ML modellen gevonden, fallback naar legacy systeem")
        return False

    if FEATURES_PATH.exists():
        try:
            with open(FEATURES_PATH, encoding="utf-8") as f:
                _features = json.load(f)
        except Exception as e:
            logger.error("Features laden mislukt: %s", e)

    logger.info("%d ensemble model(len) geladen", len(_models))
    return True


def extract_features(indicators: dict, market_regime: str, feature_names: list[str],
                     earnings: dict | None = None) -> list[float]:
    """Bouwt feature vector in exact dezelfde volgorde als training (ml_features.json).

    `earnings` bevat de earnings-nabijheid features voor deze ticker/datum. Bij
    None worden gaten-bestendige fallback-waarden gebruikt (nooit NaN/0-fout).
    """
    ind = indicators
    current = ind.get("current_price", 1) or 1
    earn = earnings or {}

    ma20  = ind.get("ma20")
    ma50  = ind.get("ma50")
    ma200 = ind.get("ma200")

    ma20_dist  = ((current - ma20)  / ma20  * 100) if ma20  else 0.0
    ma50_dist  = ((current - ma50)  / ma50  * 100) if ma50  else 0.0
    ma200_dist = ((current - ma200) / ma200 * 100) if ma200 else 0.0

    if ma20 and ma50 and ma200:
        if ma20 > ma50 > ma200:
            ma_align = 1.0
        elif ma20 < ma50 < ma200:
            ma_align = -1.0
        else:
            ma_align = 0.0
    else:
        ma_align = 0.0

    macd      = ind.get("macd") or {}
    macd_diff = (macd.get("macd", 0) or 0) - (macd.get("signal", 0) or 0)
    hist      = macd.get("histogram", 0) or 0

    rsi     = ind.get("rsi", 50) or 50
    atr     = ind.get("atr") or 0
    atr_pct = (atr / current * 100) if current else 0.0

    bb_pos      = ind.get("bb_position")
    bb_pos_norm = (bb_pos / 100.0) if bb_pos is not None else 0.5

    candle_patterns = ind.get("candle_patterns") or []
    bullish_pats    = {"hammer", "bullish_engulfing", "marubozu_bullish"}
    bearish_pats    = {"shooting_star", "bearish_engulfing", "marubozu_bearish"}

    # ── DEEL 3: nieuwe features (zelfde berekening als trainer) ──
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

    feat_map = {
        "rsi_14":               float(rsi),
        "rsi_oversold":         float(rsi < 30),
        "rsi_overbought":       float(rsi > 70),
        "macd_diff":            float(macd_diff),
        "macd_bull":            float(hist > 0),
        "macd_bear":            float(hist < 0),
        "ma20_distance":        float(ma20_dist),
        "ma50_distance":        float(ma50_dist),
        "ma200_distance":       float(ma200_dist),
        "ma_alignment":         float(ma_align),
        "mom_5d":               float(ind.get("momentum_5d", 0) or 0),
        "mom_10d":              float(ind.get("momentum_10d", 0) or 0),
        "mom_20d":              float(ind.get("momentum_20d", 0) or 0),
        "volume_ratio":         float(ind.get("volume_ratio", 1.0) or 1.0),
        "volume_trend":         float(ind.get("volume_trend_ratio", 1.0) or 1.0),
        "obv_slope":            float(ind.get("obv_slope", 0.0) or 0.0),
        "adx_14":               float(ind.get("adx", 0) or 0),
        "adx_strong":           float(ind.get("adx_strong_trend", False)),
        "bb_position":          float(bb_pos_norm),
        "stoch_rsi_k":          float((ind.get("stoch_rsi_k") or 50) / 100.0),
        "atr_pct":              float(atr_pct),
        "volatility_20d":       float(ind.get("volatility_20d", 0) or 0),
        "distance_to_52w_high": float(ind.get("pct_from_52w_high", 0) or 0),
        "distance_to_52w_low":  float(ind.get("pct_from_52w_low", 0) or 0),
        "market_regime_bull":   float(market_regime == "bull"),
        "market_regime_bear":   float(market_regime == "bear"),
        "gap_pct":              float(ind.get("gap_pct", 0) or 0),
        "candlestick_bullish":  float(any(p in bullish_pats for p in candle_patterns)),
        "candlestick_bearish":  float(any(p in bearish_pats for p in candle_patterns)),
        "relative_strength_5d":  float(rel_str_5d),
        "relative_strength_20d": float(rel_str_20d),
        "sector_relative_strength_20d": float(sector_relative_strength_20d),
        "price_acceleration":    float(price_accel),
        "volume_anomaly":        float(volume_anomaly),
        "volatility_regime":     float(volatility_regime),
        "trend_consistency":     float(trend_consistency),
        "rsi_trend":             float(rsi_trend),
        "macd_acceleration":     float(macd_accel),
        "vix_level":             vix_level,
        "tnx_level":             tnx_level,
        "dxy_change_5d":         dxy_change_5d,
        "days_since_last_earnings": float(earn.get("days_since_last_earnings", 45.0)),
        "days_until_next_earnings": float(earn.get("days_until_next_earnings", 45.0)),
        "earnings_zone":            float(earn.get("earnings_zone", 0.0)),
        "news_sentiment":       0.0,
    }

    return [feat_map.get(name, 0.0) for name in feature_names]


def _get_top_features(feature_values: list[float], feature_names: list[str], model) -> list[dict]:
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    top_idx = importances.argsort()[::-1][:5]
    result = []
    for idx in top_idx:
        if idx >= len(feature_names) or idx >= len(feature_values):
            continue
        name = feature_names[idx]
        result.append({
            "name":       _FEATURE_EXPLANATIONS.get(name, name),
            "value":      round(float(feature_values[idx]), 2),
            "importance": round(float(importances[idx]) * 100, 1),
        })
    return result


def _ensemble_predict(feature_values: list[float]) -> tuple:
    """
    Combineert voorspellingen van alle beschikbare modellen via gewogen voting.
    Returns: (direction, confidence, probs_dict, consensus_ratio)
    """
    if not _models:
        return None, 0, {}, 0.0

    WEIGHTS = {
        "gb": 1.0,
        "rf": 1.0,
        "lr": 1.0,
    }

    feat_arr = np.array(feature_values, dtype=float).reshape(1, -1)
    weighted_probs = np.zeros(3)  # [Down, Flat, Up]
    total_weight = 0.0
    predictions = []

    for name, model_data in _models.items():
        if name not in WEIGHTS:
            continue
        model  = model_data["model"]
        scaler = model_data.get("scaler")
        arr    = scaler.transform(feat_arr) if scaler is not None else feat_arr
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probs  = model.predict_proba(arr)[0]
                predictions.append(int(model.predict(arr)[0]))
            weight = WEIGHTS.get(name, 0.33)
            weighted_probs += probs * weight
            total_weight   += weight
        except Exception as e:
            logger.error("Ensemble model %s predict mislukt: %s", name, e)

    if total_weight == 0:
        return None, 0, {}, 0.0

    weighted_probs /= total_weight

    from collections import Counter
    consensus      = Counter(predictions)
    most_common    = consensus.most_common(1)[0]
    consensus_ratio = most_common[1] / len(predictions) if predictions else 0.0

    labels    = ["Down", "Flat", "Up"]
    max_idx   = int(weighted_probs.argmax())
    direction = labels[max_idx]
    # Rauwe gekalibreerde kans van het sterkste signaal — GEEN consensus-bonus
    # of kunstmatige op-/afslag (dat vervalste het gekalibreerde percentage).
    confidence = int(round(weighted_probs[max_idx] * 100))

    probs_dict = {
        "Down": int(weighted_probs[0] * 100),
        "Flat": int(weighted_probs[1] * 100),
        "Up":   int(weighted_probs[2] * 100),
    }

    return direction, confidence, probs_dict, consensus_ratio


def predict_ml(indicators: dict, market_regime: str, current_price: float, ticker: str = "") -> dict | None:
    """
    Geeft ensemble ML voorspelling of None (als geen modellen geladen zijn).
    Returns dict met: direction, confidence, probabilities, target_price, target_range,
                      top_features, consensus, models_used, news_sentiment
    """
    if not _load_models():
        return None

    if _features is None:
        return None

    try:
        # Earnings-nabijheid voor deze ticker, op de datum van vandaag. De
        # kalender wordt gecachet in data.py; bij fout/te weinig data vallen
        # we terug op de fallback-waarden binnen extract_features.
        earnings = None
        if ticker:
            try:
                from datetime import date
                import data as data_module
                earnings = data_module.compute_earnings_features(ticker, date.today())
            except Exception as e:
                logger.warning("Earnings-features live mislukt voor %s: %s", ticker, e)
                earnings = None

        features  = extract_features(indicators, market_regime, _features, earnings=earnings)
        direction, confidence, probs_dict, consensus_ratio = _ensemble_predict(features)

        if direction is None:
            return None

        # Gebruik de rauwe gekalibreerde kans als confidence — geen
        # TICKER_RELIABILITY-multiplier en geen clip naar 40-85 (die
        # vervalsten het echte gekalibreerde percentage).
        max_weighted = max(probs_dict.values()) / 100.0
        confidence = int(round(max_weighted * 100))

        # Signaalsterkte op basis van de ECHTE gekalibreerde kans.
        if max_weighted < CONFIDENCE_THRESHOLD:          # < 0.50
            signal_strength = "geen"
            betrouwbaarheids_tekst = (
                "Geen betrouwbaar signaal — het model is hier te onzeker (onder 50%). "
                "Behandel dit niet als koop- of verkoopadvies."
            )
        elif max_weighted < 0.55:                        # 0.50 - 0.55
            signal_strength = "zwak"
            betrouwbaarheids_tekst = (
                f"Zwak signaal ({confidence}% zeker). Wees voorzichtig, dit is "
                f"nauwelijks beter dan gokken."
            )
        else:                                            # >= 0.55
            signal_strength = "sterk"
            betrouwbaarheids_tekst = f"Redelijk signaal ({confidence}% zeker)."

        # Onder de drempel: geen sterk signaal. Behoud forced_flat-veld voor
        # achterwaartse compatibiliteit (nu gekoppeld aan signaalsterkte).
        forced_flat = signal_strength == "geen"

        # Nieuws wordt asynchroon geladen via /api/news/{ticker} in de frontend
        sentiment = 0.0

        atr = indicators.get("atr") or (current_price * 0.02)
        move = min(atr * 2, current_price * 0.08)
        if direction == "Up":
            target       = round(current_price + move, 2)
            target_range = (round(current_price + move * 0.5, 2), round(current_price + move * 1.5, 2))
        elif direction == "Down":
            target       = round(current_price - move, 2)
            target_range = (round(current_price - move * 1.5, 2), round(current_price - move * 0.5, 2))
        else:
            target       = round(current_price, 2)
            band         = min(atr, current_price * 0.04)
            target_range = (round(current_price - band, 2), round(current_price + band, 2))

        # Top features van het gb model (heeft feature_importances_)
        gb_model = _models.get("gb", {}).get("model") if _models else None
        top_features = _get_top_features(features, _features, gb_model) if gb_model else []

        return {
            "direction":      direction,
            "confidence":     confidence,
            "probabilities":  probs_dict,
            "target_price":   target,
            "target_range":   target_range,
            "top_features":   top_features,
            "news_sentiment": sentiment,
            "consensus":      round(consensus_ratio, 2),
            "models_used":    len(_models),
            "forced_flat":    forced_flat,
            "signal_strength":        signal_strength,
            "betrouwbaarheids_tekst": betrouwbaarheids_tekst,
        }
    except Exception as e:
        logger.error("ML ensemble voorspelling mislukt: %s", e)
        return None


# Modellen laden bij module-import (niet per request)
_load_models()
