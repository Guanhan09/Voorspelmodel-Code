from __future__ import annotations
from typing import Optional
import pandas as pd

from indicators import detect_market_regime
from ml_predictor import predict_ml


def _price_range(current: float, atr: float | None, volatility: float) -> tuple[float, float]:
    if atr and atr > 0:
        low = round(current - atr, 4)
        high = round(current + atr, 4)
    else:
        daily_vol = (volatility / 100) / (252 ** 0.5)
        low = round(current * (1 - daily_vol), 4)
        high = round(current * (1 + daily_vol), 4)
    return low, high


def _predict_legacy(indicators: dict, df: pd.DataFrame, patterns: Optional[list] = None, regime: str = "neutral") -> dict:
    score = 0.0
    reasons = []

    rsi = indicators.get("rsi", 50)
    ma20 = indicators.get("ma20")
    ma50 = indicators.get("ma50")
    ma200 = indicators.get("ma200")
    macd = indicators.get("macd", {})
    volume_ratio = indicators.get("volume_ratio", 1.0)
    volume_increasing = indicators.get("volume_increasing", False)
    volume_bullish = indicators.get("volume_bullish_profile", False)
    volume_bearish = indicators.get("volume_bearish_profile", False)
    momentum = indicators.get("momentum_5d", 0)
    momentum_10d = indicators.get("momentum_10d", 0)
    momentum_20d = indicators.get("momentum_20d", 0)
    volatility = indicators.get("volatility_20d", 0)
    atr = indicators.get("atr")
    current = indicators.get("current_price", 0)
    bb_position = indicators.get("bb_position")
    bb_squeeze = indicators.get("bb_squeeze", False)
    stoch_k = indicators.get("stoch_rsi_k")
    stoch_d = indicators.get("stoch_rsi_d")
    obv_trend = indicators.get("obv_trend")
    obv_confirmation = indicators.get("obv_confirmation")
    obv_strong = indicators.get("obv_strong", False)
    ema_signal = indicators.get("ema_cross_signal")
    ema_recent = indicators.get("ema_cross_recent", False)
    near_support = indicators.get("near_support", False)
    near_resistance = indicators.get("near_resistance", False)
    pts = indicators.get("price_to_support")
    ptr = indicators.get("price_to_resistance")
    sector_momentum = indicators.get("sector_momentum")
    macd_growing = macd.get("macd_growing", False)
    macd_shrinking = macd.get("macd_shrinking", False)
    macd_zero_up = macd.get("macd_zero_cross_up", False)
    macd_zero_down = macd.get("macd_zero_cross_down", False)
    adx = indicators.get("adx", 0)
    adx_strong = indicators.get("adx_strong_trend", False)
    adx_weak = indicators.get("adx_weak_trend", False)
    adx_bullish = indicators.get("adx_bullish", True)
    near_52w_high = indicators.get("near_52w_high", False)
    near_52w_low = indicators.get("near_52w_low", False)
    pct_from_high = indicators.get("pct_from_52w_high", -50)
    pct_from_low = indicators.get("pct_from_52w_low", 50)
    gap_up = indicators.get("gap_up", False)
    gap_down = indicators.get("gap_down", False)
    gap_large = indicators.get("gap_large", False)
    gap_pct = indicators.get("gap_pct", 0)
    candle_signal = indicators.get("candle_signal", "neutraal")
    candle_patterns = indicators.get("candle_patterns", [])

    # ADX trend multiplier — sterke trend versterkt signalen, zwakke trend verzwakt ze
    trend_mult = 1.2 if adx_strong else (0.8 if adx_weak else 1.0)

    # ── TIER 1: Trend & Momentum (zwaarst) ──────────────────────────────

    # MA20 trend (max +2.0 / -2.0)
    if ma20:
        if current > ma20 * 1.02:
            s = round(2.0 * trend_mult, 2)
            score += s
            reasons.append(f"Prijs ruim boven MA20 ({ma20:.2f}) — sterke korte termijn uptrend")
        elif current > ma20:
            s = round(1.0 * trend_mult, 2)
            score += s
            reasons.append(f"Prijs boven MA20 ({ma20:.2f}) — korte termijn uptrend")
        elif current < ma20 * 0.98:
            s = round(2.0 * trend_mult, 2)
            score -= s
            reasons.append(f"Prijs ruim onder MA20 ({ma20:.2f}) — sterke korte termijn downtrend")
        else:
            s = round(1.0 * trend_mult, 2)
            score -= s
            reasons.append(f"Prijs onder MA20 ({ma20:.2f}) — korte termijn downtrend")

    # MA50 (max +1.5 / -1.5)
    if ma50:
        if current > ma50:
            score += round(1.5 * trend_mult, 2)
            reasons.append(f"Prijs boven MA50 ({ma50:.2f}) — middellange termijn bullish")
        else:
            score -= round(1.5 * trend_mult, 2)
            reasons.append(f"Prijs onder MA50 ({ma50:.2f}) — middellange termijn bearish")

    # MA200 (max +1.0 / -0.3 — backtest: bear signal zwak)
    if ma200:
        if current > ma200:
            score += 1.0
            reasons.append(f"Prijs boven MA200 ({ma200:.2f}) — lange termijn bullish")
        else:
            score -= 0.3
            reasons.append(f"Prijs onder MA200 ({ma200:.2f}) — lange termijn bearish")

    # MA alignment bonus
    if ma20 and ma50 and ma200:
        if current > ma20 and ma20 > ma50 and ma50 > ma200:
            score += 1.0
            reasons.append("MA20 > MA50 > MA200 — perfect bullish uitlijning")
        elif current < ma20 and ma20 < ma50 and ma50 < ma200:
            score -= 1.0
            reasons.append("MA20 < MA50 < MA200 — perfect bearish uitlijning")

    # Momentum 5D (backtest: bearish momentum te dominant — max bear teruggebracht naar -1.5)
    if momentum > 5:
        score += 2.5
        reasons.append(f"5-daags momentum +{momentum:.1f}% — zeer sterke positieve trend")
    elif momentum > 3:
        score += 1.5
        reasons.append(f"5-daags momentum +{momentum:.1f}% — sterke positieve trend")
    elif momentum > 1:
        score += 0.5
        reasons.append(f"5-daags momentum +{momentum:.1f}% — licht positief")
    elif momentum < -12:
        score -= 1.5
        reasons.append(f"5-daags momentum {momentum:.1f}% — extreme negatieve trend")
    elif momentum < -8:
        score -= 1.2
        reasons.append(f"5-daags momentum {momentum:.1f}% — zeer sterke negatieve trend")
    elif momentum < -5:
        score -= 0.8
        reasons.append(f"5-daags momentum {momentum:.1f}% — sterke negatieve trend")
    elif momentum < -1:
        score -= 0.4
        reasons.append(f"5-daags momentum {momentum:.1f}% — licht negatief")

    # Momentum 10D + 20D — bevestigt of trend versnelt of vertraagt
    mom_confirms = 0
    mom_denies = 0
    if momentum_10d > 2:
        mom_confirms += 1
    elif momentum_10d < -2:
        mom_denies += 1
    if momentum_20d > 3:
        mom_confirms += 1
    elif momentum_20d < -3:
        mom_denies += 1

    if mom_confirms == 2 and momentum > 0:
        score += 0.8
        reasons.append(f"10D (+{momentum_10d:.1f}%) en 20D (+{momentum_20d:.1f}%) momentum bevestigen uptrend")
    elif mom_denies == 2 and momentum < 0:
        score -= 0.8
        reasons.append(f"10D ({momentum_10d:.1f}%) en 20D ({momentum_20d:.1f}%) momentum bevestigen downtrend")
    elif mom_confirms >= 1 and momentum > 0:
        score += 0.3
        reasons.append(f"20D momentum ({momentum_20d:.1f}%) ondersteunt positieve trend")
    elif mom_denies >= 1 and momentum < 0:
        score -= 0.3
        reasons.append(f"20D momentum ({momentum_20d:.1f}%) ondersteunt negatieve trend")

    # OBV bevestiging (backtest: bear signal omgekeerd gedrag — +0.5 contrarian)
    if obv_trend is not None and obv_confirmation is not None:
        if obv_confirmation and obv_trend == "stijgend":
            base = 2.0 if obv_strong else 1.5
            score += base
            label = "sterk " if obv_strong else ""
            reasons.append(f"OBV {label}bevestigt stijging — volume in lijn met prijs")
        elif obv_confirmation and obv_trend == "dalend":
            score += 0.5
            reasons.append("OBV dalend — contrair signaal (historisch zwak bearish)")
        elif not obv_confirmation:
            score -= 0.5
            reasons.append("OBV en prijs spreken elkaar tegen — onbetrouwbaar signaal")

    # Volume ratio (max +1.5 / -1.5)
    if volume_ratio > 2.0:
        if score > 0:
            score += 1.5
            reasons.append(f"Volume {volume_ratio:.1f}x hoger — sterk versterkt bullish signaal")
        else:
            score -= 1.5
            reasons.append(f"Volume {volume_ratio:.1f}x hoger — sterk versterkt bearish signaal")
    elif volume_ratio > 1.5:
        if score > 0:
            score += 1.0
            reasons.append(f"Volume {volume_ratio:.1f}x hoger dan gemiddeld — bevestigt uptrend")
        else:
            score -= 1.0
            reasons.append(f"Volume {volume_ratio:.1f}x hoger dan gemiddeld — bevestigt downtrend")

    # Volume profiel (meer stijgende of dalende volume-dagen)
    if volume_bullish:
        score += 0.5
        reasons.append("Volume profiel bullish — meer koopvolume dan verkoopvolume")
    elif volume_bearish:
        score -= 0.5
        reasons.append("Volume profiel bearish — meer verkoopvolume dan koopvolume")

    # Volume trend
    if volume_increasing:
        if score > 0:
            score += 0.5
            reasons.append("Volume neemt toe — groeiende interesse in uptrend")
        else:
            score -= 0.5
            reasons.append("Volume neemt toe bij dalende prijs — verkoopdruk neemt toe")

    # ── TIER 2: Timing (middel) ──────────────────────────────────────────

    # MACD (backtest: bull gewicht verlaagd naar +0.5; bear omgekeerd naar +0.3)
    hist = macd.get("histogram", 0)
    macd_val = macd.get("macd", 0)
    signal_val = macd.get("signal", 0)
    if hist > 0 and macd_val > signal_val:
        base_macd = 0.5
        if macd_growing:
            base_macd += 0.3
            reasons.append("MACD boven signaallijn en groeiend — versterkt bullish momentum")
        else:
            reasons.append("MACD boven signaallijn — bullish momentum")
        if macd_zero_up:
            base_macd += 0.2
            reasons.append("MACD kruist nul-lijn omhoog — sterk bullish signaal")
        score += base_macd
    elif hist < 0:
        if macd_shrinking:
            reasons.append("MACD onder signaallijn en verder dalend — contrair signaal")
        else:
            reasons.append("MACD onder signaallijn — contrair signaal (historisch zwak bearish)")
        if macd_zero_down:
            reasons.append("MACD kruist nul-lijn omlaag — contrair signaal")
        score += 0.3

    # RSI (backtest: oversold +1.5 beste signaal; overbought omgekeerd → +1.0 contrarian)
    if rsi < 30:
        score += 1.5
        reasons.append(f"RSI {rsi:.1f} — sterk oververkocht (bullish kans)")
    elif rsi < 40:
        score += 0.5
        reasons.append(f"RSI {rsi:.1f} — licht oververkocht")
    elif rsi > 75:
        score += 1.0
        reasons.append(f"RSI {rsi:.1f} — sterk overkocht (contrair bullish signaal)")
    elif rsi > 65:
        score += 0.3
        reasons.append(f"RSI {rsi:.1f} — licht overkocht (contrair)")
    else:
        reasons.append(f"RSI {rsi:.1f} — neutraal")

    # EMA crossover
    if ema_signal is not None:
        multiplier = 1.5 if ema_recent else 0.8
        if ema_signal == "bullish":
            score += multiplier
            label = "verse " if ema_recent else ""
            reasons.append(f"EMA9 boven EMA21 — {label}bullish crossover")
        else:
            score -= multiplier
            label = "verse " if ema_recent else ""
            reasons.append(f"EMA9 onder EMA21 — {label}bearish crossover")

    # Candlestick patronen (max +1.0 / -1.0)
    if candle_signal == "bullish":
        pattern_names = {
            "hammer": "Hammer patroon",
            "bullish_engulfing": "Bullish Engulfing patroon",
            "marubozu_bullish": "Bullish Marubozu"
        }
        matched = [pattern_names[p] for p in candle_patterns if p in pattern_names]
        if matched:
            score += 1.0
            reasons.append(f"{matched[0]} — bullish keerpunt signaal")
    elif candle_signal == "bearish":
        pattern_names = {
            "shooting_star": "Shooting Star patroon",
            "bearish_engulfing": "Bearish Engulfing patroon",
            "marubozu_bearish": "Bearish Marubozu"
        }
        matched = [pattern_names[p] for p in candle_patterns if p in pattern_names]
        if matched:
            score -= 1.0
            reasons.append(f"{matched[0]} — bearish keerpunt signaal")
    elif "doji" in candle_patterns:
        reasons.append("Doji patroon — onzekerheid, wacht op bevestiging")

    # Gap detectie (max +1.0 / -1.0)
    if gap_up:
        if gap_large:
            score += 1.0
            reasons.append(f"Grote gap omhoog (+{gap_pct:.1f}%) — sterke opening boven vorige slotkoers")
        else:
            score += 0.5
            reasons.append(f"Gap omhoog (+{gap_pct:.1f}%) — positieve opening")
    elif gap_down:
        if gap_large:
            score -= 1.0
            reasons.append(f"Grote gap omlaag ({gap_pct:.1f}%) — zwakke opening onder vorige slotkoers")
        else:
            score -= 0.5
            reasons.append(f"Gap omlaag ({gap_pct:.1f}%) — negatieve opening")

    # ── TIER 3: Context (licht) ──────────────────────────────────────────

    # Bollinger Bands (backtest: oversold +1.0; overbought omgekeerd → +0.5 contrarian)
    if bb_position is not None:
        if bb_position < 20:
            score += 1.0
            reasons.append(f"BB positie {bb_position:.0f}% — onderkant band (oversold zone)")
        elif bb_position > 80:
            score += 0.5
            reasons.append(f"BB positie {bb_position:.0f}% — bovenkant band (contrair bullish)")
        if bb_squeeze:
            reasons.append("Bollinger squeeze — uitbraak verwacht (richting onzeker)")

    # Stochastic RSI (max +0.5 / -0.5)
    if stoch_k is not None and stoch_d is not None:
        if stoch_k < 20 and stoch_d < 20:
            score += 0.5
            reasons.append(f"Stoch RSI sterk oversold (K={stoch_k:.0f}) — potentieel keerpunt")
        elif stoch_k > 80 and stoch_d > 80:
            score -= 0.5
            reasons.append(f"Stoch RSI sterk overbought (K={stoch_k:.0f}) — voorzichtigheid geboden")
        elif stoch_k > stoch_d and stoch_k < 50:
            score += 0.3
            reasons.append("Stoch RSI bullish crossover vanuit lage zone")
        elif stoch_k < stoch_d and stoch_k > 50:
            score -= 0.3
            reasons.append("Stoch RSI bearish crossover vanuit hoge zone")

    # Support/Resistance (max +0.5 / -0.8)
    if near_support and pts is not None and score > 0:
        score += 0.5
        reasons.append(f"Dicht bij steun ({pts:.1f}% erboven) — goede koop-zone")
    if near_resistance and ptr is not None and score < 0:
        score -= 0.8
        reasons.append(f"Dicht bij weerstand ({ptr:.1f}% eronder) — verkoopdruk verwacht")

    # 52-weeks high/low (max +0.5 / -0.8)
    if near_52w_high:
        score -= 0.8
        reasons.append(f"Dicht bij 52-weeks high ({pct_from_high:.1f}% eronder) — sterke weerstand verwacht")
    elif near_52w_low:
        score += 0.5
        reasons.append(f"Dicht bij 52-weeks low (+{pct_from_low:.1f}% erboven) — mogelijke bodem")

    # Sector momentum (max +1.5 / -2.5)
    if sector_momentum is not None:
        if sector_momentum > 2:
            score += 1.5
            reasons.append(f"Sector momentum +{sector_momentum:.1f}% — sector in sterke uptrend")
        elif sector_momentum > 0:
            score += 0.5
            reasons.append(f"Sector momentum +{sector_momentum:.1f}% — sector licht positief")
        elif sector_momentum < -3:
            score -= 2.5
            reasons.append(f"Sector momentum {sector_momentum:.1f}% — sector in sterke neergang")
        elif sector_momentum < -2:
            score -= 1.5
            reasons.append(f"Sector momentum {sector_momentum:.1f}% — sector onder druk")
        elif sector_momentum < 0:
            score -= 1.0
            reasons.append(f"Sector momentum {sector_momentum:.1f}% — sector licht negatief")

    # Patroon-matching (max +1.0 / -1.0)
    pattern_up = 0
    pattern_down = 0
    if patterns:
        for p in patterns:
            change = p.get("next_day_change") or p.get("next_return") or p.get("change") or 0
            if change > 0.3:
                pattern_up += 1
            elif change < -0.3:
                pattern_down += 1

        total_patterns = pattern_up + pattern_down
        if total_patterns >= 3:
            if pattern_up >= 4:
                score += 1.0
                reasons.append(f"Historisch patroon: {pattern_up}/5 vergelijkbare dagen stegen")
            elif pattern_up == 3:
                score += 0.5
                reasons.append(f"Historisch patroon: {pattern_up}/5 vergelijkbare dagen stegen — licht bullish")
            elif pattern_down >= 4:
                score -= 1.0
                reasons.append(f"Historisch patroon: {pattern_down}/5 vergelijkbare dagen daalden")
            elif pattern_down == 3:
                score -= 0.5
                reasons.append(f"Historisch patroon: {pattern_down}/5 vergelijkbare dagen daalden — licht bearish")

    # ── REGIME MODIFIER ──────────────────────────────────────────────────
    if regime == "bull":
        if score > 0:
            score *= 1.2
        else:
            score *= 0.7
    elif regime == "bear":
        if score < 0:
            score *= 1.2
        else:
            score *= 0.7

    # ── CONFLICT REGELS ──────────────────────────────────────────────────

    mixed_signals = False

    if momentum < -8 and volume_ratio > 1.5:
        if score > -2.0:
            score = -2.0
        reasons.append("Sterke daling met hoog volume — score begrensd op bearish")

    if score > 2.0 and obv_confirmation is False:
        score = min(score, 2.0)
        reasons.append("Bullish score zonder OBV-bevestiging — voorzichtigheid geboden")

    if sector_momentum is not None and sector_momentum < -2 and score < 1.5:
        direction_override = "Down"
    else:
        direction_override = None

    positive_count = sum(1 for r in reasons if any(w in r.lower() for w in [
        "bullish", "boven", "oversold", "steun", "positief", "stijging", "uptrend",
        "groeiend", "omhoog", "hammer", "engulfing", "gap omhoog"
    ]))
    negative_count = sum(1 for r in reasons if any(w in r.lower() for w in [
        "bearish", "onder", "overbought", "weerstand", "negatief", "daling",
        "downtrend", "dalend", "omlaag", "shooting", "gap omlaag"
    ]))
    if positive_count >= 2 and negative_count >= 2:
        mixed_signals = True

    # ── SCORE NAAR RICHTING ──────────────────────────────────────────────

    max_score = 22.0
    normalized = max(-1.0, min(1.0, score / max_score))

    if direction_override:
        direction = direction_override
    elif normalized > 0.15:
        direction = "Up"
    elif normalized < -0.15:
        direction = "Down"
    else:
        direction = "Flat"

    # ── CONFIDENCE ───────────────────────────────────────────────────────

    confidence = 50 + int(normalized * 35)

    if obv_confirmation:
        confidence += 4
    if obv_strong:
        confidence += 2
    if ema_recent:
        confidence += 3
    if volume_ratio > 1.5:
        confidence += 3
    if macd_growing and direction == "Up":
        confidence += 2
    if macd_shrinking and direction == "Down":
        confidence += 2
    if pattern_up >= 4 or pattern_down >= 4:
        confidence += 3
    if adx_strong and adx_bullish and direction == "Up":
        confidence += 3
    if adx_strong and not adx_bullish and direction == "Down":
        confidence += 3
    if candle_signal == direction.lower() if direction != "Flat" else False:
        confidence += 2

    if mixed_signals:
        confidence -= 8
    if volatility > 120:
        confidence -= 10
    elif volatility > 80:
        confidence -= 5
    if adx_weak:
        confidence -= 4

    confidence = max(30, min(85, confidence))

    # ── DOELPRIJS ────────────────────────────────────────────────────────

    price_low, price_high = _price_range(current, atr, volatility)

    if atr and atr > 0:
        vol_multiplier = 1.05 if bb_squeeze else 0.8
        move = atr * vol_multiplier
    else:
        daily_vol_factor = (volatility / 100) / (252 ** 0.5)
        vol_multiplier = 1.05 if bb_squeeze else 0.7
        move = current * daily_vol_factor * vol_multiplier

    if direction == "Up":
        target = round(current + move, 4)
    elif direction == "Down":
        target = round(current - move, 4)
    else:
        target = current

    return {
        "direction": direction,
        "confidence": confidence,
        "target_price": target,
        "price_range_low": price_low,
        "price_range_high": price_high,
        "score": score,
        "reasons": reasons,
    }


def _build_reasons_from_ml(ml_result: dict, indicators: dict) -> list[str]:
    reasons = []
    direction = ml_result["direction"]
    if ml_result.get("forced_flat"):
        reasons.append(
            "Signalen niet sterk genoeg voor een richting voorspelling"
            " — te veel onzekerheid voor betrouwbare Up/Down voorspelling"
        )
    reasons.append(f"Model voorspelt {direction} met {ml_result['confidence']}% zekerheid")
    for feat in ml_result.get("top_features", [])[:3]:
        reasons.append(f"{feat['name']}: waarde {feat['value']} (belang: {feat['importance']}%)")
    if ml_result["confidence"] < 60:
        reasons.append(
            "Lage zekerheid — signalen zijn niet overtuigend genoeg voor een sterke voorspelling"
        )
    return reasons


def predict(indicators: dict, df: pd.DataFrame, patterns: Optional[list] = None, regime: str = "neutral", ticker: str = "") -> dict:
    import traceback as _tb
    current = indicators.get("current_price", 0) or 0
    try:
        ml_result = predict_ml(indicators, regime, current, ticker=ticker)
    except Exception as _e:
        print(f"ML predict faalt: {_e}", flush=True)
        _tb.print_exc()
        ml_result = None

    if ml_result is not None:
        reasons = _build_reasons_from_ml(ml_result, indicators)
        tr = ml_result.get("target_range", (current, current))
        return {
            "direction":        ml_result["direction"],
            "confidence":       ml_result["confidence"],
            "target_price":     ml_result["target_price"],
            "price_range_low":  tr[0],
            "price_range_high": tr[1],
            "score":            0.0,
            "reasons":          reasons,
            "model_type":       "ml",
            "probabilities":    ml_result["probabilities"],
            "top_features":     ml_result["top_features"],
            "news_sentiment":   ml_result.get("news_sentiment", 0.0),
            "consensus":        ml_result.get("consensus", 0.0),
            "models_used":      ml_result.get("models_used", 1),
            "signal_strength":        ml_result.get("signal_strength", "sterk"),
            "betrouwbaarheids_tekst": ml_result.get("betrouwbaarheids_tekst", ""),
        }

    legacy = _predict_legacy(indicators, df, patterns, regime)
    legacy["model_type"] = "legacy"
    return legacy
