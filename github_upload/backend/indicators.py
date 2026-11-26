import time

import pandas as pd
import numpy as np

# Module-brede cache voor SPY momentum (zie _get_spy_momentum)
_spy_momentum_cache: dict = {"mom5": None, "mom20": None, "ts": 0.0}
_SPY_MOMENTUM_TTL = 3600  # 1 uur


def detect_market_regime(spy_df: pd.DataFrame) -> str:
    """Bepaalt marktregime op basis van S&P500 vs MA200.
    Returns: 'bull', 'bear', of 'neutral'
    """
    if len(spy_df) < 200:
        return "neutral"
    current = spy_df["Close"].iloc[-1]
    ma200 = spy_df["Close"].rolling(200).mean().iloc[-1]
    ma50 = spy_df["Close"].rolling(50).mean().iloc[-1]
    if current > ma200 and ma50 > ma200:
        return "bull"
    if current < ma200 and ma50 < ma200:
        return "bear"
    return "neutral"


def compute_rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0, other=np.nan)
    rsi = rs.apply(lambda x: 100.0 if np.isnan(x) else 100 - (100 / (1 + x)))
    return round(float(rsi.iloc[-1]), 2)


def compute_ma(closes: pd.Series, period: int) -> float | None:
    if len(closes) < period:
        return None
    return round(float(closes.rolling(period).mean().iloc[-1]), 4)


def compute_macd(closes: pd.Series) -> dict:
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    hist_vals = histogram.iloc[-3:]
    macd_growing = bool(
        len(hist_vals) == 3
        and hist_vals.iloc[-1] > hist_vals.iloc[-2] > hist_vals.iloc[-3]
    )
    macd_shrinking = bool(
        len(hist_vals) == 3
        and hist_vals.iloc[-1] < hist_vals.iloc[-2] < hist_vals.iloc[-3]
    )

    # Zero-cross: MACD lijn kruist de nul-lijn
    macd_vals = macd_line.iloc[-3:]
    zero_cross_up = bool(macd_vals.iloc[-1] > 0 and macd_vals.iloc[-2] <= 0)
    zero_cross_down = bool(macd_vals.iloc[-1] < 0 and macd_vals.iloc[-2] >= 0)

    return {
        "macd": round(float(macd_line.iloc[-1]), 4),
        "signal": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
        "macd_growing": macd_growing,
        "macd_shrinking": macd_shrinking,
        "macd_zero_cross_up": zero_cross_up,
        "macd_zero_cross_down": zero_cross_down,
    }


def compute_volume_ratio(volumes: pd.Series, period: int = 20) -> float:
    if len(volumes) < period + 1:
        return 1.0
    avg_vol = volumes.iloc[-(period + 1):-1].mean()
    if avg_vol == 0:
        return 1.0
    return round(float(volumes.iloc[-1] / avg_vol), 2)


def compute_volume_trend(volumes: pd.Series) -> dict | None:
    if len(volumes) < 20:
        return None
    avg_5 = float(volumes.iloc[-5:].mean())
    avg_20 = float(volumes.iloc[-20:].mean())
    if avg_20 == 0:
        return None
    ratio = round(avg_5 / avg_20, 2)
    return {
        "volume_trend_ratio": ratio,
        "volume_increasing": ratio > 1.1,
    }


def compute_volume_profile(closes: pd.Series, volumes: pd.Series, period: int = 20) -> dict | None:
    if len(closes) < period + 1:
        return None
    up_volume = 0.0
    down_volume = 0.0
    for i in range(1, period + 1):
        idx = -(period + 1) + i
        if closes.iloc[idx] > closes.iloc[idx - 1]:
            up_volume += float(volumes.iloc[idx])
        else:
            down_volume += float(volumes.iloc[idx])
    total = up_volume + down_volume
    if total == 0:
        return None
    up_ratio = round(up_volume / total, 3)
    return {
        "volume_up_ratio": up_ratio,
        "volume_bullish_profile": up_ratio > 0.55,
        "volume_bearish_profile": up_ratio < 0.45,
    }


def compute_momentum(closes: pd.Series, period: int = 5) -> float:
    if len(closes) < period + 1:
        return 0.0
    start = closes.iloc[-(period + 1)]
    end = closes.iloc[-1]
    if start == 0:
        return 0.0
    return round(float((end - start) / start * 100), 2)


def compute_volatility(closes: pd.Series, period: int = 20) -> float:
    if len(closes) < period:
        return 0.0
    returns = closes.pct_change().dropna()
    return round(float(returns.iloc[-period:].std() * np.sqrt(252) * 100), 2)


def compute_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    if len(df) < period + 1:
        return None
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
    return round(atr, 4)


def compute_adx(df: pd.DataFrame, period: int = 14) -> dict | None:
    if len(df) < period * 2:
        return None
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = float(dx.ewm(span=period, adjust=False).mean().iloc[-1])
    plus_di_val = float(plus_di.iloc[-1])
    minus_di_val = float(minus_di.iloc[-1])

    if np.isnan(adx):
        return None

    return {
        "adx": round(adx, 2),
        "adx_strong_trend": adx > 25,
        "adx_weak_trend": adx < 20,
        "adx_bullish": plus_di_val > minus_di_val,
    }


def compute_bollinger(closes: pd.Series, period: int = 20) -> dict | None:
    if len(closes) < period:
        return None
    ma = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    upper = float((ma + 2 * std).iloc[-1])
    lower = float((ma - 2 * std).iloc[-1])
    ma_val = float(ma.iloc[-1])
    price = float(closes.iloc[-1])
    band_width = upper - lower
    bb_position = round((price - lower) / band_width * 100, 2) if band_width > 0 else 50.0
    bb_squeeze = (band_width / ma_val) < 0.05 if ma_val != 0 else False
    return {
        "upper_band": round(upper, 4),
        "lower_band": round(lower, 4),
        "bb_position": bb_position,
        "bb_squeeze": bb_squeeze,
    }


def compute_stoch_rsi(closes: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> dict | None:
    if len(closes) < rsi_period + stoch_period + 3:
        return None
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0, other=np.nan)
    rsi_series = rs.apply(lambda x: 100.0 if np.isnan(x) else 100 - (100 / (1 + x)))
    rsi_min = rsi_series.rolling(stoch_period).min()
    rsi_max = rsi_series.rolling(stoch_period).max()
    rsi_range = (rsi_max - rsi_min).replace(0, np.nan)
    stoch = (rsi_series - rsi_min) / rsi_range * 100
    stoch_k = float(stoch.iloc[-1])
    stoch_d = float(stoch.rolling(3).mean().iloc[-1])
    if np.isnan(stoch_k) or np.isnan(stoch_d):
        return None
    return {
        "stoch_rsi_k": round(stoch_k, 2),
        "stoch_rsi_d": round(stoch_d, 2),
    }


def compute_obv(closes: pd.Series, volumes: pd.Series) -> dict | None:
    if len(closes) < 21:
        return None
    direction = np.sign(closes.diff()).fillna(0)
    obv_series = (direction * volumes).cumsum()

    last5 = obv_series.iloc[-5:]
    obv_trend_5 = "stijgend" if float(last5.iloc[-1]) > float(last5.iloc[0]) else "dalend"

    last20 = obv_series.iloc[-20:]
    obv_trend_20 = "stijgend" if float(last20.iloc[-1]) > float(last20.iloc[0]) else "dalend"

    price_trend = "stijgend" if float(closes.iloc[-1]) > float(closes.iloc[-5]) else "dalend"

    if len(obv_series) >= 10:
        obv_tail = obv_series.iloc[-10:].values.astype(float)
        x = np.arange(10, dtype=float)
        if obv_tail.std() > 0:
            slope = float(np.polyfit(x, obv_tail, 1)[0])
            mean_abs = float(np.abs(obv_tail).mean()) or 1.0
            obv_slope = round(slope / mean_abs, 6)
        else:
            obv_slope = 0.0
    else:
        obv_slope = 0.0

    return {
        "obv_trend": obv_trend_5,
        "obv_confirmation": obv_trend_5 == price_trend,
        "obv_strong": obv_trend_5 == obv_trend_20,
        "obv_slope": obv_slope,
    }


def compute_ema_cross(closes: pd.Series, fast: int = 9, slow: int = 21) -> dict | None:
    if len(closes) < slow + 4:
        return None
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    ema9_val = round(float(ema_fast.iloc[-1]), 4)
    ema21_val = round(float(ema_slow.iloc[-1]), 4)
    diff = ema_fast - ema_slow
    tail = diff.iloc[-4:]
    recent_cross = any(
        (tail.iloc[i] > 0) != (tail.iloc[i - 1] > 0) for i in range(1, len(tail))
    )
    return {
        "ema9": ema9_val,
        "ema21": ema21_val,
        "ema_cross_signal": "bullish" if ema9_val > ema21_val else "bearish",
        "ema_cross_recent": bool(recent_cross),
    }


def compute_support_resistance(closes: pd.Series) -> dict | None:
    if len(closes) < 30:
        return None
    lookback = min(60, len(closes))
    prices = closes.iloc[-lookback:].values
    price = float(closes.iloc[-1])

    supports = []
    resistances = []

    for i in range(2, len(prices) - 2):
        if (prices[i] < prices[i - 1] and prices[i] < prices[i - 2]
                and prices[i] < prices[i + 1] and prices[i] < prices[i + 2]):
            supports.append(float(prices[i]))
        if (prices[i] > prices[i - 1] and prices[i] > prices[i - 2]
                and prices[i] > prices[i + 1] and prices[i] > prices[i + 2]):
            resistances.append(float(prices[i]))

    if not supports:
        supports = [float(closes.iloc[-20:].min())]
    if not resistances:
        resistances = [float(closes.iloc[-20:].max())]

    supports_below = [s for s in supports if s < price]
    resistances_above = [r for r in resistances if r > price]

    support = max(supports_below) if supports_below else min(supports)
    resistance = min(resistances_above) if resistances_above else max(resistances)

    pts = round((price - support) / support * 100, 2) if support != 0 else 0.0
    ptr = round((resistance - price) / resistance * 100, 2) if resistance != 0 else 0.0

    return {
        "support": round(support, 4),
        "resistance": round(resistance, 4),
        "price_to_support": pts,
        "price_to_resistance": ptr,
        "near_support": pts < 3.0,
        "near_resistance": ptr < 3.0,
    }


def compute_52week(closes: pd.Series) -> dict | None:
    if len(closes) < 50:
        return None
    lookback = min(252, len(closes))
    period_closes = closes.iloc[-lookback:]
    high_52 = float(period_closes.max())
    low_52 = float(period_closes.min())
    price = float(closes.iloc[-1])

    if high_52 == 0:
        return None

    pct_from_high = round((price - high_52) / high_52 * 100, 2)
    pct_from_low = round((price - low_52) / low_52 * 100, 2)

    return {
        "high_52w": round(high_52, 4),
        "low_52w": round(low_52, 4),
        "pct_from_52w_high": pct_from_high,
        "pct_from_52w_low": pct_from_low,
        "near_52w_high": pct_from_high > -5,
        "near_52w_low": pct_from_low < 10,
    }


def compute_gap(df: pd.DataFrame) -> dict | None:
    if len(df) < 2:
        return None
    prev_close = float(df["Close"].iloc[-2])
    today_open = float(df["Open"].iloc[-1])
    if prev_close == 0:
        return None
    gap_pct = round((today_open - prev_close) / prev_close * 100, 2)
    return {
        "gap_pct": gap_pct,
        "gap_up": gap_pct > 0.5,
        "gap_down": gap_pct < -0.5,
        "gap_large": abs(gap_pct) > 2.0,
    }


def compute_candlestick(df: pd.DataFrame) -> dict | None:
    if len(df) < 3:
        return None

    o = float(df["Open"].iloc[-1])
    h = float(df["High"].iloc[-1])
    l = float(df["Low"].iloc[-1])
    c = float(df["Close"].iloc[-1])
    body = abs(c - o)
    total_range = h - l
    if total_range == 0:
        return None

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body_ratio = body / total_range

    # Vorige twee kaarsen voor context
    prev_c = float(df["Close"].iloc[-2])
    prev2_c = float(df["Close"].iloc[-3])
    prev_o = float(df["Open"].iloc[-2])

    patterns = []

    # Doji — lichaam zeer klein
    if body_ratio < 0.1:
        patterns.append("doji")

    # Hammer — kleine body bovenaan, lange onderste wick, bullish
    if (lower_wick > body * 2 and upper_wick < body * 0.5
            and c > o and prev_c < prev2_c):
        patterns.append("hammer")

    # Shooting Star — kleine body onderaan, lange bovenste wick, bearish
    if (upper_wick > body * 2 and lower_wick < body * 0.5
            and c < o and prev_c > prev2_c):
        patterns.append("shooting_star")

    # Bullish Engulfing
    if (c > o and prev_c < prev_o
            and c > prev_o and o < prev_c):
        patterns.append("bullish_engulfing")

    # Bearish Engulfing
    if (c < o and prev_c > prev_o
            and c < prev_o and o > prev_c):
        patterns.append("bearish_engulfing")

    # Marubozu bullish — grote groene kaars zonder wicks
    if c > o and body_ratio > 0.85:
        patterns.append("marubozu_bullish")

    # Marubozu bearish — grote rode kaars zonder wicks
    if c < o and body_ratio > 0.85:
        patterns.append("marubozu_bearish")

    bullish_patterns = ["hammer", "bullish_engulfing", "marubozu_bullish"]
    bearish_patterns = ["shooting_star", "bearish_engulfing", "marubozu_bearish"]

    candle_signal = "neutraal"
    if any(p in bullish_patterns for p in patterns):
        candle_signal = "bullish"
    elif any(p in bearish_patterns for p in patterns):
        candle_signal = "bearish"

    return {
        "candle_patterns": patterns,
        "candle_signal": candle_signal,
    }


def compute_trend_consistency(closes: pd.Series, period: int = 10) -> float:
    """Aandeel positieve sluitdagen in de laatste `period` dagen (0-1)."""
    if len(closes) < period + 1:
        return 0.5
    diffs = closes.diff().iloc[-period:]
    positive = float((diffs > 0).sum())
    return round(positive / period, 4)


def compute_rsi_trend(closes: pd.Series, period: int = 14, lookback: int = 5) -> float:
    """Huidige RSI minus RSI `lookback` dagen geleden."""
    if len(closes) < period + lookback + 1:
        return 0.0
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0, other=np.nan)
    rsi = rs.apply(lambda x: 100.0 if np.isnan(x) else 100 - (100 / (1 + x)))
    cur = float(rsi.iloc[-1])
    past = float(rsi.iloc[-1 - lookback])
    if np.isnan(cur) or np.isnan(past):
        return 0.0
    return round(cur - past, 4)


def compute_macd_acceleration(closes: pd.Series, lookback: int = 3) -> float:
    """Huidig MACD-histogram minus dat van `lookback` dagen geleden."""
    if len(closes) < 26 + lookback + 1:
        return 0.0
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    cur = float(histogram.iloc[-1])
    past = float(histogram.iloc[-1 - lookback])
    if np.isnan(cur) or np.isnan(past):
        return 0.0
    return round(cur - past, 4)


def _get_spy_momentum() -> tuple[float, float]:
    """5d/20d momentum van SPY via get_spy_data(), module-breed gecached. Fallback (0.0, 0.0)."""
    now = time.time()
    if _spy_momentum_cache["mom5"] is not None and (now - _spy_momentum_cache["ts"]) < _SPY_MOMENTUM_TTL:
        return _spy_momentum_cache["mom5"], _spy_momentum_cache["mom20"]
    try:
        from data import get_spy_data
        spy = get_spy_data()
        if spy is not None and len(spy) >= 21:
            closes = spy["Close"]
            mom5 = compute_momentum(closes, 5)
            mom20 = compute_momentum(closes, 20)
            _spy_momentum_cache.update({"mom5": mom5, "mom20": mom20, "ts": now})
            return mom5, mom20
    except Exception:
        pass
    return 0.0, 0.0


def compute_all(df: pd.DataFrame, ticker: str = "") -> dict:
    closes = df["Close"]
    volumes = df["Volume"]

    result = {
        "rsi": compute_rsi(closes),
        "ma20": compute_ma(closes, 20),
        "ma50": compute_ma(closes, 50),
        "ma200": compute_ma(closes, 200),
        "macd": compute_macd(closes),
        "volume_ratio": compute_volume_ratio(volumes),
        "momentum_5d": compute_momentum(closes, 5),
        "momentum_10d": compute_momentum(closes, 10),
        "momentum_20d": compute_momentum(closes, 20),
        "volatility_20d": compute_volatility(closes, 20),
        "current_price": round(float(closes.iloc[-1]), 4),
    }

    bollinger = compute_bollinger(closes)
    if bollinger:
        result.update(bollinger)

    stoch_rsi = compute_stoch_rsi(closes)
    if stoch_rsi:
        result.update(stoch_rsi)

    obv = compute_obv(closes, volumes)
    if obv:
        result.update(obv)

    ema_cross = compute_ema_cross(closes)
    if ema_cross:
        result.update(ema_cross)

    sr = compute_support_resistance(closes)
    if sr:
        result.update(sr)

    vol_trend = compute_volume_trend(volumes)
    if vol_trend:
        result.update(vol_trend)

    vol_profile = compute_volume_profile(closes, volumes)
    if vol_profile:
        result.update(vol_profile)

    atr = compute_atr(df)
    if atr:
        result["atr"] = atr

    adx = compute_adx(df)
    if adx:
        result.update(adx)

    week52 = compute_52week(closes)
    if week52:
        result.update(week52)

    gap = compute_gap(df)
    if gap:
        result.update(gap)

    candle = compute_candlestick(df)
    if candle:
        result.update(candle)

    # ── DEEL 1: extra features met veilige fallbacks ──
    vol20 = result["volatility_20d"]
    if len(closes) >= 60:
        vol60 = compute_volatility(closes, 60)
        result["volatility_60d"] = vol60 if vol60 > 0 else vol20
    else:
        result["volatility_60d"] = vol20

    result["trend_consistency"] = compute_trend_consistency(closes)
    result["rsi_trend"] = compute_rsi_trend(closes)
    result["macd_acceleration"] = compute_macd_acceleration(closes)

    spy5, spy20 = _get_spy_momentum()
    result["spy_momentum_5d"] = spy5
    result["spy_momentum_20d"] = spy20

    # Sector-momentum (20d) via ETF-mapping; SPY 20d-momentum als fallback.
    sector_mom_20 = None
    if ticker:
        try:
            from data import get_sector_momentum_20d
            sector_mom_20 = get_sector_momentum_20d(ticker)
        except Exception:
            sector_mom_20 = None
    if sector_mom_20 is None:
        sector_mom_20 = spy20
    result["sector_momentum_20d"] = sector_mom_20

    # Macro-context (live): VIX, 10y-rente, dollar-index 5d change.
    try:
        from data import get_macro_current
        macro = get_macro_current() or {}
    except Exception:
        macro = {}
    result["vix_level"]      = float(macro.get("vix_level") or 0.0)
    result["tnx_level"]      = float(macro.get("tnx_level") or 0.0)
    result["dxy_change_5d"]  = float(macro.get("dxy_change_5d") or 0.0)

    return result
