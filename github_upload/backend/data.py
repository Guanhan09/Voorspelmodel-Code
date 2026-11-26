import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

_spy_cache: dict = {"df": None, "ts": 0.0}
_SPY_CACHE_TTL = 3600  # 1 uur

_market_cache: dict = {"data": None, "ts": 0.0}
MARKET_CACHE_TTL = 300  # 5 minuten

_info_cache: dict = {}
INFO_CACHE_TTL = 1800  # 30 minuten

# Sector-ETF mapping voor de meest voorkomende S&P500/NASDAQ tickers.
# Tickers die hier niet in staan vallen terug op SPY.
SECTOR_ETF_MAP: dict[str, str] = {
    # XLK — Technology
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "ORCL": "XLK",
    "CRM": "XLK", "ADBE": "XLK", "CSCO": "XLK", "AMD": "XLK", "INTC": "XLK",
    "IBM": "XLK", "QCOM": "XLK", "NOW": "XLK", "INTU": "XLK", "TXN": "XLK",
    "AMAT": "XLK", "MU": "XLK", "LRCX": "XLK", "KLAC": "XLK", "ANET": "XLK",
    "PANW": "XLK", "FTNT": "XLK", "CRWD": "XLK", "SNPS": "XLK", "CDNS": "XLK",
    "ADI": "XLK", "MRVL": "XLK", "ROP": "XLK", "ACN": "XLK", "IT": "XLK",
    "FICO": "XLK", "MSI": "XLK", "GLW": "XLK", "HPQ": "XLK", "DELL": "XLK",
    "WDC": "XLK", "STX": "XLK", "NTAP": "XLK", "SMCI": "XLK", "ON": "XLK",
    # XLF — Financials
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
    "C": "XLF", "BLK": "XLF", "AXP": "XLF", "SCHW": "XLF", "USB": "XLF",
    "PNC": "XLF", "TFC": "XLF", "COF": "XLF", "BK": "XLF", "TRV": "XLF",
    "AIG": "XLF", "ALL": "XLF", "MET": "XLF", "PRU": "XLF", "AFL": "XLF",
    "CB": "XLF", "MMC": "XLF", "AON": "XLF", "ICE": "XLF", "CME": "XLF",
    "SPGI": "XLF", "MCO": "XLF", "FIS": "XLF", "V": "XLF", "MA": "XLF",
    "PYPL": "XLF",
    # XLV — Health Care
    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "PFE": "XLV", "ABBV": "XLV",
    "MRK": "XLV", "TMO": "XLV", "ABT": "XLV", "DHR": "XLV", "BMY": "XLV",
    "AMGN": "XLV", "ELV": "XLV", "CVS": "XLV", "MDT": "XLV", "GILD": "XLV",
    "ISRG": "XLV", "SYK": "XLV", "REGN": "XLV", "VRTX": "XLV", "BSX": "XLV",
    "ZTS": "XLV", "CI": "XLV", "HUM": "XLV", "BDX": "XLV",
    # XLY — Consumer Discretionary
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",
    "LOW": "XLY", "SBUX": "XLY", "TJX": "XLY", "BKNG": "XLY", "ABNB": "XLY",
    "ORLY": "XLY", "GM": "XLY", "F": "XLY", "MAR": "XLY", "HLT": "XLY",
    "CMG": "XLY", "AZO": "XLY", "ROST": "XLY", "BBY": "XLY", "EBAY": "XLY",
    # XLP — Consumer Staples
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP", "COST": "XLP",
    "PM": "XLP", "MO": "XLP", "CL": "XLP", "TGT": "XLP", "KMB": "XLP",
    "GIS": "XLP", "KR": "XLP", "SYY": "XLP", "MDLZ": "XLP", "EL": "XLP",
    "MNST": "XLP", "KHC": "XLP", "STZ": "XLP", "HSY": "XLP",
    # XLE — Energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "EOG": "XLE", "MPC": "XLE",
    "PSX": "XLE", "SLB": "XLE", "OXY": "XLE", "VLO": "XLE", "HES": "XLE",
    "FANG": "XLE", "WMB": "XLE", "KMI": "XLE", "OKE": "XLE",
    # XLI — Industrials
    "CAT": "XLI", "BA": "XLI", "GE": "XLI", "HON": "XLI", "UNP": "XLI",
    "RTX": "XLI", "LMT": "XLI", "UPS": "XLI", "FDX": "XLI", "DE": "XLI",
    "NOC": "XLI", "MMM": "XLI", "GD": "XLI", "ETN": "XLI", "EMR": "XLI",
    "ITW": "XLI", "CSX": "XLI", "NSC": "XLI", "WM": "XLI", "RSG": "XLI",
    # XLB — Materials
    "LIN": "XLB", "APD": "XLB", "SHW": "XLB", "ECL": "XLB", "FCX": "XLB",
    "NEM": "XLB", "NUE": "XLB", "DD": "XLB", "DOW": "XLB", "CTVA": "XLB",
    "MLM": "XLB", "VMC": "XLB", "IFF": "XLB", "PPG": "XLB",
    # XLRE — Real Estate
    "PLD": "XLRE", "AMT": "XLRE", "EQIX": "XLRE", "CCI": "XLRE", "PSA": "XLRE",
    "WELL": "XLRE", "SPG": "XLRE", "O": "XLRE", "DLR": "XLRE", "AVB": "XLRE",
    "EQR": "XLRE", "VICI": "XLRE",
    # XLU — Utilities
    "NEE": "XLU", "SO": "XLU", "DUK": "XLU", "AEP": "XLU", "SRE": "XLU",
    "D": "XLU", "EXC": "XLU", "XEL": "XLU", "PCG": "XLU", "AES": "XLU",
    # XLC — Communication Services
    "GOOGL": "XLC", "GOOG": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC",
    "T": "XLC", "VZ": "XLC", "TMUS": "XLC", "CMCSA": "XLC", "EA": "XLC",
    "TTWO": "XLC", "WBD": "XLC", "CHTR": "XLC", "MTCH": "XLC", "OMC": "XLC",
}

_sector_etf_cache: dict[str, dict] = {}  # ETF -> {"df": ..., "ts": ...}
_SECTOR_ETF_CACHE_TTL = 3600  # 1 uur per ETF

_macro_history_cache: dict = {"data": None, "ts": 0.0}
_macro_current_cache: dict = {"data": None, "ts": 0.0}
_MACRO_CACHE_TTL = 3600  # 1 uur


def get_spy_data() -> pd.DataFrame | None:
    """Haalt SPY (S&P500 ETF) data op met 1 jaar history. Gecached voor 1 uur."""
    now = time.time()
    if _spy_cache["df"] is not None and (now - _spy_cache["ts"]) < _SPY_CACHE_TTL:
        return _spy_cache["df"]
    try:
        df = _fetch_yfinance("SPY", period="1y")
        _spy_cache["df"] = df
        _spy_cache["ts"] = now
        return df
    except Exception as e:
        logger.warning("Could not fetch SPY data: %s", e)
        return _spy_cache["df"]  # geef gecachte data terug bij fout


def _get_sector_etf_data(etf: str) -> pd.DataFrame | None:
    """Haalt 1 jaar data voor een sector-ETF op. Gecached voor 1 uur per ETF."""
    now = time.time()
    cached = _sector_etf_cache.get(etf)
    if cached and cached["df"] is not None and (now - cached["ts"]) < _SECTOR_ETF_CACHE_TTL:
        return cached["df"]
    try:
        df = _fetch_yfinance(etf, period="1y")
        _sector_etf_cache[etf] = {"df": df, "ts": now}
        return df
    except Exception as e:
        logger.warning("Could not fetch sector ETF %s: %s", etf, e)
        return cached["df"] if cached else None


def _fetch_dxy_series(period: str) -> pd.Series | None:
    """Probeer DX-Y.NYB; bij fout terugvallen op DX=F."""
    for sym in ("DX-Y.NYB", "DX=F"):
        try:
            df = _fetch_yfinance(sym, period=period)
            if df is not None and not df.empty and "Close" in df.columns:
                return df["Close"]
        except Exception as e:
            logger.warning("DXY fetch via %s mislukt: %s", sym, e)
    return None


def get_macro_history() -> dict | None:
    """Haalt 5j slotkoersen op van VIX, TNX en de dollar-index.
    Returns: dict met keys 'vix', 'tnx', 'dxy' (elk een pandas Series) of None bij fout.
    Gecached voor 1 uur.
    """
    now = time.time()
    if _macro_history_cache["data"] is not None and (now - _macro_history_cache["ts"]) < _MACRO_CACHE_TTL:
        return _macro_history_cache["data"]

    result: dict = {"vix": None, "tnx": None, "dxy": None}
    try:
        vix_df = _fetch_yfinance("^VIX", period="5y")
        result["vix"] = vix_df["Close"]
    except Exception as e:
        logger.warning("VIX history mislukt: %s", e)
    try:
        tnx_df = _fetch_yfinance("^TNX", period="5y")
        result["tnx"] = tnx_df["Close"]
    except Exception as e:
        logger.warning("TNX history mislukt: %s", e)
    result["dxy"] = _fetch_dxy_series("5y")

    if all(v is None for v in result.values()):
        return _macro_history_cache["data"]

    _macro_history_cache["data"] = result
    _macro_history_cache["ts"] = now
    return result


def get_macro_current() -> dict:
    """Meest recente macro-waarden voor live voorspelling.
    Returns: dict met 'vix_level', 'tnx_level', 'dxy_change_5d' (None bij fout).
    Gecached voor 1 uur.
    """
    now = time.time()
    if _macro_current_cache["data"] is not None and (now - _macro_current_cache["ts"]) < _MACRO_CACHE_TTL:
        return _macro_current_cache["data"]

    result = {"vix_level": None, "tnx_level": None, "dxy_change_5d": None}

    def _fetch_vix():
        df = _fetch_yfinance("^VIX", period="3mo")
        return round(float(df["Close"].iloc[-1]), 4)

    def _fetch_tnx():
        df = _fetch_yfinance("^TNX", period="3mo")
        return round(float(df["Close"].iloc[-1]), 4)

    def _fetch_dxy_change():
        dxy = _fetch_dxy_series("3mo")
        if dxy is not None and len(dxy) >= 6:
            start = float(dxy.iloc[-6])
            end = float(dxy.iloc[-1])
            if start != 0:
                return round((end - start) / start * 100, 4)
        return None

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_vix = ex.submit(_fetch_vix)
        f_tnx = ex.submit(_fetch_tnx)
        f_dxy = ex.submit(_fetch_dxy_change)
        try:
            result["vix_level"] = f_vix.result()
        except Exception as e:
            logger.warning("VIX current mislukt: %s", e)
        try:
            result["tnx_level"] = f_tnx.result()
        except Exception as e:
            logger.warning("TNX current mislukt: %s", e)
        try:
            result["dxy_change_5d"] = f_dxy.result()
        except Exception as e:
            logger.warning("DXY current mislukt: %s", e)

    _macro_current_cache["data"] = result
    _macro_current_cache["ts"] = now
    return result


def get_sector_momentum_20d(ticker: str) -> float | None:
    """20-daags momentum (%) van de sector-ETF die bij `ticker` hoort.
    Onbekende tickers vallen terug op SPY. Geeft None bij fout.
    """
    try:
        etf = SECTOR_ETF_MAP.get((ticker or "").upper(), "SPY")
        df = _get_sector_etf_data(etf)
        if df is None or len(df) < 21:
            return None
        closes = df["Close"]
        start = float(closes.iloc[-21])
        end = float(closes.iloc[-1])
        if start == 0:
            return None
        return round((end - start) / start * 100, 2)
    except Exception:
        return None


# ── Earnings-kalender (voor earnings-nabijheid feature) ──────────────────────
# Cache per ticker: ticker -> sorted list[date] OF None (te weinig data / fout).
_earnings_cache: dict[str, list | None] = {}
_EARNINGS_MIN_DATES = 15          # minder dan dit => ticker krijgt fallback
# Gaten-bestendige fallback-waarden (nooit NaN naar het model).
EARNINGS_FALLBACK = {
    "days_since_last_earnings": 45.0,
    "days_until_next_earnings": 45.0,
    "earnings_zone": 0.0,
}


def get_earnings_calendar(ticker: str) -> list | None:
    """Haalt en cachet de earnings-kalender (gesorteerde lijst van datums) van
    een ticker via yfinance. Returns None als er < 15 datums zijn of bij fout —
    de aanroeper gebruikt dan de fallback-waarden.
    """
    if ticker in _earnings_cache:
        return _earnings_cache[ticker]

    result: list | None = None
    for attempt in range(2):
        try:
            df = yf.Ticker(ticker).get_earnings_dates(limit=50)
            if df is not None and len(df) > 0:
                # Non-lege respons: accepteer. >=15 datums -> echte kalender,
                # <15 -> result blijft None zodat de aanroeper fallback gebruikt.
                if len(df) >= _EARNINGS_MIN_DATES:
                    idx = pd.to_datetime(df.index)
                    if getattr(idx, "tz", None) is not None:
                        idx = idx.tz_convert("UTC").tz_localize(None)
                    result = sorted({ts.date() for ts in idx})
                break
            # Leeg: mogelijk throttle -> nog eens proberen.
        except Exception as e:
            logger.warning("Earnings-kalender ophalen mislukt voor %s (poging %d): %s",
                           ticker, attempt + 1, e)
        if attempt < 1:
            time.sleep(2)

    _earnings_cache[ticker] = result
    return result


def compute_earnings_features(ticker: str, as_of) -> dict:
    """Drie gaten-bestendige earnings-nabijheid features t.o.v. datum `as_of`.

    - days_since_last_earnings: dagen sinds de vorige earnings-datum
    - days_until_next_earnings: dagen tot de volgende earnings-datum
    - earnings_zone: 1 als days_since < 5 OF days_until < 10, anders 0

    Geeft ALTIJD geldige waarden terug (nooit NaN). Bij een ticker met te weinig
    earnings-data, of als er geen vorige/volgende datum beschikbaar is voor deze
    rij, worden fallback-waarden (45 / 45 / 0) gebruikt.
    """
    cal = get_earnings_calendar(ticker)
    if not cal:                      # None of lege lijst => ticker-fallback
        return dict(EARNINGS_FALLBACK)

    try:
        d = pd.Timestamp(as_of).date()
    except Exception:
        return dict(EARNINGS_FALLBACK)

    past   = [e for e in cal if e <= d]
    future = [e for e in cal if e > d]

    days_since = float((d - past[-1]).days) if past else 45.0
    days_until = float((future[0] - d).days) if future else 45.0
    zone = 1.0 if (days_since < 5 or days_until < 10) else 0.0

    return {
        "days_since_last_earnings": days_since,
        "days_until_next_earnings": days_until,
        "earnings_zone": zone,
    }


def _fetch_yfinance(ticker: str, period: str = "2y") -> pd.DataFrame:
    # Retry-met-backoff: een lege respons is vaak een tijdelijke Yahoo-throttle,
    # niet een echt delisted symbool. Opnieuw proberen voorkomt dat gedeelde
    # caches (SPY, sector-ETF) leeg blijven en daardoor elke dataset-rij opnieuw
    # gaat fetchen — de oorzaak van de vorige vastloper. De succes-/live-pad
    # blijft snel: bij data in één keer keren we direct terug, geen sleep.
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                return df
        except Exception as e:
            last_err = e
        if attempt < 2:
            time.sleep(2 * (attempt + 1))  # 2s, dan 4s
    if last_err is not None:
        raise last_err
    raise ValueError(f"yfinance returned no data for {ticker}")


def _fetch_stooq(ticker: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}&i=d"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), index_col="Date")
    df.index = pd.to_datetime(df.index)
    df.columns = [c.capitalize() for c in df.columns]
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"Stooq returned no usable data for {ticker}")
    cutoff = pd.Timestamp.now() - pd.DateOffset(months=6)
    return df[df.index >= cutoff].sort_index()


def fetch_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        return _fetch_yfinance(ticker, period)
    except Exception as e:
        logger.warning("yfinance failed for %s: %s — trying Stooq fallback", ticker, e)
        return _fetch_stooq(ticker)


_WARMUP_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLC"]


def warmup_caches() -> None:
    """Vul veelgebruikte caches eenmalig bij serverstart zodat de eerste
    analyse niet hoeft te wachten op koude downloads.
    """
    def _warmup_etf(etf: str) -> None:
        try:
            _get_sector_etf_data(etf)
            logger.info("Warmup: sector-ETF %s geladen", etf)
        except Exception as e:
            logger.warning("Warmup: sector-ETF %s mislukt: %s", etf, e)

    try:
        get_macro_current()
        logger.info("Warmup: macro_current geladen")
    except Exception as e:
        logger.warning("Warmup: macro_current mislukt: %s", e)

    try:
        get_spy_data()
        logger.info("Warmup: SPY geladen")
    except Exception as e:
        logger.warning("Warmup: SPY mislukt: %s", e)

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_warmup_etf, etf): etf for etf in _WARMUP_ETFS}
        for f in futures:
            try:
                f.result()
            except Exception:
                pass

    logger.info("Warmup klaar")


def fetch_ticker_info(ticker: str) -> dict:
    now = time.time()
    cached = _info_cache.get(ticker)
    if cached and (now - cached["ts"]) < INFO_CACHE_TTL:
        return cached["data"]

    result = {"name": ticker, "sector": "", "exchange": "", "currency": "USD",
              "pre_market_price": None, "earnings_date": None}
    try:
        t = yf.Ticker(ticker)
        fast = t.fast_info
        try:
            result["currency"] = fast.currency or "USD"
        except Exception:
            pass
        try:
            result["pre_market_price"] = round(float(fast.pre_market_price), 4) if fast.pre_market_price else None
        except Exception:
            pass
        try:
            result["name"] = getattr(fast, "display_name", None) or ticker
        except Exception:
            pass
    except Exception as e:
        logger.warning("fast_info mislukt voor %s: %s", ticker, e)

    _info_cache[ticker] = {"data": result, "ts": time.time()}
    return result


def get_earnings_date(ticker: str):
    """Haalt de eerstvolgende earnings datum op via yfinance.
    Returns: (datum_str, dagen_tot_earnings) of (None, None)
    """
    try:
        from datetime import date as _date
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        if calendar is None or (hasattr(calendar, "empty") and calendar.empty):
            return None, None
        if "Earnings Date" in calendar.index:
            earnings_date = calendar.loc["Earnings Date"].iloc[0]
            if hasattr(earnings_date, "date"):
                earnings_date = earnings_date.date()
            today = _date.today()
            dagen = (earnings_date - today).days
            if 0 <= dagen <= 14:
                return str(earnings_date), dagen
        return None, None
    except Exception as e:
        print(f"Earnings check fout voor {ticker}: {e}")
        return None, None


def fetch_quick_prices(tickers: list) -> dict:
    """Haalt snel huidige prijs en dagverandering op voor een lijst tickers."""
    result = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="2d", auto_adjust=True, progress=False)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            if len(df) >= 2:
                prev = float(df["Close"].iloc[-2])
                curr = float(df["Close"].iloc[-1])
                change_pct = round((curr - prev) / prev * 100, 2)
                result[ticker] = {"price": round(curr, 2), "change_pct": change_pct}
            elif len(df) == 1:
                curr = float(df["Close"].iloc[-1])
                result[ticker] = {"price": round(curr, 2), "change_pct": None}
            else:
                result[ticker] = None
        except Exception as e:
            logger.warning("Could not fetch price for %s: %s", ticker, e)
            result[ticker] = None
    return result


def fetch_analyze_extras(ticker: str) -> tuple:
    """Haalt ticker info en marktcontext parallel op."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_info   = executor.submit(fetch_ticker_info, ticker)
        f_market = executor.submit(fetch_market_context)
        info = f_info.result()
        try:
            market = f_market.result()
        except Exception:
            market = {}
    return info, market


def fetch_market_context() -> dict:
    now = time.time()
    if _market_cache["data"] is not None and (now - _market_cache["ts"]) < MARKET_CACHE_TTL:
        return _market_cache["data"]

    symbols = {"sp500": "^GSPC", "nasdaq": "^IXIC", "aex": "^AEX", "dax": "^GDAXI", "dji": "^DJI"}

    def _fetch_one(name_sym):
        name, sym = name_sym
        try:
            df = yf.Ticker(sym).history(period="5d")
            if df.empty:
                return name, None
            if len(df) >= 2:
                prev = float(df["Close"].iloc[-2])
                curr = float(df["Close"].iloc[-1])
                return name, {"price": round(curr, 2), "change_pct": round((curr - prev) / prev * 100, 2)}
            elif len(df) == 1:
                curr = float(df["Close"].iloc[-1])
                return name, {"price": round(curr, 2), "change_pct": None}
            return name, None
        except Exception as e:
            logger.warning("Could not fetch %s: %s", sym, e)
            return name, None

    with ThreadPoolExecutor(max_workers=5) as ex:
        result = dict(ex.map(_fetch_one, symbols.items()))

    _market_cache["data"] = result
    _market_cache["ts"] = time.time()
    return result
