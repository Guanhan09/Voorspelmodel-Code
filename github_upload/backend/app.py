from __future__ import annotations

import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

import data as data_module
import db as db_module
import indicators as ind_module
import news as news_module
import patterns as pat_module
import predictor as pred_module

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

SCAN_TICKERS = [
    # AEX 25
    "ASML.AS", "SHELL.AS", "UNA.AS", "INGA.AS", "ADYEN.AS", "PHIA.AS", "AD.AS", "MT.AS",
    "REN.AS", "NN.AS", "ABN.AS", "ASRNL.AS", "KPN.AS", "AKZA.AS", "HEIA.AS", "DSFIR.AS",
    "PRX.AS", "RAND.AS", "IMCD.AS", "BESI.AS", "WKL.AS", "EXO.AS", "ASM.AS", "GLPG.AS", "AGN.AS",
    # S&P 200
    "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "BRK-B",
    "JPM", "WMT", "LLY", "V", "ORCL", "MA", "NFLX", "XOM", "COST", "JNJ",
    "HD", "PG", "ABBV", "BAC", "CVX", "KO", "PLTR", "AMD", "GE", "TMUS",
    "CSCO", "CRM", "WFC", "PM", "ABT", "IBM", "MS", "MCD", "LIN", "AXP",
    "ISRG", "GS", "TMO", "NOW", "T", "MRK", "ACN", "INTU", "RTX", "PEP",
    "DIS", "VZ", "BX", "CAT", "QCOM", "BKNG", "TXN", "ADBE", "UBER", "BA",
    "SCHW", "BLK", "AMGN", "NEE", "C", "SPGI", "GILD", "BSX", "DHR", "HON",
    "PFE", "SYK", "AMAT", "LOW", "COF", "PGR", "ANET", "CMCSA", "UNH", "LMT",
    "ETN", "DE", "ADP", "MU", "PANW", "KKR", "FI", "TJX", "MDT", "BMY",
    "ADI", "COP", "MO", "VRTX", "GEV", "NKE", "INTC", "SBUX", "PLD", "LRCX", "CB",
    "MMC", "ICE", "SO", "REGN", "ELV", "CI", "EQIX", "KLAC", "USB", "FCX",
    "MAR", "CME", "PNC", "MCO", "ZTS", "BDX", "DUK", "TGT", "EOG", "SLB",
    "AON", "CL", "NSC", "ITW", "TFC", "APD", "MMM", "ECL", "EMR", "PYPL",
    "FDX", "GD", "AJG", "MRVL", "NOC", "CSX", "ROP", "MNST", "MCK", "WM",
    "ORLY", "MSI", "FTNT", "RSG", "HCA", "AZO", "CARR", "ADSK", "PSA", "COIN",
    "AEP", "AFL", "BK", "PCAR", "OXY", "KMB", "SRE", "KMI", "TRV", "TT",
    "DLR", "PAYX", "CCI", "NXPI", "SPG", "GM", "MPC", "ROST", "CHTR", "HUM",
    "F", "KDP", "EW", "PSX", "AIG", "AMP", "GIS", "WMB", "JCI", "EXC",
    "VLO", "ALL", "TEL", "KVUE", "CMG", "MET", "FAST", "A", "OKE", "COR",
    "STZ", "IDXX", "PRU", "CTAS", "IQV", "DOW", "LHX", "GWW", "CTSH", "KHC",
    "HSY", "KR", "HLT", "BIIB",
]

CACHE_TTL_SECONDS = 7200  # 2 uur

# In-memory cache
laatste_scan_resultaat: Optional[dict] = None
laatste_scan_tijd: Optional[datetime] = None

# Alle scan resultaten per ticker (voor sector panel)
_scan_alle_resultaten: dict = {}

# Per-ticker cache voor scanner (1 uur TTL)
_ticker_cache: dict = {}
TICKER_CACHE_TTL_SECONDS = 3600

# Per-ticker cache voor analyze endpoint (30 minuten TTL)
_analyze_cache: dict = {}
ANALYZE_CACHE_TTL_SECONDS = 1800

# Progress tracking
_scan_status = {"status": "idle", "progress": 0, "current_ticker": "", "current": 0, "total": 0, "ticker": ""}


def _is_yahoo_auth_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(x in msg for x in ["401", "403", "invalid crumb", "unauthorized"])


def _scan_single_ticker(ticker: str) -> Optional[dict]:
    cached = _ticker_cache.get(ticker)
    if cached:
        ts, result = cached
        if (datetime.now() - ts).total_seconds() < TICKER_CACHE_TTL_SECONDS:
            return result

    for attempt in range(2):
        try:
            df = data_module.fetch_ohlcv(ticker)
            if len(df) < 30:
                return None
            info = data_module.fetch_ticker_info(ticker)
            inds = ind_module.compute_all(df)
            pred = pred_module.predict(inds, df)
            current = inds["current_price"]
            target = pred["target_price"]
            expected_move_pct = round(abs(target - current) / current * 100, 2) if current else 0
            expected_change_pct = round((target - current) / current * 100, 2) if current else 0
            result = {
                "ticker": ticker,
                "name": info.get("name", ticker),
                "price": current,
                "currency": info.get("currency", "USD"),
                "direction": pred["direction"],
                "confidence": pred["confidence"],
                "expected_move_pct": expected_move_pct,
                "expected_change_pct": expected_change_pct,
                "target_price": pred["target_price"],
                "price_range_low": pred["price_range_low"],
                "price_range_high": pred["price_range_high"],
            }
            _ticker_cache[ticker] = (datetime.now(), result)
            return result
        except Exception as e:
            if attempt == 0 and _is_yahoo_auth_error(e):
                logger.warning("Yahoo auth error voor %s (poging 1): %s — retry na 2s", ticker, e)
                time.sleep(2)
                continue
            logger.warning("Scan mislukt voor %s: %s", ticker, e)
            return None
    return None


def _run_full_scan() -> list:
    global _scan_status
    total = len(SCAN_TICKERS)
    _scan_status = {"status": "scanning", "progress": 0, "current_ticker": "", "current": 0, "total": total, "ticker": ""}

    results = []
    failed = 0
    done_count = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_scan_single_ticker, t): t for t in SCAN_TICKERS}
        for future in as_completed(futures):
            t = futures[future]
            _scan_status["current_ticker"] = t
            _scan_status["ticker"] = t
            result = future.result()
            if result:
                results.append(result)
            else:
                failed += 1
            done_count += 1
            _scan_status["current"] = done_count
            _scan_status["progress"] = round(done_count / total * 100)

    global _scan_alle_resultaten
    _scan_alle_resultaten = {r["ticker"]: r for r in results}

    ups   = [r for r in results if r["direction"] == "Up"   and r["confidence"] >= 52]
    downs = [r for r in results if r["direction"] == "Down" and r["confidence"] >= 52]

    ups.sort(key=lambda x: x["confidence"], reverse=True)
    downs.sort(key=lambda x: x["confidence"], reverse=True)

    stijgers = ups[:10]
    dalers   = downs[:10]

    up_count   = len([r for r in results if r["direction"] == "Up"])
    down_count = len([r for r in results if r["direction"] == "Down"])
    flat_count = len([r for r in results if r["direction"] == "Flat"])
    logger.info("Scan resultaten: %d Up, %d Down, %d Flat van %d totaal", up_count, down_count, flat_count, len(results))
    logger.info("Na confidence-filter (>=52): %d stijgers, %d dalers getoond", len(stijgers), len(dalers))
    _scan_status = {"status": "done", "progress": 100, "current_ticker": "", "current": total, "total": total, "ticker": ""}
    return {"stijgers": stijgers, "dalers": dalers}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_module.init_db()

    try:
        from data import warmup_caches
        warmup_caches()
    except Exception as e:
        logger.warning("Cache warmup mislukt (server start gewoon): %s", e)

    from pathlib import Path
    print("=" * 60, flush=True)
    print("STARTUP CHECK", flush=True)
    print(f"Working dir: {os.getcwd()}", flush=True)
    print(f"ml_model.pkl bestaat:    {(Path(__file__).parent / 'ml_model.pkl').exists()}", flush=True)
    print(f"ml_features.json bestaat: {(Path(__file__).parent / 'ml_features.json').exists()}", flush=True)
    try:
        from ml_predictor import _load_models, _models, _features
        success = _load_models()
        if success:
            print(f"✓ ML ensemble geladen: {len(_models)} model(len) — {list(_models.keys())}", flush=True)
            print(f"✓ Aantal features: {len(_features) if _features else 0}", flush=True)
        else:
            print("✗ ML model niet beschikbaar — fallback actief", flush=True)
    except Exception as e:
        print(f"✗ FOUT bij laden ML model: {e}", flush=True)
        import traceback
        traceback.print_exc()
    print("=" * 60, flush=True)

    yield


app = FastAPI(title="Stock Predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", response_class=FileResponse)
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/api/scan")
def scan(force: bool = Query(default=False)):
    global laatste_scan_resultaat, laatste_scan_tijd

    cache_geldig = (
        laatste_scan_resultaat is not None
        and laatste_scan_tijd is not None
        and (datetime.now() - laatste_scan_tijd).total_seconds() < CACHE_TTL_SECONDS
    )

    if not force and cache_geldig:
        logger.info("Scan-cache gebruikt (leeftijd: %.0fs)", (datetime.now() - laatste_scan_tijd).total_seconds())
        return {
            "stijgers": laatste_scan_resultaat["stijgers"],
            "dalers": laatste_scan_resultaat["dalers"],
            "scan_timestamp": laatste_scan_tijd.strftime("%H:%M"),
            "from_cache": True,
        }

    scan_data = _run_full_scan()
    laatste_scan_resultaat = scan_data
    laatste_scan_tijd = datetime.now()

    return {
        "stijgers": scan_data["stijgers"],
        "dalers": scan_data["dalers"],
        "scan_timestamp": laatste_scan_tijd.strftime("%H:%M"),
        "from_cache": False,
    }


@app.get("/api/scan/status")
def scan_status():
    return _scan_status


@app.get("/api/news/{ticker}")
async def news_endpoint(ticker: str, company: str = ""):
    from backend.news import get_news
    items = get_news(ticker.upper(), company_name=company)
    return {"ticker": ticker.upper(), "news": items}


@app.get("/api/analyze/{ticker}")
def analyze(ticker: str, db: Session = Depends(db_module.get_db)):
    ticker = ticker.upper().strip()

    cached = _analyze_cache.get(ticker)
    if cached:
        ts, result = cached
        if (datetime.now() - ts).total_seconds() < ANALYZE_CACHE_TTL_SECONDS:
            return result

    try:
        df = data_module.fetch_ohlcv(ticker)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Kon geen data ophalen voor '{ticker}': {e}")

    if len(df) < 30:
        raise HTTPException(status_code=422, detail=f"Onvoldoende historische data voor '{ticker}'.")

    # Haal ticker info en marktcontext parallel op terwijl OHLCV al binnen is
    info, market = data_module.fetch_analyze_extras(ticker)

    inds = ind_module.compute_all(df)
    spy_df = data_module.get_spy_data()
    regime = ind_module.detect_market_regime(spy_df) if spy_df is not None else "neutral"
    prediction = pred_module.predict(inds, df, regime=regime, ticker=ticker)
    similar = pat_module.find_similar_days(df)

    earnings_date, dagen_tot_earnings = None, None
    earnings_warning = None

    try:
        _, was_saved = db_module.save_prediction(db, ticker, prediction, inds)
        if not was_saved:
            logger.info("Voorspelling identiek aan laatste, niet opgeslagen")
    except Exception as e:
        logger.warning("Could not save prediction: %s", e)

    chart_df = df
    dates = [str(d.date()) for d in chart_df.index]

    def _to_json_list(series):
        return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
                for v in series.tolist()]

    closes  = _to_json_list(chart_df["Close"].astype(float))
    opens   = _to_json_list(chart_df["Open"].astype(float))
    highs   = _to_json_list(chart_df["High"].astype(float))
    lows    = _to_json_list(chart_df["Low"].astype(float))
    volumes = _to_json_list(chart_df["Volume"].astype(float))
    ma20_series = _to_json_list(chart_df["Close"].rolling(20).mean().round(4))
    ma50_series = _to_json_list(chart_df["Close"].rolling(50).mean().round(4))

    response = {
        "ticker": ticker,
        "name": info.get("name", ticker),
        "sector": info.get("sector", ""),
        "currency": info.get("currency", "USD"),
        "pre_market_price": info.get("pre_market_price"),
        "earnings_warning": earnings_warning,
        "indicators": inds,
        "prediction": prediction,
        "market_regime": regime,
        "similar_days": similar,
        "market_context": market,
        "chart": {
            "dates":   dates,
            "opens":   opens,
            "highs":   highs,
            "lows":    lows,
            "closes":  closes,
            "volumes": volumes,
            "ma20":    ma20_series,
            "ma50":    ma50_series,
        },
    }
    _analyze_cache[ticker] = (datetime.now(), response)
    return response


SECTOR_TOP_TICKERS = {
    "tech":         ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "CRM", "ORCL", "QCOM", "TXN", "AVGO"],
    "financials":   ["JPM", "BAC", "GS", "MS", "WFC", "BLK", "AXP", "V", "MA", "C"],
    "energy":       ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO", "PSX", "OXY"],
    "healthcare":   ["JNJ", "UNH", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY", "AMGN"],
    "consdiscr":    ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "BKNG", "GM"],
    "consstaples":  ["WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "CL", "GIS", "KMB"],
    "industrials":  ["GE", "HON", "UPS", "CAT", "DE", "LMT", "RTX", "BA", "MMM", "FDX"],
    "utilities":    ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PCG", "ED", "XEL"],
    "materials":    ["LIN", "APD", "ECL", "DD", "NEM", "FCX", "NUE", "VMC", "MLM", "ALB"],
    "realestate":   ["AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "O", "SPG", "AVB", "EQR"],
    "comms":        ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "EA", "TTWO"],
}

TICKER_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMD": "AMD",
    "INTC": "Intel", "CRM": "Salesforce", "ORCL": "Oracle", "QCOM": "Qualcomm",
    "TXN": "Texas Instruments", "AVGO": "Broadcom",
    "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "WFC": "Wells Fargo", "BLK": "BlackRock",
    "AXP": "American Express", "V": "Visa", "MA": "Mastercard", "C": "Citigroup",
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "SLB": "SLB", "EOG": "EOG Resources", "PXD": "Pioneer Natural",
    "MPC": "Marathon Petroleum", "VLO": "Valero", "PSX": "Phillips 66", "OXY": "Occidental",
    "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth", "PFE": "Pfizer",
    "MRK": "Merck", "ABBV": "AbbVie", "TMO": "Thermo Fisher",
    "ABT": "Abbott", "DHR": "Danaher", "BMY": "Bristol-Myers", "AMGN": "Amgen",
    "AMZN": "Amazon", "TSLA": "Tesla", "HD": "Home Depot", "MCD": "McDonald's",
    "NKE": "Nike", "SBUX": "Starbucks", "TGT": "Target", "LOW": "Lowe's",
    "BKNG": "Booking Holdings", "GM": "General Motors",
    "WMT": "Walmart", "PG": "Procter & Gamble", "KO": "Coca-Cola",
    "PEP": "PepsiCo", "COST": "Costco", "PM": "Philip Morris",
    "MO": "Altria", "CL": "Colgate", "GIS": "General Mills", "KMB": "Kimberly-Clark",
    "GE": "GE Aerospace", "HON": "Honeywell", "UPS": "UPS", "CAT": "Caterpillar",
    "DE": "John Deere", "LMT": "Lockheed Martin", "RTX": "RTX",
    "BA": "Boeing", "MMM": "3M", "FDX": "FedEx",
    "NEE": "NextEra Energy", "DUK": "Duke Energy", "SO": "Southern Company",
    "D": "Dominion Energy", "AEP": "Am. Electric Power", "EXC": "Exelon",
    "SRE": "Sempra", "PCG": "PG&E", "ED": "Consolidated Edison", "XEL": "Xcel Energy",
    "LIN": "Linde", "APD": "Air Products", "ECL": "Ecolab", "DD": "DuPont",
    "NEM": "Newmont", "FCX": "Freeport-McMoRan", "NUE": "Nucor",
    "VMC": "Vulcan Materials", "MLM": "Martin Marietta", "ALB": "Albemarle",
    "AMT": "American Tower", "PLD": "Prologis", "CCI": "Crown Castle",
    "EQIX": "Equinix", "PSA": "Public Storage", "DLR": "Digital Realty",
    "O": "Realty Income", "SPG": "Simon Property", "AVB": "AvalonBay", "EQR": "Equity Residential",
    "GOOGL": "Alphabet", "META": "Meta", "NFLX": "Netflix", "DIS": "Disney",
    "CMCSA": "Comcast", "T": "AT&T", "VZ": "Verizon", "TMUS": "T-Mobile",
    "EA": "Electronic Arts", "TTWO": "Take-Two",
}

@app.get("/api/sector/{sector_naam}")
def sector_detail(sector_naam: str):
    import re

    key = re.sub(r'[^a-z0-9]', '', sector_naam.lower())
    tickers = SECTOR_TOP_TICKERS.get(key)
    if not tickers:
        raise HTTPException(status_code=404, detail=f"Sector '{sector_naam}' niet gevonden")

    sector_results = []
    for t in tickers:
        r = _scan_alle_resultaten.get(t)
        if r:
            sector_results.append({
                "ticker":     t,
                "name":       TICKER_NAMES.get(t, r.get("name", t)),
                "price":      r.get("price"),
                "direction":  r.get("direction"),
                "confidence": r.get("confidence"),
            })

    # Sorteer: Up-voorspellingen met hoogste confidence eerst
    sector_results.sort(
        key=lambda x: (x["direction"] == "Up", x["confidence"] or 0),
        reverse=True,
    )
    top3 = sector_results[:3]

    return {"sector": sector_naam, "top3": top3}


SECTOR_ETFS = [
    ("Tech", "XLK"),
    ("Financials", "XLF"),
    ("Energy", "XLE"),
    ("Healthcare", "XLV"),
    ("Cons. Discr.", "XLY"),
    ("Cons. Staples", "XLP"),
    ("Industrials", "XLI"),
    ("Utilities", "XLU"),
    ("Materials", "XLB"),
    ("Real Estate", "XLRE"),
    ("Comms", "XLC"),
]

SECTOR_CACHE_TTL = 1800  # 30 minuten
_sector_cache: Optional[dict] = None
_sector_cache_tijd: Optional[datetime] = None


@app.get("/api/sectors")
def sectors():
    global _sector_cache, _sector_cache_tijd
    cache_geldig = (
        _sector_cache is not None
        and _sector_cache_tijd is not None
        and (datetime.now() - _sector_cache_tijd).total_seconds() < SECTOR_CACHE_TTL
    )
    if cache_geldig:
        return _sector_cache

    import yfinance as yf
    result = []
    for sector_name, etf in SECTOR_ETFS:
        try:
            df = yf.download(etf, period="2d", auto_adjust=True, progress=False)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            if len(df) >= 2:
                prev = float(df["Close"].iloc[-2])
                curr = float(df["Close"].iloc[-1])
                change_pct = round((curr - prev) / prev * 100, 2)
            else:
                change_pct = None
            result.append({"sector": sector_name, "etf": etf, "change_pct": change_pct})
        except Exception as e:
            logger.warning("Sector fetch failed voor %s: %s", etf, e)
            result.append({"sector": sector_name, "etf": etf, "change_pct": None})

    _sector_cache = result
    _sector_cache_tijd = datetime.now()
    return result


@app.get("/api/market")
def market_context():
    try:
        return data_module.fetch_market_context()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prices")
def get_prices(tickers: str = Query(default="")):
    if not tickers:
        return {}
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()][:10]
    return data_module.fetch_quick_prices(ticker_list)


@app.get("/api/watchlist")
def get_watchlist(db: Session = Depends(db_module.get_db)):
    return {"tickers": db_module.get_watchlist(db)}


@app.post("/api/watchlist/{ticker}")
def add_to_watchlist(ticker: str, db: Session = Depends(db_module.get_db)):
    db_module.add_watchlist(db, ticker.upper())
    return {"status": "added", "ticker": ticker.upper()}


@app.delete("/api/watchlist/{ticker}")
def remove_from_watchlist(ticker: str, db: Session = Depends(db_module.get_db)):
    db_module.remove_watchlist(db, ticker.upper())
    return {"status": "removed", "ticker": ticker.upper()}


@app.get("/api/accuracy")
def accuracy(db: Session = Depends(db_module.get_db)):
    try:
        db_module.fill_actuals(db)
    except Exception as e:
        logger.warning("fill_actuals failed: %s", e)
    return {
        "last_30d": db_module.get_accuracy(db, 30),
        "last_90d": db_module.get_accuracy(db, 90),
        "last_365d": db_module.get_accuracy(db, 365),
    }


@app.get("/api/backtest")
def backtest_endpoint():
    import json as _json
    results_path = os.path.join(os.path.dirname(__file__), "backtest_results.json")
    if not os.path.exists(results_path):
        raise HTTPException(
            status_code=404,
            detail="Nog geen backtest resultaten. Voer eerst 'python backend/backtest.py' uit.",
        )
    with open(results_path, "r", encoding="utf-8") as f:
        return _json.load(f)


@app.get("/api/history")
def all_prediction_history(db: Session = Depends(db_module.get_db)):
    from db import Prediction
    import json as _json
    rows = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(100)
        .all()
    )
    result = []
    for r in rows:
        entry_price = None
        if r.inputs_json:
            try:
                entry_price = _json.loads(r.inputs_json).get("current_price")
            except Exception:
                pass
        result.append({
            "id": r.id,
            "ticker": r.ticker,
            "date": str(r.prediction_date),
            "direction": r.direction,
            "confidence": r.confidence,
            "target_price": r.target_price,
            "entry_price": entry_price,
            "actual_close": r.actual_close,
            "actual_direction": r.actual_direction,
        })
    return result


@app.get("/api/history/{ticker}")
def prediction_history(ticker: str, db: Session = Depends(db_module.get_db)):
    from db import Prediction
    rows = (
        db.query(Prediction)
        .filter(Prediction.ticker == ticker.upper())
        .order_by(Prediction.created_at.desc())
        .limit(50)
        .all()
    )
    import json as _json
    result = []
    for r in rows:
        entry_price = None
        if r.inputs_json:
            try:
                entry_price = _json.loads(r.inputs_json).get("current_price")
            except Exception:
                pass
        result.append({
            "id": r.id,
            "date": str(r.prediction_date),
            "direction": r.direction,
            "confidence": r.confidence,
            "target_price": r.target_price,
            "entry_price": entry_price,
            "actual_close": r.actual_close,
            "actual_direction": r.actual_direction,
        })
    return result


