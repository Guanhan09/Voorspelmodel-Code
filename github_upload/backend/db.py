from __future__ import annotations

import os
import threading
from datetime import date, datetime

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sqlalchemy import Column, Date, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from evaluation import FORECAST_HORIZON, classify_direction

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "stocks.db")
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

_US_BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def _add_trading_days(d: date, n: int) -> date:
    """d + n handelsdagen vooruit (weekenden + Amerikaanse federale feestdagen overgeslagen)."""
    return (pd.Timestamp(d) + n * _US_BDAY).date()


class Base(DeclarativeBase):
    pass


class WatchlistItem(Base):
    __tablename__ = "watchlist"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), unique=True, nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    prediction_date = Column(Date, nullable=False)
    direction = Column(String(10))
    confidence = Column(Integer)
    target_price = Column(Float)
    actual_close = Column(Float, nullable=True)
    actual_direction = Column(String(10), nullable=True)
    inputs_json = Column(Text)
    horizon_days = Column(Integer, default=FORECAST_HORIZON)
    target_date = Column(Date, nullable=True)


def _migrate_schema():
    """Voegt kolommen toe aan een al bestaande predictions-tabel (create_all doet dit niet)."""
    with engine.connect() as conn:
        existing = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(predictions)")}
        if "horizon_days" not in existing:
            conn.exec_driver_sql(
                f"ALTER TABLE predictions ADD COLUMN horizon_days INTEGER DEFAULT {FORECAST_HORIZON}"
            )
        if "target_date" not in existing:
            conn.exec_driver_sql("ALTER TABLE predictions ADD COLUMN target_date DATE")
        conn.commit()


def init_db():
    Base.metadata.create_all(bind=engine)
    _migrate_schema()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def add_watchlist(db: Session, ticker: str):
    existing = db.query(WatchlistItem).filter(WatchlistItem.ticker == ticker.upper()).first()
    if not existing:
        db.add(WatchlistItem(ticker=ticker.upper()))
        db.commit()


def remove_watchlist(db: Session, ticker: str):
    item = db.query(WatchlistItem).filter(WatchlistItem.ticker == ticker.upper()).first()
    if item:
        db.delete(item)
        db.commit()


def get_watchlist(db: Session) -> list[str]:
    return [w.ticker for w in db.query(WatchlistItem).order_by(WatchlistItem.added_at).all()]


_save_lock = threading.Lock()


def get_latest_prediction(db: Session, ticker: str) -> Prediction | None:
    return (
        db.query(Prediction)
        .filter(Prediction.ticker == ticker.upper())
        .order_by(Prediction.created_at.desc())
        .first()
    )


def save_prediction(db: Session, ticker: str, prediction: dict, indicators: dict) -> tuple[Prediction, bool]:
    import json

    with _save_lock:
        existing_today = (
            db.query(Prediction)
            .filter(Prediction.ticker == ticker.upper())
            .filter(Prediction.prediction_date == date.today())
            .first()
        )

        if existing_today is not None:
            return existing_today, False

        p = Prediction(
            ticker=ticker.upper(),
            prediction_date=date.today(),
            direction=prediction["direction"],
            confidence=prediction["confidence"],
            target_price=prediction["target_price"],
            inputs_json=json.dumps(indicators),
            horizon_days=FORECAST_HORIZON,
            target_date=_add_trading_days(date.today(), FORECAST_HORIZON),
        )
        db.add(p)
        db.commit()
        db.refresh(p)
        return p, True


def fill_actuals(db: Session):
    """Fill in actual outcomes for predictions whose horizon (target_date) has passed."""
    from data import fetch_ohlcv
    import json

    pending = (
        db.query(Prediction)
        .filter(Prediction.actual_close.is_(None))
        .filter(Prediction.target_date.isnot(None))
        .filter(Prediction.target_date <= date.today())
        .all()
    )
    tickers = list({p.ticker for p in pending})
    for ticker in tickers:
        try:
            df = fetch_ohlcv(ticker, period="3mo")
        except Exception:
            continue
        for pred in [p for p in pending if p.ticker == ticker]:
            pred_date = pred.prediction_date
            # find the first trading day on/after the horizon's target_date
            later = df[df.index.date >= pred.target_date]
            if later.empty:
                continue
            row = later.iloc[0]
            actual = round(float(row["Close"]), 4)
            pred.actual_close = actual
            prev_rows = df[df.index.date <= pred_date]
            if not prev_rows.empty:
                prev_close = float(prev_rows.iloc[-1]["Close"])
                change_pct = (actual - prev_close) / prev_close * 100
                pred.actual_direction = classify_direction(change_pct)
    db.commit()


def get_accuracy(db: Session, days: int = 30) -> dict:
    from datetime import timedelta
    cutoff = date.today() - timedelta(days=days)
    preds = (
        db.query(Prediction)
        .filter(Prediction.prediction_date >= cutoff)
        .filter(Prediction.actual_direction.isnot(None))
        .all()
    )
    total = len(preds)
    if total == 0:
        return {"total": 0, "correct": 0, "hit_rate": None, "avg_price_error_pct": None}
    correct = sum(1 for p in preds if p.direction == p.actual_direction)
    price_errors = []
    for p in preds:
        if p.actual_close and p.target_price:
            err = abs(p.actual_close - p.target_price) / p.actual_close * 100
            price_errors.append(err)
    return {
        "total": total,
        "correct": correct,
        "hit_rate": round(correct / total * 100, 1),
        "avg_price_error_pct": round(sum(price_errors) / len(price_errors), 2) if price_errors else None,
    }
