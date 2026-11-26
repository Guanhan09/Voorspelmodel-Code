"""Gedeelde constanten/logica voor het evalueren van voorspellingen.

Één bron van waarheid voor de horizon en de Up/Down/Flat-drempel, gebruikt
door zowel backtest.py (offline) als db.py (live evaluatie), zodat beide
dezelfde maatstaf hanteren als waarop het ML-model getraind is.
"""
from __future__ import annotations

DIRECTION_THRESHOLD = 3.0  # 3% drempel over FORECAST_HORIZON handelsdagen
FORECAST_HORIZON = 5       # Aantal handelsdagen vooruit


def classify_direction(change_pct: float) -> str:
    if change_pct > DIRECTION_THRESHOLD:
        return "Up"
    if change_pct < -DIRECTION_THRESHOLD:
        return "Down"
    return "Flat"
