"""
Experiment 2, stap B: pws_exp2_transacties.pkl (uit excel_exp2_transacties.py)
omzetten naar experiment2_resultaten.xlsx.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from excel_exp2_transacties import CACHE, OUT_XLSX, START_BEDRAG, STRAT_NL, STRAT_VOLGORDE, summary

with open(CACHE, "rb") as f:
    c = pickle.load(f)
trades, detail, skipped, top3 = c["trades"], c["detail"], c["skipped"], c["top3"]
volg = {s: i for i, s in enumerate(STRAT_VOLGORDE)}

t = trades.assign(_s=trades["strategie"].map(volg)).sort_values(["_s", "ticker", "nr"])
tab_tr = pd.DataFrame({
    "Strategie": t["strategie"].map(STRAT_NL),
    "Aandeel": t["ticker"],
    "Trade nr.": t["nr"],
    "Koopdatum": pd.to_datetime(t["entry_date"]).dt.tz_localize(None).dt.date,
    "Koopkoers": t["entry_price"].round(2),
    "Verkoopdatum": pd.to_datetime(t["exit_date"]).dt.tz_localize(None).dt.date,
    "Verkoopkoers": t["exit_price"].round(2),
    "Rendement transactie (%)": t["return_pct"].round(2),
    "Waarde na transactie (€)": t["waarde_na"].round(2),
    "Gedwongen verkocht op laatste handelsdag": np.where(t["forced_close"], "ja", "nee"),
})

d = detail.assign(_s=detail["strategie"].map(volg)).sort_values(["_s", "ticker"])
tab_pa = pd.DataFrame({
    "Strategie": d["strategie"].map(STRAT_NL),
    "Aandeel": d["ticker"],
    "Aantal trades": d["n_trades"],
    "Winnend": d["wins"], "Verliezend": d["losses"],
    "Eindwaarde (€)": d["eindwaarde"].round(2),
    "Totaalrendement (%)": d["total_return_pct"].round(2),
})

s = summary(detail).round(2)

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl", date_format="DD-MM-YYYY") as xw:
    s.to_excel(xw, sheet_name="Samenvatting", index=False)
    tab_pa.to_excel(xw, sheet_name="Per aandeel", index=False)
    tab_tr.to_excel(xw, sheet_name="Transacties", index=False)
    ws = xw.book["Samenvatting"]
    r0 = ws.max_row + 2
    for i, txt in enumerate([
        f"Startkapitaal: €{START_BEDRAG:,.0f} fictief per aandeel per strategie; testperiode 2025.".replace(",", "."),
        "Voorspelmodel = Ensemble C (gemiddelde kansen van de top-3 modellen uit experiment 1: "
        + ", ".join(top3) + "), getraind op koersdata 01-01-2015 t/m 31-12-2024.",
        "Long-only, geen transactiekosten; een open positie wordt op de laatste handelsdag van 2025 verkocht.",
        "Waarde na transactie = 10.000 × product van (1 + rendement/100) over alle trades t/m die regel.",
        "Koersen gecorrigeerd voor dividend en splitsingen (yfinance, auto_adjust=True).",
        f"Overgeslagen aandelen ({len(skipped)}): " + ("; ".join(f"{a}: {b}" for a, b in skipped) or "geen"),
    ]):
        ws.cell(row=r0 + i, column=1, value=txt)
    for ws in xw.book.worksheets:
        ws.freeze_panes = "A2"
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = max(12, min(45, len(str(col[0].value)) + 2))
            for cell in col[1:]:
                if isinstance(cell.value, float):
                    cell.number_format = "0.00"
                elif hasattr(cell.value, "year"):
                    cell.number_format = "DD-MM-YYYY"

print(f"Opgeslagen: {OUT_XLSX}")
print(f"Transacties: {len(tab_tr)} | Per aandeel: {len(tab_pa)}")
print(s.to_string(index=False))
