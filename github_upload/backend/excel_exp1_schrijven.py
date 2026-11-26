"""
Experiment 1, stap B: pws_exp1_voorspellingen.npz (uit excel_exp1_voorspellingen.py)
omzetten naar experiment1_resultaten.xlsx.

Tabblad 'Samenvatting': overall accuracy en top-20%-accuracy per model/ensemble.
Per model een tabblad met elke voorspelling (aandeel, datum, voorspeld en werkelijk label).
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "pws_exp1_voorspellingen.npz")
OUT = os.path.join(HERE, "..", "experiment1_resultaten.xlsx")
LABEL = np.array(["Down", "Flat", "Up"])  # 0/1/2
NL_NAAM = {"LogisticRegression": "Logistic Regression", "RandomForest": "Random Forest",
           "GradientBoosting": "Gradient Boosting", "LightGBM": "LightGBM",
           "Ensemble_A (RF+GB)": "Ensemble A", "Ensemble_B (alle)": "Ensemble B",
           "Ensemble_C (top-3)": "Ensemble C"}
UITLEG = {"Ensemble A": "gemiddelde kansen Random Forest + Gradient Boosting",
          "Ensemble B": "gemiddelde kansen alle 4 modellen",
          "Ensemble C": "gemiddelde kansen top-3 op top-20%-accuracy: "}

z = np.load(NPZ, allow_pickle=False)
names = list(z["model_names"])
probas, y = z["probas"], z["y_test"].astype(int)
tick, dat = z["test_tickers"], z["test_dates"]
top3 = list(z["top3"])
skipped = list(z["skipped"])
UITLEG["Ensemble C"] += ", ".join(NL_NAAM[n] for n in top3)
n = len(y)

rows, flags = [], {}
for name, p in zip(names, probas):
    pred, conf = p.argmax(1), p.max(1)
    thr = np.quantile(conf, 0.80)
    hc = conf >= thr
    flags[name] = hc
    rows.append({
        "Model/combinatie": NL_NAAM[name],
        "Aantal voorspellingen": n,
        "Overall accuracy (%)": round((pred == y).mean() * 100, 2),
        "Top-20%-accuracy (%)": round((pred[hc] == y[hc]).mean() * 100, 2),
        "Aantal in top-20%": int(hc.sum()),
        "Drempel zekerheid top-20% (%)": round(thr * 100, 2),
        "Toelichting": UITLEG.get(NL_NAAM[name], ""),
    })
counts = np.bincount(y, minlength=3)
rows.append({
    "Model/combinatie": "Baseline: altijd meest voorkomende klasse",
    "Aantal voorspellingen": n,
    "Overall accuracy (%)": round(counts.max() / n * 100, 2),
    "Top-20%-accuracy (%)": None, "Aantal in top-20%": None, "Drempel zekerheid top-20% (%)": None,
    "Toelichting": f"werkelijke verdeling testset: Down {counts[0]}, Flat {counts[1]}, Up {counts[2]}",
})
samenvatting = pd.DataFrame(rows)
print(samenvatting.to_string(index=False))

wb = Workbook(write_only=True)
bold = Font(bold=True)


def _b(ws, v):
    c = WriteOnlyCell(ws, value=v); c.font = bold; return c


ws = wb.create_sheet("Samenvatting")
for col, w in zip("ABCDEFG", [44, 12, 12, 12, 12, 14, 70]):
    ws.column_dimensions[col].width = w
ws.append([_b(ws, c) for c in samenvatting.columns])
for r in samenvatting.itertuples(index=False):
    out = []
    for v in r:
        if isinstance(v, float) and np.isnan(v):
            v = None
        c = WriteOnlyCell(ws, value=v)
        if isinstance(v, float):
            c.number_format = "0.00"
        out.append(c)
    ws.append(out)
ws.append([])
for t in [
    "Training: koersdata 01-01-2015 t/m 31-12-2024 (splitsing op datum). Test: 2025.",
    f"Testperiode in deze resultaten: {dat.min()} t/m {dat.max()}, {len(set(tick))} S&P 500-aandelen.",
    "Accuracy = aandeel voorspellingen waarvan het voorspelde label gelijk is aan het werkelijke label.",
    "Top-20%-accuracy = idem, alleen over de 20% voorspellingen met de hoogste zekerheid van dát model.",
    f"Overgeslagen aandelen ({len(skipped)}): {'; '.join(skipped) if skipped else 'geen'}",
]:
    ws.append([t])

datums = [datetime.strptime(d, "%Y-%m-%d") for d in dat]
true_lbl = LABEL[y]
for name, p in zip(names, probas):
    mn = NL_NAAM[name]
    ws = wb.create_sheet(mn)
    for col, w in zip("ABCDEFG", [10, 12, 16, 14, 16, 15, 12]):
        ws.column_dimensions[col].width = w
    ws.append([_b(ws, c) for c in ["Aandeel", "Datum", "Voorspeld label", "Zekerheid (%)",
                                   "Werkelijk label", "Goed voorspeld", "In top-20%"]])
    pred = p.argmax(1)
    pl = LABEL[pred]
    conf = np.round(p.max(1) * 100, 2)
    goed = (pred == y).astype(int)
    hc = flags[name].astype(int)
    for i in range(n):
        dc = WriteOnlyCell(ws, value=datums[i]); dc.number_format = "dd-mm-yyyy"
        zc = WriteOnlyCell(ws, value=float(conf[i])); zc.number_format = "0.00"
        ws.append([tick[i], dc, pl[i], zc, true_lbl[i], int(goed[i]), int(hc[i])])
    print(f"  {mn}: {n} regels", flush=True)

wb.save(OUT)
print(f"Opgeslagen: {os.path.abspath(OUT)}")
