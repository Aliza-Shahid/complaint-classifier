"""Inspect one CFPB narratives-archive zip: columns, narrative counts,
product mix, narrative length. Reads in chunks (low memory)."""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

path = Path(sys.argv[1])
print("File:", path.name, f"({path.stat().st_size / 1e6:,.0f} MB)")

with zipfile.ZipFile(path) as z:
    print("Inside zip:", [(i.filename, round(i.file_size / 1e6)) for i in z.infolist()], "(MB uncompressed)")

cols = list(pd.read_csv(path, compression="zip", nrows=0).columns)
print("Columns:", cols)

narr = next((c for c in cols if "narrative" in c.lower()), None)
prod = "Product" if "Product" in cols else None
date = "Date received" if "Date received" in cols else None
if not (narr and prod and date):
    raise SystemExit(f"Could not find narrative/product/date columns: {narr}, {prod}, {date}")
print("Using:", narr, "|", prod, "|", date)

total = with_text = 0
prod_counts = pd.Series(dtype="int64")
lengths = []
dmin, dmax = None, None
for chunk in pd.read_csv(path, compression="zip", usecols=[narr, prod, date],
                         dtype=str, chunksize=200_000):
    total += len(chunk)
    chunk[narr] = chunk[narr].fillna("").str.strip()
    has = chunk[chunk[narr] != ""]
    with_text += len(has)
    prod_counts = prod_counts.add(has[prod].value_counts(), fill_value=0)
    lengths.append(has[narr].str.split().str.len().to_numpy())
    d = chunk[date]
    dmin = d.min() if dmin is None else min(dmin, d.min())
    dmax = d.max() if dmax is None else max(dmax, d.max())

lengths = np.concatenate(lengths) if lengths else np.array([0])
print(f"\nTotal rows: {total:,}")
print(f"Rows with a narrative: {with_text:,} ({with_text / max(total, 1):.1%})")
print(f"Date range: {dmin} to {dmax}")
print("\nNarrative length (words): "
      f"median {np.median(lengths):.0f}, 90th pct {np.percentile(lengths, 90):.0f}, max {lengths.max()}")
print("\nProducts among narrative rows:")
print(prod_counts.sort_values(ascending=False).astype(int).to_string())