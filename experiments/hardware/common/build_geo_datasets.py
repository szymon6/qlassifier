"""Build the BT-5 candidate tables (label-free 4,000-row samples with leakage-safe groups).

Up: ../README.md (protocol §2.2) · Data: ../data/README.md
Run with ``python -I`` from the repo root; inputs are the groundwork's cleaned CSVs (copied into ../data/raw/ with
their SHA-256 recorded), outputs ../data/beijing_pm25_4k.csv and ../data/solar_boulder_4k.csv.

Rules (declared before any model run):
- beijing_pm25: UCI 501 Dongsi station, hourly 2013-03-01 .. 2017-02-28; rows with any missing value dropped;
  wind direction = 16-point compass -> degrees (N = 0, NNE = 22.5, ...); label polluted = PM2.5 > 75 ug/m3;
  group = ISO year-week of the timestamp (no week on both sides of a split); 4,000 rows drawn uniformly without
  replacement, numpy seed 0, labels not used.
- solar_boulder: USCRN hourly02, CO_Boulder_14_W 2021-2023; rows with flagged / missing radiation dropped;
  features hour (mid-hour, local standard time) and day of year; label bright = SOLARAD > 200 W/m2;
  group = date; 4,000 rows, numpy seed 0, labels not used.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
RAW = DATA / "raw"
N = 4000
SEED = 0


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def beijing() -> None:
    src = next((RAW / "uci_beijing").glob("PRSA_Data_Dongsi_*.csv"))
    df = pd.read_csv(src)
    wd = {k: 22.5 * i for i, k in enumerate("N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split())}
    df["wd_deg"] = df["wd"].map(wd)
    cols = ["year", "month", "day", "hour", "wd_deg", "WSPM", "TEMP", "PRES", "DEWP", "RAIN", "PM2.5"]
    df = df[cols].dropna().reset_index(drop=True)
    ts = pd.to_datetime(df[["year", "month", "day", "hour"]])
    iso = ts.dt.isocalendar()
    df["week"] = iso["year"].astype(int) * 100 + iso["week"].astype(int)
    df["polluted"] = (df["PM2.5"] > 75).astype(int)
    print(f"beijing: {len(df)} complete rows of {src.name}, {df.week.nunique()} weeks, polluted share {df.polluted.mean():.3f}")
    rng = np.random.default_rng(SEED)
    keep = np.sort(rng.choice(len(df), size=N, replace=False))
    out = df.iloc[keep][["wd_deg", "hour", "month", "WSPM", "TEMP", "PRES", "DEWP", "RAIN", "week", "polluted"]]
    out.to_csv(DATA / "beijing_pm25_4k.csv", index=False)
    print(f"  sample: {len(out)} rows, {out.week.nunique()} weeks, polluted share {out.polluted.mean():.3f}, sha256 {sha(DATA / 'beijing_pm25_4k.csv')}")


def solar() -> None:
    hdr = (RAW / "uscrn" / "headers.txt").read_text().splitlines()[1].split()
    frames = [
        pd.read_csv(f, sep=r"\s+", header=None, names=hdr, dtype={"LST_DATE": str, "LST_TIME": str})
        for f in sorted((RAW / "uscrn").glob("CO_Boulder_14_W_*.txt"))
    ]
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.SOLARAD > -99) & (d.SOLARAD_FLAG == 0)].copy()
    d["doy"] = pd.to_datetime(d.LST_DATE, format="%Y%m%d").dt.dayofyear
    d["hour"] = d.LST_TIME.str[:2].astype(int) - 0.5  # end-of-hour stamp -> mid-hour
    d.loc[d.hour < 0, "hour"] += 24
    d["date"] = d.LST_DATE
    d["bright"] = (d.SOLARAD > 200).astype(int)
    d = d[["hour", "doy", "date", "bright"]].reset_index(drop=True)
    print(f"solar: {len(d)} rows, {d.date.nunique()} days, bright share {d.bright.mean():.3f}")
    rng = np.random.default_rng(SEED)
    keep = np.sort(rng.choice(len(d), size=N, replace=False))
    out = d.iloc[keep]
    out.to_csv(DATA / "solar_boulder_4k.csv", index=False)
    print(f"  sample: {len(out)} rows, {out.date.nunique()} days, bright share {out.bright.mean():.3f}, sha256 {sha(DATA / 'solar_boulder_4k.csv')}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["beijing", "solar"]
    for w in which:
        {"beijing": beijing, "solar": solar}[w]()
