"""Register stage-2 datasets (not bundled in the wheel) into the shipped benchmark engine.

Up: ../README.md (protocol §2.2, dataset entry rule) · Data: ../data/README.md (sources, licences, SHA-256).

Importing this module injects `_Spec` entries into ``qlassifier.datasets._SPECS`` and published quantum numbers
into ``qlassifier.bench.protocol.PUBLISHED``; every loader verifies the file's SHA-256 itself (the engine's own
check only covers files inside the package). Workers import this module too (joblib pickles tasks by reference),
so every process sees the same registry. Nothing in ``src/`` changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qlassifier import datasets as ds
from qlassifier.bench import protocol
from qlassifier.bench.result import Published

DATA = Path(__file__).resolve().parents[1] / "data"
DOWNLOADS = DATA / "downloads"

SHA256 = {
    "spambase/spambase.data": "b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12",
    "banknote/data_banknote_authentication.txt": "d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9",
    "haberman/haberman.data": "b4b7a32586a5668f9f4d6dc8be9d1bc8cd4822523affb1f6b5bfc350681ef3e2",
}
# Derived tables built by build_geo_datasets.py (hashes recorded in ../data/README.md after the build).
DERIVED = {
    "beijing_pm25": DATA / "beijing_pm25_4k.csv",
    "solar_boulder": DATA / "solar_boulder_4k.csv",
    "deep_quakes": Path(__file__).resolve().parents[2] / "classifier" / "data" / "deep_quakes.csv",
}


def _verified(rel: str) -> Path:
    path = DOWNLOADS / rel
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[rel]:
        raise RuntimeError(f"SHA-256 mismatch for {rel}: {digest}")
    return path


def _parse_spambase(_: Any) -> tuple[pd.DataFrame, NDArray[np.int64], None]:
    df = pd.read_csv(_verified("spambase/spambase.data"), header=None)
    X = df.iloc[:, :-1].set_axis([f"f{j + 1:02d}" for j in range(df.shape[1] - 1)], axis=1)
    return X, df.iloc[:, -1].to_numpy(np.int64), None  # 1 = spam (1,813 of 4,601)


def _parse_banknote(_: Any) -> tuple[pd.DataFrame, NDArray[np.int64], None]:
    df = pd.read_csv(_verified("banknote/data_banknote_authentication.txt"), header=None)
    X = df.iloc[:, :4].set_axis(["variance", "skewness", "curtosis", "entropy"], axis=1)
    return X, df.iloc[:, 4].to_numpy(np.int64), None  # 1 = forged (610 of 1,372)


def _parse_haberman(_: Any) -> tuple[pd.DataFrame, NDArray[np.int64], None]:
    df = pd.read_csv(_verified("haberman/haberman.data"), header=None)
    X = df.iloc[:, :3].set_axis(["age", "year", "nodes"], axis=1)
    return X, (df.iloc[:, 3] == 2).to_numpy(np.int64), None  # 1 = died within 5 years (81 of 306)


def _parse_beijing(_: Any) -> tuple[pd.DataFrame, NDArray[np.int64], NDArray[Any]]:
    df = pd.read_csv(DERIVED["beijing_pm25"])
    X = df[["wd_deg", "hour", "month", "WSPM", "TEMP", "PRES", "DEWP", "RAIN"]].copy()
    return X, df["polluted"].to_numpy(np.int64), df["week"].to_numpy()


def _parse_solar(_: Any) -> tuple[pd.DataFrame, NDArray[np.int64], NDArray[Any]]:
    df = pd.read_csv(DERIVED["solar_boulder"])
    X = df[["hour", "doy"]].copy()
    return X, df["bright"].to_numpy(np.int64), df["date"].to_numpy()


def _parse_quakes(_: Any) -> tuple[pd.DataFrame, NDArray[np.int64], None]:
    df = pd.read_csv(DERIVED["deep_quakes"])
    return df[["latitude", "longitude"]].copy(), df["deep"].to_numpy(np.int64), None


SPECS = [
    ds._Spec(
        "spambase", "Spambase (e-mail spam)", None, None, "spam", "https://archive.ics.uci.edu/dataset/94",
        "CC BY 4.0", "Hopkins, Reeber, Forman, Suermondt (1999), UCI Machine Learning Repository, doi:10.24432/C53G6X",
        "pca8", _parse_spambase,
    ),
    ds._Spec(
        "banknote", "Banknote authentication", None, None, "forged", "https://archive.ics.uci.edu/dataset/267",
        "CC BY 4.0", "Lohweg (2012), UCI Machine Learning Repository, doi:10.24432/C55P57", "pca8", _parse_banknote,
    ),
    ds._Spec(
        "haberman", "Haberman's survival", None, None, "died within 5 years",
        "https://archive.ics.uci.edu/dataset/43", "CC BY 4.0",
        "Haberman (1976); UCI Machine Learning Repository, doi:10.24432/C5XK51", "pca8", _parse_haberman,
    ),
    ds._Spec(
        "beijing_pm25", "Beijing PM2.5 (Dongsi)", None, None, "PM2.5 > 75 µg/m³",
        "https://archive.ics.uci.edu/dataset/501", "CC BY 4.0",
        "Zhang et al. (2017), Proc. R. Soc. A 473:20170457; UCI 501, doi:10.24432/C5RK5G", "native", _parse_beijing,
        {"wd_deg": 360.0, "hour": 24.0, "month": 12.0},
    ),
    ds._Spec(
        "solar_boulder", "Solar radiation (USCRN Boulder)", None, None, "solar radiation > 200 W/m²",
        "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/", "U.S. public domain (NOAA NCEI)",
        "Diamond et al. (2013), Bull. Am. Meteorol. Soc. 94:485-498 (U.S. Climate Reference Network)", "native",
        _parse_solar, {"hour": 24.0, "doy": 365.25},
    ),
    ds._Spec(
        "deep_quakes", "Deep earthquakes (USGS ComCat)", None, None, "depth ≥ 70 km",
        "https://earthquake.usgs.gov/fdsnws/event/1/", "U.S. public domain (USGS)",
        "U.S. Geological Survey (2017), ANSS Comprehensive Earthquake Catalog, doi:10.5066/F7MS3QZH", "native",
        _parse_quakes, None,  # sphere geometry is passed to Qlassifier directly in e05: the shipped engine's
    ),  # ``protocol.arrays`` resolves circle declarations only (phase-8 note: extend it before bundling a sphere set)
]  # fmt: skip

PUBLISHED = {
    "spambase": Published(value=0.888, metric="balanced_accuracy", source="arXiv:2604.18837", method="Belis fixed map (1,000-row subsample)"),
    "banknote": Published(value=0.921, metric="balanced_accuracy", source="arXiv:2604.18837", method="Rot2DoF"),
    "haberman": Published(value=0.586, metric="balanced_accuracy", source="arXiv:2604.18837", method="QKT (Sakhnenko10 fixed 0.581)"),
}  # fmt: skip


def register() -> None:
    """Idempotent: add the stage-2 specs and published numbers to the shipped registries."""
    for s in SPECS:
        ds._SPECS.setdefault(s.id, s)
    for k, v in PUBLISHED.items():
        protocol.PUBLISHED.setdefault(k, v)


register()
