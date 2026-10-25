# Data files (stage 2)

Up: [../README.md](../README.md) (protocol §2.2, dataset entry rule) · Loaders: [`../common/extra_datasets.py`](../common/extra_datasets.py) · Derived-table builder: [`../common/build_geo_datasets.py`](../common/build_geo_datasets.py) · Stage-1 files: [`../../../01-winning-classifier/experiments/data/README.md`](../../classifier/data/README.md)

Downloaded files live under `downloads/<name>/` and `raw/` (both gitignored; re-fetched and hash-checked by [`../common/fetch_data.sh`](../common/fetch_data.sh)); they are data and are never executed. Only the derived ≤ 1 MB tables below are committed. Every loader verifies the SHA-256 below before reading. Downloaded 2026-10-25 unless stated.

## UCI datasets (BT-1 candidates)
| File | Dataset | Source | Licence | Rows × cols | SHA-256 |
|---|---|---|---|---|---|
| `downloads/spambase/spambase.zip` → `spambase.data` | Spambase | UCI id 94, https://archive.ics.uci.edu/static/public/94/spambase.zip | CC BY 4.0 | 4,601 × 57 + label | zip `813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636`; data `b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12` |
| `downloads/banknote/banknote.zip` → `data_banknote_authentication.txt` | Banknote authentication | UCI id 267, https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip | CC BY 4.0 | 1,372 × 4 + label | zip `1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227`; data `d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9` |
| `downloads/haberman/haberman.zip` → `haberman.data` | Haberman's survival | UCI id 43, https://archive.ics.uci.edu/static/public/43/haberman+s+survival.zip | CC BY 4.0 | 306 × 3 + label | zip `1001bac29022350e279b064003c426c8ed60f4d76ab4916fb05baf886c8e3ac7`; data `b4b7a32586a5668f9f4d6dc8be9d1bc8cd4822523affb1f6b5bfc350681ef3e2` |

Loader rules:
- Spambase: 57 numeric columns (`f01`…`f57`: word / char frequencies, capital-run statistics), label column 58: 1 = spam (1,813 of 4,601 = 39.4%), 0 = non-spam. No missing values. Recipe `pca8`.
- Banknote: 4 wavelet statistics (variance, skewness, curtosis, entropy), label 1 = forged (610 of 1,372 = 44.5%). Recipe `pca8` (d = 4 ≤ 8 → no PCA; 2 qubits). Published: Rot2DoF BA 0.921, classical RBF 1.000 (arXiv:2604.18837).
- Haberman: age, operation year, positive axillary nodes; label `2` (died within 5 years) = 1 (81 of 306 = 26.5%). Recipe `pca8` (d = 3 → no PCA; 2 qubits: one RX·RY pair + one lone RX). Published: Sakhnenko10 fixed map BA 0.581, QKT 0.586, RBF 0.549 (arXiv:2604.18837, the only quantum lead in that study).
- Citations: Hopkins, Reeber, Forman, Suermondt (1999), Spambase, UCI, doi:10.24432/C53G6X; Lohweg (2012), Banknote authentication, UCI, doi:10.24432/C55P57; Haberman (1976), UCI, doi:10.24432/C5XK51.

## Angle-native datasets (BT-5 candidates)
Raw inputs copied from the stage-1 groundwork download (`.cache/ideas/bloch-sphere-angular-data/data/`, fetched 2026-10-24); hashes of the copies below.

| File | Dataset | Source | Licence | SHA-256 |
|---|---|---|---|---|
| `raw/uci_beijing/PRSA_Data_Dongsi_20130301-20170228.csv` | UCI 501 Beijing Multi-Site Air Quality, Dongsi station (from `PRSA2017_Data_20130301-20170228.zip`, sha256 `d1b9261c…8b8`) | https://archive.ics.uci.edu/dataset/501 (doi:10.24432/C5RK5G) | CC BY 4.0 | `9c0020214425e5ee2a64d42f97a1f632e1d8f6b53ad50edd85a686cc7999f27e` |
| `raw/uscrn/CO_Boulder_14_W_2021.txt`, `_2022.txt`, `_2023.txt`, `headers.txt` | NOAA USCRN hourly02, Boulder CO | https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/ | U.S. government work (public domain) | `b1c389b5f45c7312d243b06cc1f8cca5dd971687d6a0c79defee9f1a1ed9aa11`, `e2c8b22fcef736cc3a287bde5a3267af4be2985026cd1802ecd65b5a9f6e9cc5`, `db472400baedd545dfab49ed7db33b7da78deb9cc29faaa5821afc51e98b896b`, `44564d2ca37fe7c2e893495c7fd339621817052e70373c3bd562d6fbdd1375a4` |

Derived tables (built 2026-10-25 by `python -I common/build_geo_datasets.py`, before any model run; label-free sampling, numpy seed 0):

| File | Rows × cols | Build rule | Positive share | Groups | SHA-256 |
|---|---|---|---|---|---|
| `beijing_pm25_4k.csv` | 4,000 × 10 (`wd_deg, hour, month, WSPM, TEMP, PRES, DEWP, RAIN, week, polluted`) | 34,230 complete hourly rows (rows with any missing value dropped); wind direction 16-point compass → degrees (N = 0°, NNE = 22.5°, …); `polluted` = PM2.5 > 75 µg/m³; `week` = ISO year × 100 + ISO week; 4,000 rows uniform without replacement | 0.424 (full 0.424) | 210 weeks (all weeks present) | `0dba313318071926c9c89996ca565fa4cc77be63da1642bf4e82afb99497a704` |
| `solar_boulder_4k.csv` | 4,000 × 4 (`hour, doy, date, bright`) | 26,272 rows with unflagged radiation (`SOLARAD > −99`, `SOLARAD_FLAG == 0`); `hour` = mid-hour local standard time; `doy` = day of year; `bright` = SOLARAD > 200 W/m²; 4,000 rows uniform without replacement | 0.291 (full 0.293) | 1,076 dates | `057196f8f98e97e2f9a9374a90749c4613c91fda1d17645094898c3e6bf6e1f2` |

- Geometry declared: `beijing_pm25` {wd_deg: 360, hour: 24, month: 12} + plain WSPM, TEMP, PRES, DEWP, RAIN (3 circle qubits + 3 pair qubits = 6 qubits); `solar_boulder` {hour: 24, doy: 365.25} (2 circle qubits). Recipe `native`, grouped splits (`week` / `date`), drift rule as stage 1.
- Citations: Zhang, Guo, Dong, He, Xu, Chen (2017), Cautionary tales on air-quality improvement in Beijing, Proc. R. Soc. A 473:20170457; Diamond et al. (2013), U.S. Climate Reference Network after one decade of operations, Bull. Am. Meteorol. Soc. 94:485–498.

## Reused stage-1 file
- `deep_quakes.csv` (USGS ComCat M ≥ 6, 2000–2024; 3,729 × 5; U.S. public domain; sha256 `1ead2a1b7119781adadd6cc51a11a5bb156e6dfcce9a1223b2cee7e013482f65`), read in place from the stage-1 data folder by `extra_datasets.py` for [e05](../e05-quake-shots/README.md).
