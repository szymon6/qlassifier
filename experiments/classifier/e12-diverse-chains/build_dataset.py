"""e12 step 2: build `protein_div.csv` from the downloaded chains (pre-registered data plan, learning log iteration 5).

Per entry: first chain, altloc ' '/'A', no insertion codes, backbone N/CA/C; phi/psi per residue (chain breaks skipped:
C(i-1)-N(i) and C(i)-N(i+1) <= 1.5 A); label = residue inside a SHEET record. Windows i-2..i+2 with consecutive
sequence numbers and all 10 dihedrals defined. Chains without any HELIX and SHEET record are skipped (no annotation).
Sampling (label-free): per chain up to 3 residues at sequence spacing >= 5 (numpy seed 0), then 4,000 rows (seed 0).
Usage: .venv/bin/python -I build_dataset.py RAW_DIR OUT_CSV
"""
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

raw, out_csv = Path(sys.argv[1]), Path(sys.argv[2])


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - (b0 @ b1n) * b1n
    w = b2 - (b2 @ b1n) * b1n
    return float(np.degrees(np.arctan2(np.cross(b1n, v) @ w, v @ w)))


def parse(path: Path):
    helix, sheet, atoms, order, chain0 = set(), set(), {}, [], None
    for line in gzip.open(path, "rt", errors="ignore"):
        rec = line[:6]
        if rec == "HELIX ":
            try:
                helix |= {(line[19], i) for i in range(int(line[21:25]), int(line[33:37]) + 1)}
            except ValueError:
                pass
        elif rec == "SHEET ":
            try:
                sheet |= {(line[21], i) for i in range(int(line[22:26]), int(line[33:37]) + 1)}
            except ValueError:
                pass
        elif rec == "ATOM  ":
            if line[16] not in (" ", "A") or line[26] != " ":
                continue
            ch = line[21]
            chain0 = ch if chain0 is None else chain0
            if ch != chain0:
                continue
            name, resn, seq = line[12:16].strip(), line[17:20], int(line[22:26])
            if name not in ("N", "CA", "C"):
                continue
            key = (ch, seq)
            if key not in atoms:
                atoms[key] = {"resn": resn}
                order.append(key)
            atoms[key][name] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif rec == "ENDMDL":
            break
    if chain0 is None or not ({k for k in helix if k[0] == chain0} or {k for k in sheet if k[0] == chain0}):
        return None
    phi, psi = {}, {}
    for i in range(1, len(order) - 1):
        kp, k, kn = order[i - 1], order[i], order[i + 1]
        A, P, Nn = atoms[k], atoms[kp], atoms[kn]
        if not all(x in A for x in ("N", "CA", "C")) or "C" not in P or "N" not in Nn:
            continue
        if np.linalg.norm(P["C"] - A["N"]) > 1.5 or np.linalg.norm(A["C"] - Nn["N"]) > 1.5:
            continue
        phi[k[1]] = dihedral(P["C"], A["N"], A["CA"], A["C"])
        psi[k[1]] = dihedral(A["N"], A["CA"], A["C"], Nn["N"])
    rows = []
    for (ch, s) in order:
        win = [s - 2, s - 1, s, s + 1, s + 2]
        if all(w in phi for w in win):
            r = {"pdb": path.name[:4].upper(), "seq": s, "resn": atoms[(ch, s)]["resn"], "ss": "E" if (ch, s) in sheet else "-"}
            for tag, w in zip(["m2", "m1", "0", "p1", "p2"], win):
                r[f"phi_{tag}"], r[f"psi_{tag}"] = phi[w], psi[w]
            rows.append(r)
    return rows


frames = []
skipped = 0
for f in sorted(raw.glob("*.pdb.gz")):
    try:
        r = parse(f)
    except Exception:
        r = None
    if not r:
        skipped += 1
        continue
    frames.append(pd.DataFrame(r))
allr = pd.concat(frames, ignore_index=True)
print("chains used", allr.pdb.nunique(), "skipped", skipped, "candidate residues", len(allr), "strand share", round((allr.ss == "E").mean(), 3))
rng = np.random.default_rng(0)
picked = []
for pdb, g in allr.groupby("pdb", sort=True):
    seqs = g.seq.to_numpy()
    order = rng.permutation(len(g))
    chosen = []
    for j in order:
        if all(abs(seqs[j] - seqs[c]) >= 5 for c in chosen):
            chosen.append(j)
        if len(chosen) == 3:
            break
    picked.append(g.iloc[sorted(chosen)])
pool = pd.concat(picked, ignore_index=True)
take = np.sort(np.random.default_rng(0).choice(len(pool), size=min(4000, len(pool)), replace=False))
df = pool.iloc[take].reset_index(drop=True)
cols = ["pdb", "seq", "resn"] + [f"{a}_{t}" for t in ["m2", "m1", "0", "p1", "p2"] for a in ("phi", "psi")] + ["ss"]
df = df[cols]
df.to_csv(out_csv, index=False)
print("pool", len(pool), "rows", len(df), "chains", df.pdb.nunique(), "max per chain", df.groupby("pdb").size().max(),
      "strand share", round((df.ss == "E").mean(), 3))
