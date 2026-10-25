"""e12 step 1: fetch non-redundant protein chains from RCSB (PDB data: CC0 1.0). Pre-registered data plan (learning log, iteration 5).

Query = groundwork filters (`.cache/ideas/bloch-sphere-angular-data/fetch_pdb.py`): X-ray, resolution <= 1.6 A,
protein-only entry, 1 polymer entity, length 80-400, one representative per 30%-identity cluster, release date ascending.
Usage: .venv/bin/python -I fetch_pdb.py OUT_DIR N_WANT   (downloads legacy <ID>.pdb.gz into OUT_DIR; writes OUT_DIR/ids.txt)
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

out, n_want = Path(sys.argv[1]), int(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
q = {
    "query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {"attribute": "exptl.method", "operator": "exact_match", "value": "X-RAY DIFFRACTION"}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": 1.6}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.selected_polymer_entity_types", "operator": "exact_match", "value": "Protein (only)"}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count", "operator": "equals", "value": 1}},
        {"type": "terminal", "service": "text", "parameters": {"attribute": "entity_poly.rcsb_sample_sequence_length", "operator": "range",
                                                                 "value": {"from": 80, "to": 400, "include_lower": True, "include_upper": True}}}]},
    "return_type": "polymer_entity",
    "request_options": {
        "group_by": {"aggregation_method": "sequence_identity", "similarity_cutoff": 30,
                     "ranking_criteria_type": {"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}},
        "group_by_return_type": "representatives",
        "paginate": {"start": 0, "rows": n_want},
        "sort": [{"sort_by": "rcsb_accession_info.initial_release_date", "direction": "asc"}]}}
req = urllib.request.Request("https://search.rcsb.org/rcsbsearch/v2/query", data=json.dumps(q).encode(),
                             headers={"Content-Type": "application/json"})
res = json.load(urllib.request.urlopen(req, timeout=120))
ids = sorted({r["identifier"].split("_")[0] for r in res.get("result_set", [])})
print("groups:", res.get("group_by_count", res.get("total_count")), "representatives:", len(ids), flush=True)
(out / "ids.txt").write_text("\n".join(ids) + "\n")


def get(pid: str) -> bool:
    f = out / f"{pid}.pdb.gz"
    if f.exists() and f.stat().st_size > 500:
        return True
    try:
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pid}.pdb.gz", f)
        return True
    except Exception:
        return False  # some entries have no legacy format


with ThreadPoolExecutor(16) as ex:
    ok = sum(ex.map(get, ids))
print("downloaded", ok, "of", len(ids), flush=True)
