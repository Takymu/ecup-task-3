"""Write a pruned copy of a feature tag: keep only listed feature columns (+ meta cols).
Usage: python build_pruned_tag.py --tag v2 --out-tag v2p77 --list artifacts/feat_prune_lists.json --key p77
"""
from __future__ import annotations
import argparse, json
import polars as pl
from common import FEATURES_DIR, TASK_DIR

META = ["user_id", "anchor_date", "target", "days_active_total"]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2"); ap.add_argument("--out-tag", required=True)
    ap.add_argument("--list", default=str(TASK_DIR / "artifacts" / "feat_prune_lists.json")); ap.add_argument("--key", required=True)
    a = ap.parse_args()
    keep = json.loads(open(a.list).read())[a.key]
    out = FEATURES_DIR / a.out_tag; out.mkdir(parents=True, exist_ok=True)
    files = sorted((FEATURES_DIR / a.tag).glob("anchor_*.parquet"))
    for f in files:
        df = pl.read_parquet(f)
        cols = [c for c in META if c in df.columns] + [c for c in keep if c in df.columns and c not in META]
        df.select(cols).write_parquet(out / f.name)
    print(f"{a.out_tag}: {len(files)} files, {len(cols)} cols ({len(keep)} features)", flush=True)

if __name__ == "__main__":
    main()
