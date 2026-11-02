#!/usr/bin/env python3
"""Reproduce Table 3 of the paper (detector quality on the gold standard) from the released files.

    python tools/reproduce_table3.py            # writes reports/table3.md

Uses the evaluation functions of src/step2_4_eval_detector.py on:
  data/gold/gold_2.3.csv                      gold labels (part 1 — full reading, part 2 — candidates)
  data/gold/candidates_sample_key.csv         split, group and detector-v2 stratum of the 400 candidates
  data/gold/candidate_strata_population.csv   stratum sizes used as weights
  data/annotation/pilot/pilot_gold.csv        pilot books (dev, guidelines v0.3)
  data/detector_output/candidates_*.csv.gz    detector outputs: v2, v3 without lexicon, v3 (frozen)
No book texts are needed.
"""
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ev", ROOT / "src" / "step2_4_eval_detector.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def load_cand(name):
    c = pd.read_csv(ROOT / "data" / "detector_output" / f"candidates_{name}.csv.gz", keep_default_na=False,
                    dtype={"kidlit_id": str})
    c["context"] = ""          # sentences are not released; the evaluation does not need them
    return c


def main():
    gdir = ROOT / "data" / "gold"
    p1, p2 = ev.load_gold(gdir, ROOT / "data" / "annotation" / "pilot" / "pilot_gold.csv")
    key2 = pd.read_csv(gdir / "candidates_sample_key.csv")
    pop = pd.read_csv(gdir / "candidate_strata_population.csv")
    rows = []
    for name, label in [("v2", "v2"), ("v3_nolex", "v3 without lexicon"), ("v3", "v3")]:
        cand = load_cand(name)
        for split in ["test", "dev"]:
            L, err = [], []
            r1 = ev.eval_part1(cand, p1, split, L, err)
            r2 = ev.eval_part2(cand, p2, key2, pop, split, L, err)
            P, R, f1, tp, ndet, ngold = r1["DIM"]
            rows.append((label, split, P, R, f1, r2["P_w"], r2["acc_w"], round(r2["P_u"][0] * r2["n_dim"]), r2["n_dim"]))
    L = ["# Table 3 reproduced from the released files\n",
         "Full reading: precision (P), recall (R) and F1 for DIM over all words of the books, 95% Wilson intervals; "
         "candidates: weighted DIM precision and weighted share of correct labels (5 classes).\n",
         "| Detector | Sample | P | R | F1 | Candidate P | Share correct | Candidates labelled DIM (correct / all) |",
         "|---|---|---|---|---|---|---|---|"]
    for label, split, P, R, f1, pw, acc, cd, nd in rows:
        L.append(f"| {label} | {split} | {P[0]:.3f} ({P[1]:.3f}–{P[2]:.3f}) | {R[0]:.3f} ({R[1]:.3f}–{R[2]:.3f}) | "
                 f"{f1:.3f} | {pw:.3f} | {acc:.3f} | {cd} / {nd} |")
    out = "\n".join(L) + "\n"
    (ROOT / "reports" / "table3.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
