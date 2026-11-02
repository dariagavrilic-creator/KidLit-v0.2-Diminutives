#!/usr/bin/env python3
"""Recompute inter-annotator agreement from the released annotation tables.

    python tools/compute_iaa.py            # writes reports/iaa.md

Needs only the files in data/annotation/ (no book texts). Reproduces the agreement
figures of the paper (Section 5.2) and of the annotation guide (pilot, v0.3).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "annotation"
KEY = ["kidlit_id", "sent_id", "tok_id"]


def kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def pos_f1(a, b, lab):
    sa, sb = (a == lab).sum(), (b == lab).sum()
    both = ((a == lab) & (b == lab)).sum()
    return 2 * both / (sa + sb) if sa + sb else float("nan"), int(sa), int(sb), int(both)


def read(p):
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    d["label"] = d.label.str.strip().str.upper()
    return d


def full_reading(a, b):
    m = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    la, lb = m.label_a.replace("", "—"), m.label_b.replace("", "—")
    f1, sa, sb, both = pos_f1(la, lb, "DIM")
    return [f"| Words compared | {len(m)} |",
            f"| Positive agreement on DIM (F1) | {f1:.3f} (A {sa}, B {sb}, both {both}) |",
            f"| κ, 4 classes (—/DIM/HYP/YOUNG) | {kappa(la, lb):.3f} |",
            f"| κ, DIM / non-DIM | {kappa(la == 'DIM', lb == 'DIM'):.3f} |"]


def candidates(a, b, idc):
    m = a.merge(b, on=idc, suffixes=("_a", "_b"))
    return [f"| Candidates compared | {len(m)} |",
            f"| κ, 5 labels | {kappa(m.label_a, m.label_b):.3f} |",
            f"| κ, DIM / non-DIM | {kappa(m.label_a == 'DIM', m.label_b == 'DIM'):.3f} |",
            f"| Raw agreement | {(m.label_a == m.label_b).mean():.3f} |"]


def main():
    L = ["# Inter-annotator agreement (before adjudication)\n",
         "Computed by `tools/compute_iaa.py` from the released annotation tables.\n"]
    for name, sub, pre in [("Pilot (guide v0.2 → v0.3)", "pilot", "pilot"), ("KidLit-Gold (guide v0.3 → v0.4)", "gold", "gold")]:
        A1, B1 = read(ANN / sub / f"{pre}_A_part1_full_reading.csv"), read(ANN / sub / f"{pre}_B_part1_full_reading.csv")
        A2, B2 = read(ANN / sub / f"{pre}_A_part2_candidates.csv"), read(ANN / sub / f"{pre}_B_part2_candidates.csv")
        L += [f"## {name}\n", "### Part 1. Full reading (doubly annotated books)\n", "| Measure | Value |", "|---|---|"]
        L += full_reading(A1, B1)
        L += ["\n### Part 2. Candidates (doubly annotated items)\n", "| Measure | Value |", "|---|---|"]
        L += candidates(A2, B2, "item_id")
        L.append("")
    out = ROOT / "reports" / "iaa.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
