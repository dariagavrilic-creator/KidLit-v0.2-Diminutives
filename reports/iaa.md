# Inter-annotator agreement (before adjudication)

Computed by `tools/compute_iaa.py` from the released annotation tables.

## Pilot (guide v0.2 → v0.3)

### Part 1. Full reading (doubly annotated books)

| Measure | Value |
|---|---|
| Words compared | 1000 |
| Positive agreement on DIM (F1) | 0.975 (A 59, B 62, both 59) |
| κ, 4 classes (—/DIM/HYP/YOUNG) | 0.970 |
| κ, DIM / non-DIM | 0.974 |

### Part 2. Candidates (doubly annotated items)

| Measure | Value |
|---|---|
| Candidates compared | 103 |
| κ, 5 labels | 0.967 |
| κ, DIM / non-DIM | 0.960 |
| Raw agreement | 0.981 |

## KidLit-Gold (guide v0.3 → v0.4)

### Part 1. Full reading (doubly annotated books)

| Measure | Value |
|---|---|
| Words compared | 2109 |
| Positive agreement on DIM (F1) | 0.904 (A 114, B 116, both 104) |
| κ, 4 classes (—/DIM/HYP/YOUNG) | 0.902 |
| κ, DIM / non-DIM | 0.899 |

### Part 2. Candidates (doubly annotated items)

| Measure | Value |
|---|---|
| Candidates compared | 100 |
| κ, 5 labels | 0.736 |
| κ, DIM / non-DIM | 0.760 |
| Raw agreement | 0.830 |

