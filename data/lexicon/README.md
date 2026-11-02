# Precedent lexicon

Word lists the detector consults first (rule R1) for words whose label cannot be derived from the form:
lexicalized diminutives, look-alikes, names of young animals and fixed decisions from the precedent log.

| File | Lemmas | Used for |
|---|---|---|
| `dim_lexicon_v3.csv` | 81 | Detector v3 as evaluated in Table 3 (frozen). |
| `dim_lexicon_v3.1.csv` | 104 | Detector v3.1: the same rules, the v3 lexicon plus 23 lemmas added after the test run; used to annotate the whole corpus. |

| Column | Meaning |
|---|---|
| `lemma` | Lemma (pymorphy3 normal form). |
| `label` | `DIM`, `LEX`, `YOUNG`, `NO`, or `CONTEXT` (*ручка*, *ножка*: labelled DIM and flagged `needs_review`, precedent P-04). |
| `base` | Base word (precedent P-17). |
| `pej`, `name` | Attributes, 0/1. |
| `precedent`, `source`, `comment` | Precedent or guideline section, version the entry was added in (seed v2 / v3 / v3.1), note. |

Usage: `python src/kidlit_dim_detect_v3.py ... --lexicon data/lexicon/dim_lexicon_v3.1.csv`.
