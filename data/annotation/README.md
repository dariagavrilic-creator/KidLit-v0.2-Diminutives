# Annotation

Independent diminutive annotations by annotators A and B, **before adjudication**, blind to the detector's labels.
Final decisions are in `../adjudication/` and the resulting gold labels in `../gold/`.

| Folder / file | Rows | What it is |
|---|---|---|
| `pilot/pilot_{A,B}_part1_full_reading.csv` | 1,000 each | Pilot, full reading of 2 books (guidelines v0.2). |
| `pilot/pilot_{A,B}_part2_candidates.csv` | 103 each | Pilot, detector candidates. |
| `pilot/pilot_gold.csv` | 1,103 | Pilot gold labels relabelled by guidelines v0.3; used as dev data in Table 3. |
| `pilot/pilot_gold_v0.2.csv` | 1,103 | Original pilot gold labels (guidelines v0.2). |
| `gold/gold_{A,B}_part1_full_reading.csv` | 4,609 / 4,703 | KidLit-Gold, full reading of 6 books; 2 books were read by both annotators. |
| `gold/gold_{A,B}_part2_candidates.csv` | 252 / 248 | KidLit-Gold, 400 detector candidates from other books; 100 were labelled by both annotators. |

## Format

One row per token. In part 1 every word of the book is listed in order, but `token` and `lemma` are filled
only for tokens an annotator labelled or that were discussed in adjudication; the other rows keep only their address.

| Column | Meaning |
|---|---|
| `kidlit_id`, `sent_id`, `tok_id` | Token address (0-based sentence and token, punctuation counted; razdel tokenization of step 0). |
| `row_id` / `item_id` | Row in the file (part 1) / candidate ID `C001`… (part 2). |
| `label` | Part 1: `DIM`, `HYP`, `YOUNG` or empty. Part 2: `DIM`, `LEX`, `HYP`, `YOUNG`, `NO`. |
| `base` | Base word (precedent P-17). |
| `pej`, `name`, `nonce`, `foreign` | Attributes, 0/1. |
| `precedent` | Precedent applied (P-01…P-26). |
| `unsure`, `comment` | Annotator's doubt (0/1) and comment. |

Pilot gold files: `part`, address, `token`, `gold` (label), `guideline` (version).
Agreement figures of the paper: `python tools/compute_iaa.py` (report `reports/iaa.md`).
