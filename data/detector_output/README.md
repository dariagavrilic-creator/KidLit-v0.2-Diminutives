# Detector output

Labels assigned by the detector to every candidate token of the 120 books (one row per token). Sentences and context windows are removed.

| File | Rows | Detector |
|---|---|---|
| `candidates_v2.csv.gz` | 13,958 | v2, used to stratify the gold sample. |
| `candidates_v3_nolex.csv.gz` | 13,904 | v3 rules without the lexicon (ablation in Table 3). |
| `candidates_v3.csv.gz` | 13,958 | v3, frozen, evaluated in Table 3 (lexicon of 81 lemmas). |
| `candidates_v3.1.csv.gz` | 13,967 | v3.1, used for the corpus analysis (lexicon of 104 lemmas): 5,508 DIM, 1,596 LEX, 1,363 YOUNG, 100 HYP. |

## Columns

| Column | Meaning |
|---|---|
| `kidlit_id`, `book_id`, `group`, `sent_id`, `tok_id`, `start`, `end` | Token address; `start` / `end` are character offsets in the normalized text. |
| `token`, `lemma`, `pos`, `gram` | Word form and pymorphy3 analysis. |
| `cand_suffix` | Suffix that made the token a candidate. |
| `base_lemma`, `base_source` | Base word and where it was found (`list`, `derivbase`, `pymorphy`). |
| `det_label` | `DIM`, `LEX`, `HYP`, `YOUNG`, `NO`. |
| `rule_id` | Rule that fired: R0 form without suffix → NO, R1 lexicon, R2 YOUNG, R3 HYP, R4 DIM, R5 NO. |
| `reject_reason` | Why the candidate got NO (`gender`, `animacy`, `base_rarer`, `proper_base`, `feminitive`, `form_without_suffix`…). |
| `pej`, `name`, `nonce`, `precedent` | Attributes and precedent applied. |
| `component` | For hyphenated compounds (*мама-лисица*): the part that was analysed. |
| `needs_review` | 1 = context-dependent case (lexicon label `CONTEXT`). |
| `detector_version` | Version string. |
