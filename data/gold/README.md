# Gold standard

KidLit-Gold after adjudication, plus everything needed to reproduce the sample and the weighted evaluation (Table 3).

| File | Rows | What it is |
|---|---|---|
| `gold_2.3.csv` | 7,603 | Gold labels. `part` 1 = full reading (every word of the book), 2 = detector candidates; `split` = dev / test; `gold`, `gold_base`, attributes; `source` = how the label was obtained (`agreed`, `single_A`, `single_B`, `adjudicated_<type>`). |
| `gold_books.csv` | 120 | Role of every book in the gold standard (`split`, `role`, `annotation`) with word and candidate counts. |
| `candidates_sample_key.csv` | 400 | The candidate sample: stratum (`split`, `group`, detector-v2 label), annotator, `double` = labelled by both, and the detector-v2 output for the token. |
| `candidate_strata_population.csv` | 20 | Size of each stratum in the corpus (`population`) and in the sample (`sampled`). Candidate weights = `population / sampled`. |
| `full_reading_detector_v2_key.csv` | 674 | Detector-v2 output on the full-reading books at the time of sampling. |
| `gold_manifest.json` | — | Selection settings: seed, books per role, quotas per stratum, file hashes. |

## Splits

- **test**: RUS-O_long_03, RUS-O_medium_04, RUS-T_long_01, RUS-T_medium_02 (full reading, 4,405 words) and the test candidates.
- **dev**: pilot books RUS-O_medium_05 and RUS-T_short_03, RUS-O_long_01, RUS-T_long_05, and the dev candidates.

The detector rules and lexicon were changed only on dev data. `python tools/reproduce_table3.py` recomputes Table 3 from this folder,
`../annotation/pilot/pilot_gold.csv` and `../detector_output/`.
