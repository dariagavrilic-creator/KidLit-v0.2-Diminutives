# Results

Outputs of step 3 (`src/step3_*.py`), computed with detector v3.1 on all 120 books. Every number of the paper
is listed in `results_master.csv` with the file it comes from.

| File | What it is |
|---|---|
| `results_master.csv` | All numbers of the paper: `section`, `key`, `value`, `ci95`, `n`, `source`, `note`. Start here. |
| `FREEZE_results_v1.json`, `FREEZE_results_v2.json` | Hashes and row counts of the frozen result files. |
| **Book-level frequency** | |
| `book_metrics.csv` | Per-book metrics: DIM per 1,000 words, DIM nouns per 100 nouns, DIM choice (DIM vs. base), HYP / YOUNG rates, pejorative and name shares, book attributes. Source of Fig. 2. |
| `book_metrics_variants.csv` | The same for 4 variants: `base`, `p01_dim`, `p19_lex`, `no_context`. |
| `group_tests.csv` | RUS-O vs. RUS-T: medians, Mann–Whitney, Cliff's δ with CI, OLS with log length, BH q. |
| `sensitivity.csv` | The same tests for the variants of `book_metrics_variants.csv` and for detector labels corrected by the gold standard (`detector_corrected`). |
| `sensitivity_excl.csv` | The same tests without the -k- words that P-20 treats as non-diminutive. |
| `rus_t_by_language.csv` | RUS-T medians by source language. |
| **Semantics** | |
| `semantic_classes_by_group.csv`, `semantic_class_tests.csv` | Semantic classes of DIM by group (Fig. 3) and book-level tests. |
| **Collocations and address** | |
| `amod_relations.csv` | Adjective + noun pairs from the dependency parse (lemmas only). |
| `colloc_amod_dim.csv`, `colloc_amod_pairs.csv` | Adjectives attracted to DIM nouns (logDice, LL) and frequent adjective + DIM pairs. |
| `pleonasm.csv` | Size adjectives (*маленький* etc.) with DIM vs. with the base word (Fig. 5). |
| `address.csv`, `address_examples.csv` | DIM in forms of address; addresses of these tokens with the token before and after. |
| **Sentiment and NER (Section 6.4)** | |
| `ls_alignment_check.csv` | Alignment of the sentiment / NER annotation to the corpus texts, per book. |
| `sentiment_by_sentence.csv`, `sentiment_dim.csv` | Sentence sentiment (−1 / 0 / 1, 32 books) and whether the sentence has a DIM; group comparisons. |
| `ner_mentions.csv`, `ner_dim.csv` | Person mentions (PER_NAME, PER_ROLE, PER_FAMILY; ≤ 3 tokens, 60 books) and the share containing DIM. |
| `sentiment_iaa_dim.csv`, `sentiment_disagreement_by_sentence.csv` | Sentiment disagreement in sentences with and without DIM. |
| `gold30_*.csv` | The same analyses on the double sentiment / NER annotation of 30 books (annotators A and B): sentence labels, agreement, DIM effect, NER, coverage of the adjudication log. |

Sentences are identified by `kidlit_id` + `sent_id`; no sentence text is included.
