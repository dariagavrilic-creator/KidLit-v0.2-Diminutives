# Pipeline scripts

Scripts are numbered by the steps of the project. All of them read and write the working folder
`kidlit_dim/` (contains the texts, ignored by git).

| Script | Step | Reads | Writes |
|---|---|---|---|
| `step0_corpus_preprocess.py` | 0.2–0.4: freeze the corpus, normalize, tokenize (razdel), parse (spaCy ru_core_news_sm 3.8.0), lemmatize (pymorphy3), reference frequency list | corpus CSV, extra texts (3 books), `Freq2011.zip` | `data/`, `reference/` |
| `step0_5_lemma_repair.py` | 0.3 v2: lemma repair against the frequency dictionary (260 tokens, 54 lemma pairs) | `data/tokens.parquet`, reference | `data/tokens.parquet`, `reference/lemma_repairs.csv` |
| `kidlit_dim_detect_v3.py` | 1.2 / 2.4: diminutive detector v3 (frozen; v3.1 = same rules + full precedent lexicon `dim_lexicon_v3.1.csv`) | `data/tokens.parquet`, lexicon, `reference/freqrnc2011_clean.csv`, DerivBase.Ru | `detector*/candidates.parquet` |
| `step1_pilot.py` | 1.3: pilot annotation files and agreement | candidates | pilot files, `iaa_report.md` |
| `step2_1_gold_sample.py` | 2.1: KidLit-Gold sample (books for full reading, stratified candidates, dev/test split) | candidates, books | `gold/` |
| `step2_3_iaa_adjudication.py` | 2.3: agreement, adjudication file, gold labels | filled annotator files, adjudication | `gold/gold_2.3.csv`, `iaa_report_2.3.md` |
| `step2_4_eval_detector.py` | 2.4: detector evaluation on dev, single run on test | gold, candidates | `eval_dev_test.md/json`, `TEST_RUN.json` |
| `step3_1_book_metrics.py` | 3.1–3.2: book-level metrics, Mann–Whitney, Cliff's δ, BH, OLS (HC3), sensitivity | candidates v3.1, books | `analysis/` |
| `step1_4_keyness.py` | 1.4: keywords vs. the reference dictionary and RUS-O vs. RUS-T (LL, log ratio) | tokens, reference | `keyness/` |
| `step3_3_semantics.py` | 3.3: semantic classes of diminutive lemmas | `dim_lemma_classes` | `analysis/` |
| `step3_4_collocations.py` | 3.4: amod collocations (logDice), size-modifier pleonasm, forms of address | tokens, candidates | `analysis/` |
| `step3_5a_sentiment_ner_link.py`, `step3_5b_gold30_sentiment_ner.py` | 3.5: diminutives and the KidLit-Gold sentiment / NER annotation (separate repository) | candidates, KidLit-Gold export | `analysis/` |
| `step3_7_figures.py` | 3.6: figures 1–5 of the paper | `analysis/` | `analysis/figures/` |
| `step3_6_results_master.py` | 3.6: `results_master.csv` — every number of the paper with its source file | all of the above | `results_master.csv` |

Detector runs used in the paper:

```bash
DB=DerivBaseRu/data/wiktionary/v001.00-connections.zip
python src/kidlit_dim_detect_v3.py --data kidlit_dim --derivbase $DB --lexicon data/lexicon/dim_lexicon_v3.csv   --out-name detector_v3          # evaluated version
python src/kidlit_dim_detect_v3.py --data kidlit_dim --derivbase $DB --lexicon data/lexicon/dim_lexicon_v3.csv   --no-lexicon --out-name detector_v3_nolex
python src/kidlit_dim_detect_v3.py --data kidlit_dim --derivbase $DB --lexicon data/lexicon/dim_lexicon_v3.1.csv --out-name detector_v3.1        # corpus annotation (Section 6)
```

The outputs (without sentences) are in `data/detector_output/`. Rerunning v3.1 on the project's token table gives
exactly the released file: 5,508 DIM, 100 HYP, 1,363 YOUNG, 1,596 LEX, the same counts in every book as
`data/results/book_metrics.csv`.

Full chain, checked on the project data: `step0_corpus_preprocess.py` reproduces the project's frozen texts (same
SHA-256), book IDs, book statistics and token table (lemmas, POS, tags; only `pm_score` differs in 5 tokens);
`step0_5_lemma_repair.py` then gives the token table used by all later steps (identical, 260 repaired tokens);
detector v3.1 on it gives `data/detector_output/candidates_v3.1.csv.gz`. Versions used for the check: razdel 0.5.0,
spaCy 3.8 with ru_core_news_sm 3.8.0, pymorphy3 2.0.6 with pymorphy3-dicts-ru 2.4.417150.4580142,
DerivBase.Ru commit 0f246dc (`data/wiktionary/v001.00-connections.zip`).
