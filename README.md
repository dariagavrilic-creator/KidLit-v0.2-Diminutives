# KidLit by Numbers: diminutives in Russian 0+ children's books

Data and code for the paper

> [Anonymized Authors] KidLit by Numbers: Diminutives in Original and Translated Russian 0+ Books.
> *Trudy ISP RAN / Proc. ISP RAS*, 2026 (in press).

The study compares 60 original Russian books rated 0+ (RUS-O) and 60 translated books (RUS-T) from the
KidLit v0.2 corpus (120 books, 176,175 words). A rule-based detector finds diminutives (DIM) and separates
them from lexicalized forms (LEX), hypocoristic names (HYP), names of young animals (YOUNG) and look-alikes (NO).
It is evaluated on a double-annotated gold standard with a held-out test set.

**The book texts are not included.** The books are protected by copyright. This repository contains
annotations, metadata, lemma-level tables and code. Annotations point to tokens by
`kidlit_id / sent_id / tok_id`; with your own copy of the books, the pipeline reproduces the token
positions. See [DATA_STATEMENT.md](DATA_STATEMENT.md).

## Contents

| Path | What it is |
|---|---|
| `docs/annotation_guide_v0.5_public.docx`, `.pdf` | Annotation guidelines v0.5 (in Russian): definition, 5 labels, decision procedure, log of 26 precedents, detector logic, control examples, disputed cases. Quotations from the books are shortened to at most four words. |
| `data/metadata/books.csv` | Bibliographic metadata of the 120 books (no texts, no publisher blurbs) and the `kidlit_id` ↔ `book_id` mapping. |
| `data/metadata/book_stats.csv` | Per-book counts used as denominators (tokens, words, sentences, nouns, OOV). |
| `data/metadata/lemma_repairs.csv` | The 54 lemma pairs corrected after lemmatization (step 0). |
| `data/metadata/freeze_manifest.json`, `normalization_log.csv` | SHA-256 of the frozen raw and normalized texts (to check your copy of the books) and the log of text normalization. |
| `data/lexicon/` | Precedent lexicons of the detector: `dim_lexicon_v3.csv` (81 lemmas, evaluated version) and `dim_lexicon_v3.1.csv` (104 lemmas, corpus annotation). |
| `data/detector_output/` | Detector outputs for the whole corpus without sentences: v2, v3 without lexicon, v3, v3.1 (one row per candidate token). |
| `data/gold/` | Gold standard: gold labels after adjudication (`gold_2.3.csv`), selection manifest and books, the candidate sample key with detector-v2 strata and stratum sizes (weights), detector-v2 labels on the full-reading books. |
| `data/annotation/pilot/` | Pilot (2 books): independent annotations of annotators A and B, part 1 (full reading) and part 2 (detector candidates), and the pilot gold labels (`pilot_gold.csv` — relabelled by guidelines v0.3, used for evaluation; `pilot_gold_v0.2.csv` — original). |
| `data/annotation/gold/` | KidLit-Gold diminutive annotation: part 1 — full reading of 6 books (2 of them by both annotators); part 2 — 400 detector candidates from other books (100 by both annotators). |
| `data/adjudication/` | Adjudication logs: pilot (7 cases) and step 2.3 (81 cases J001–J081) with final labels, precedents and comments; decisions on new precedents P-19–P-26. |
| `data/lemma_tables/` | Semantic classes of the 204 most frequent diminutive lemmas; keyword tables (KidLit vs. the Lyashevskaya–Sharoff frequency dictionary; RUS-O vs. RUS-T) with semantic classes; the class scheme; lemma frequencies of KidLit. `keyness_full/` — keyness statistics for all 12,642 lemmas (reference frequencies are not redistributed). |
| `data/results/` | Step 3 outputs: book-level metrics, group tests, sensitivity analyses, semantic classes by group, collocations and pleonasm, forms of address, links to the KidLit-Gold sentiment / NER annotation (no sentences), and `results_master.csv` — every number of the paper with its source file. |
| `docs/figures/` | Figures of the paper: Fig. 1 `fig1_pipeline`, Fig. 2 `fig2_dim_by_book`, Fig. 3 `fig4_semantic_classes`, Fig. 4 `fig3_keyword_classes`, Fig. 5 `fig5_size_pleonasm` (file names follow the order in which the figures were made). |
| `src/` | Pipeline scripts (steps 0–3) as used in the project. See [src/README.md](src/README.md). |
| `tools/compute_iaa.py` | Recomputes inter-annotator agreement from the released tables (no texts needed). |
| `tools/reproduce_table3.py` | Recomputes the detector evaluation (Table 3) from the released gold labels and detector outputs. |
| `tools/check_no_text.py` | Checks that no fragment of 5+ consecutive words from the books is in the repository (run locally with your copy of the texts). |
| `reports/` | Generated reports: agreement, text-leak check, step 3 analysis reports (in Russian). |

## Quick check: agreement figures of the paper

```bash
pip install pandas
python tools/compute_iaa.py
```

| Measure (before adjudication) | Pilot | KidLit-Gold |
|---|---|---|
| Full reading, positive agreement on DIM (F1) | 0.975 | 0.904 (2,109 words) |
| Candidates, κ over 5 labels | 0.967 | 0.736 (100 items) |
| Candidates, κ DIM / non-DIM | 0.960 | 0.760 |

## Quick check: detector evaluation (Table 3)

```bash
python tools/reproduce_table3.py
```

The script runs the project's evaluation code (`src/step2_4_eval_detector.py`) on the released gold labels and
detector outputs and gives the figures of Table 3: test — v3: P 0.917, R 0.904, F1 0.910, candidate P 0.858,
share of correct labels 0.846; v3 without lexicon: 0.858 / 0.896 / 0.877, 0.733, 0.691; dev — v3: 0.985 / 1.000 / 0.992,
0.942, 0.926.

## Annotation format

One row per token. Part 1 tables list every word of the book in order; the word form (`token`, `lemma`)
is given only for tokens that an annotator labelled, other rows keep only their position.

| Column | Meaning |
|---|---|
| `kidlit_id`, `sent_id`, `tok_id` | Token address: book, sentence (0-based), token in sentence (0-based, punctuation counted), as produced by `src/step0_corpus_preprocess.py` (razdel tokenization). |
| `row_id` / `item_id` | Row in the annotator's file (part 1) / candidate ID (part 2, `C001`…). |
| `label` | Part 1: `DIM`, `HYP`, `YOUNG` or empty (= none of them). Part 2: `DIM`, `LEX`, `HYP`, `YOUNG`, `NO`. |
| `base` | Base word (rule P-17: DIM — nearest morphological base; LEX — the word itself; YOUNG — adult animal; HYP — full or short name). |
| `pej`, `name`, `nonce`, `foreign` | Attributes, 0/1: pejorative shade, used as a character name, absent from the pymorphy3 dictionary (DIM, YOUNG for `name`); foreign hypocoristic (HYP). |
| `precedent` | ID of the precedent applied (P-01…P-26). |
| `unsure`, `comment` | Annotator's doubt (0/1) and comment. |

In the adjudication log, `label_A/B` etc. are the annotators' decisions, `final_*` the adjudicated ones,
`type` the reason the case was sent to adjudication (`label`, `base`, `attr`, `unsure`, `consistency`, `gap`,
`rule`), `confirmed = 1` that the decision was checked. `suggestion`, `confidence` and `suggestion_reason`
are pre-filled proposals made from the guidelines before adjudication (AI-assisted); they are not decisions.

## Reproducing the pipeline

You need your own copy of the 120 books in the corpus CSV format described in `src/step0_corpus_preprocess.py`,
the Lyashevskaya–Sharoff frequency dictionary (archive `Freq2011.zip`, http://dict.ruslang.ru/freq.php) and
DerivBase.Ru (`data/wiktionary/v001.00-connections.zip` from https://github.com/s231644/DerivBaseRu).
Both resources are third-party and are not redistributed here.

```bash
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/ru_core_news_sm-3.8.0/ru_core_news_sm-3.8.0-py3-none-any.whl
python src/step0_corpus_preprocess.py --corpus corpus.csv --extra-texts kid_lit_all_ru.txt --freq-zip Freq2011.zip --out kidlit_dim
python src/step0_5_lemma_repair.py --data kidlit_dim
python src/kidlit_dim_detect_v3.py --data kidlit_dim --derivbase v001.00-connections.zip --lexicon data/lexicon/dim_lexicon_v3.1.csv --out-name detector_v3.1
```

With the project's copy of the books this chain reproduces the frozen texts (SHA-256 in `data/metadata/freeze_manifest.json`),
the token table and the released detector output exactly (see `src/README.md`).

The working folder `kidlit_dim/` contains the texts and is ignored by git.

## Licence

Code: MIT ([LICENSE](LICENSE)). Annotations, metadata and tables: CC BY 4.0 ([LICENSE-DATA.md](LICENSE-DATA.md)).
The book texts belong to their rights holders and are not distributed.

## Citation

See [CITATION.cff](CITATION.cff).

---

### Кратко по-русски

Репозиторий к статье «KidLit by Numbers: диминутивы в оригинальных и переводных детских книгах 0+ на
русском языке». Здесь лежат:
- инструкция по разметке v0.5 (цитаты из книг сокращены до четырех слов);
- разметка пилота и KidLit-Gold, журналы адъюдикации;
- метаданные 120 книг и таблицы по леммам;
- детектор диминутивов v3, словари прецедентов, разметка всего корпуса, золотой стандарт;
- результаты анализа и рисунки статьи;
- код всех шагов.

Тексты книг не публикуются: разметка ссылается на слова по координатам `kidlit_id / sent_id / tok_id`.
Согласие разметчиков пересчитывается командой `python tools/compute_iaa.py`, таблица 3 — `python tools/reproduce_table3.py`.
