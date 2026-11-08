# Data statement

## What the corpus is

KidLit v0.2: 120 fiction books for children with the Russian age rating 0+, written and published 2015–2026:
60 original books by Russian authors (RUS-O) and 60 translations into Russian (RUS-T) from English (29),
French (11), German (10), Italian (4), Polish (2), Swedish (2), Dutch (1) and Finnish (1).
Book IDs have the form `<group>_<length>_<NN>` (`RUS-O_long_07`); length strata: short < 500 words,
medium 500–999, long ≥ 1000. `data/metadata/books.csv` gives title, author, translator, year, publisher
and source language for every book, so the books can be obtained from libraries or shops.

## What is released and what is not

| Released | Not released |
|---|---|
| Annotations with token addresses (`kidlit_id`, `sent_id`, `tok_id`) | Book texts, raw or normalized |
| Word forms of annotated tokens and of the 400 detector candidates | Sentences, context windows, neighbouring sentences |
| Bibliographic metadata | Publisher blurbs (`description`) |
| Lemma-level tables (frequencies, keyness, semantic classes) | Token and sentence tables of the corpus (`tokens.parquet`, `sentences.parquet`) |
| Adjudication logs with decisions and comments; sentence-level sentiment labels and short NER mentions (≤ 3 tokens) of KidLit-Gold used in Section 6.4 | Example sentences from the working tables (`ex1`–`ex3`, `context`) |
| Annotation guidelines (quotations ≤ 4 words) | Third-party resources (Freq2011, DerivBase.Ru) and their frequency values |

## How the release was cleaned

1. Every table lost the columns that hold sentences (`context`, `context_more`, `ex1`–`ex3`, `description`, `text`).
2. In the full-reading tables (one row per word), `token` and `lemma` are kept only for tokens labelled by
   at least one annotator or discussed in adjudication; all other rows keep only their position. A sequence
   of positions without word forms does not reproduce the text.
3. In the public version of the guidelines, quotations from the books are shortened to at most four words
   around the target word.
4. `tools/check_no_text.py` compared every text cell of every file with all 120 books and their publisher
   blurbs: no sequence of 5 consecutive words from a book occurs in the repository, except book titles in
   `data/metadata/books.csv` and `data/gold/gold_books.csv` (allow-list `tools/text_check_allowlist.json`). Report: `reports/text_leak_check.md`.

## Annotation

Two annotators (A and B) labelled the data independently and blind to the detector's labels, following the guidelines in `docs/`. The pilot (guidelines v0.2) covered 2 books and
103 candidates; KidLit-Gold (v0.3) covered 6 books in full reading (2 doubly annotated, 2,109 words) and 400
candidates (100 doubly annotated). Disagreements, doubts and inconsistencies (81 cases) were adjudicated
and turned into precedents P-19–P-26 (guidelines v0.4–v0.5). The released annotator files are the
independent submissions before adjudication (for the pilot, later corrected copies are not released);
all decisions live in the adjudication logs and in the pilot gold labels.

Before adjudication, each case got a proposal derived from the guidelines (`suggestion`, `confidence`,
`suggestion_reason`). The final
decisions (`final_*`, `precedent`, `decision_comment`) were made and confirmed by the authors.
