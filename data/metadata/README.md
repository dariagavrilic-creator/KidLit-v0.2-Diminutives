# Metadata

Bibliographic data and per-book counts for the 120 books of KidLit v0.2. No texts and no publisher blurbs.

| File | Rows | What it is |
|---|---|---|
| `books.csv` | 120 | One row per book: `kidlit_id` (`RUS-O_long_07`), `book_id`, `group` (RUS-O / RUS-T), title, original title, author, translator, illustrator, year, publisher, source language, genre, age rating, pages, UDC, `length_stratum`. |
| `book_stats.csv` | 120 | Denominators for the book-level metrics: characters, sentences, tokens, words, lemmas, nouns (pymorphy3 and spaCy UD), adjectives, adverbs, OOV words, mean sentence length; `words_vs_meta` compares the word count with the catalogue data. |
| `lemma_repairs.csv` | 54 | Lemma pairs corrected after lemmatization (`lemma_pm` → `lemma`, `n` tokens; 260 tokens in total). Produced by `src/step0_5_lemma_repair.py`. |
| `normalization_log.csv` | 103 | Changes made to the raw texts in step 0 (68 books): `kind` (line breaks, soft hyphens, manual fixes…), `before`, `after`, `count`. |
| `freeze_manifest.json` | — | SHA-256 of the source corpus file and of every frozen raw and normalized text. Use it to check that your copy of the books matches ours before reusing token addresses. |

`kidlit_id` is the key used in every other table of the repository.
