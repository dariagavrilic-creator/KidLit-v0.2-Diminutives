# Lemma tables

Lemma-level counts, keyword analysis and semantic classes (Figs. 3–4). No running text.

| File | Rows | What it is |
|---|---|---|
| `kidlit_lemma_freq.csv` | 13,413 | Lemma frequencies in KidLit: total, by group, number of books, share of OOV forms. |
| `dim_lemmas_all.csv` | 736 | All diminutive lemmas (detector v3.1) with base, part of speech, frequencies by group, number of books, `name_share`, cumulative share. |
| `dim_lemma_classes.csv` | 204 | Semantic classes of the most frequent diminutive lemmas: first assignment (`sem_class`, `class_source`), `final_class`, `excluded` with `decision_note`. Classes used in Fig. 3. |
| `semantic_classes.csv` | 21 | The class scheme: class, scope, examples. |
| `keywords_kidlit_vs_reference_top100.csv` | 100 | Top-100 keywords of KidLit against the Lyashevskaya–Sharoff frequency dictionary (LL, log ratio) with semantic classes. Source of Fig. 4. |
| `keywords_rus_o_vs_rus_t.csv` | 69 | Keywords of RUS-O against RUS-T, with book-level robustness test and semantic classes. |
| `keyword_classes_second_expert_30.csv` | 30 | Classes assigned by the second expert to 30 keywords (20 from the top-100 table, 10 from RUS-O vs. RUS-T; `table`). |
| `keyness_full/` | 12,642 | Keyness statistics for all lemmas (KidLit vs. reference; RUS-O vs. RUS-T) and the lemma merges applied before keyness. Reference-dictionary frequencies are not included. |

Keyness is computed on lemmas after the repairs in `../metadata/lemma_repairs.csv` (`src/step1_4_keyness.py`).
