# Table 3 reproduced from the released files

Full reading: precision (P), recall (R) and F1 for DIM over all words of the books, 95% Wilson intervals; candidates: weighted DIM precision and weighted share of correct labels (5 classes).

| Detector | Sample | P | R | F1 | Candidate P | Share correct | Candidates labelled DIM (correct / all) |
|---|---|---|---|---|---|---|---|
| v2 | test | 0.885 (0.821–0.928) | 0.911 (0.851–0.948) | 0.898 | 0.841 | 0.839 | 89 / 111 |
| v2 | dev | 0.935 (0.881–0.965) | 1.000 (0.971–1.000) | 0.966 | 0.843 | 0.888 | 89 / 111 |
| v3 without lexicon | test | 0.858 (0.791–0.906) | 0.896 (0.833–0.937) | 0.877 | 0.733 | 0.691 | 85 / 119 |
| v3 without lexicon | dev | 0.915 (0.857–0.951) | 1.000 (0.971–1.000) | 0.956 | 0.788 | 0.776 | 80 / 106 |
| v3 | test | 0.917 (0.858–0.953) | 0.904 (0.842–0.943) | 0.910 | 0.858 | 0.846 | 87 / 106 |
| v3 | dev | 0.985 (0.946–0.996) | 1.000 (0.971–1.000) | 0.992 | 0.942 | 0.926 | 87 / 95 |
