# Adjudication

Cases where the annotators disagreed, doubted or were inconsistent, with the final decisions.

| File | Rows | What it is |
|---|---|---|
| `pilot_adjudication.csv` | 7 | Pilot cases: labels and bases of A and B, `final_label`, `final_base`, `precedent`, `decision_comment`. |
| `adjudication_2.3.csv` | 81 | KidLit-Gold cases J001–J081 (step 2.3). |
| `adjudication_2.3.xlsx` | 81 | The same log as a spreadsheet, as used during adjudication. |
| `new_precedents_2.3.csv` | 9 | Proposed precedents (an extension of P-05 and P-19–P-26): case, proposal, reasoning, case rows, decision (yes / no / otherwise). They entered guidelines v0.4–v0.5. |

## Columns of `adjudication_2.3.csv`

| Column | Meaning |
|---|---|
| `adj_id` | Case ID (J001…). |
| `type` | Why the case was sent to adjudication: `label`, `base`, `attr`, `unsure`, `consistency`, `gap`, `rule`. |
| `part`, `id`, `split`, `kidlit_id`, `token`, `lemma` | Where the case is: part 1 / part 2, row or candidate ID, dev / test, book, word form. |
| `label_A/B`, `base_A/B`, `attrs_A/B`, `prec_A/B`, `comment_A/B` | The annotators' decisions. |
| `note` | Why the row was flagged. |
| `suggestion`, `confidence`, `suggestion_reason` | Proposal prepared from the guidelines before adjudication (AI-assisted, without the detector's labels). Not a decision. |
| `final_label`, `final_base`, `final_pej`, `final_name`, `final_nonce`, `final_foreign` | Adjudicated decision. |
| `precedent`, `decision_comment` | Precedent the decision relies on or creates, and the reasoning. |
| `confirmed` | 1 = decision checked by the authors. |

The gold labels are built from the annotator files and this log by `src/step2_3_iaa_adjudication.py` (`apply_adjudication`).
