#!/usr/bin/env python3
"""KidLit by Numbers — шаг 3.6: results_master.csv. Каждое число статьи берётся отсюда; у каждого — файл-источник.

    python kidlit_dim_results_master.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = []


def add(section, key, value, ci="", n="", source="", note=""):
    if isinstance(value, (float, np.floating)):
        value = round(float(value), 4)
    R.append({"section": section, "key": key, "value": value, "ci95": ci, "n": n, "source": source, "note": note})


def f(x, d=3):
    return f"{x:.{d}f}"


def main():
    # корпус
    b = pd.read_csv("kidlit_dim/data/books.csv")
    st = pd.read_csv("kidlit_dim/data/book_stats.csv")
    add("corpus", "books_total", len(b), source="data/books.csv")
    for g, x in st.groupby("group"):
        add("corpus", f"words_{g}", int(x.n_words.sum()), n=len(x), source="data/book_stats.csv")
        add("corpus", f"median_words_per_book_{g}", float(x.n_words.median()), source="data/book_stats.csv")
    add("corpus", "words_total", int(st.n_words.sum()), source="data/book_stats.csv")

    # IAA диминутивов (шаг 2.3)
    add("iaa_dim", "full_reading_F1_DIM", 0.904, n="2 books, 2109 words", source="gold/iaa_report_2.3.md")
    add("iaa_dim", "full_reading_kappa_DIM_vs_not", 0.899, source="gold/iaa_report_2.3.md")
    add("iaa_dim", "candidates_kappa_5labels", 0.736, n="100 candidates", source="gold/iaa_report_2.3.md")
    add("iaa_dim", "candidates_kappa_DIM_vs_not", 0.760, n="100 candidates", source="gold/iaa_report_2.3.md")
    add("iaa_dim", "candidates_raw_agreement", 0.830, n="100 candidates", source="gold/iaa_report_2.3.md")
    adj = pd.read_excel("kidlit_dim/gold/adjudication_2.3.xlsx", "Адъюдикация")
    add("iaa_dim", "adjudicated_cases", len(adj), source="gold/adjudication_2.3.xlsx")
    add("iaa_dim", "new_precedents_v0.4", 8, note="P-19–P-26", source="Инструкция v0.4")

    # детектор (шаг 2.4): test, единственный прогон
    for name, p in [("v3", "kidlit_dim/detector_v3"), ("v2", "kidlit_dim/baselines/v2"), ("v3_nolex", "kidlit_dim/baselines/v3_nolex")]:
        s = json.loads(Path(p, "eval_dev_test.json").read_text())
        for sp in ["dev", "test"]:
            d = s[sp]["part1"]["DIM"]
            add("detector", f"{name}_{sp}_full_reading_DIM_P", d["P"], n=f"{d['tp']}/{d['det']}", source=f"{p}/eval_dev_test.json")
            add("detector", f"{name}_{sp}_full_reading_DIM_R", d["R"], n=f"{d['tp']}/{d['gold']}", source=f"{p}/eval_dev_test.json")
            add("detector", f"{name}_{sp}_full_reading_DIM_F1", d["F1"], source=f"{p}/eval_dev_test.json")
            add("detector", f"{name}_{sp}_candidates_DIM_P_weighted", s[sp]["part2"]["P_dim_weighted"], source=f"{p}/eval_dev_test.json")
            add("detector", f"{name}_{sp}_candidates_acc_weighted", s[sp]["part2"]["acc_weighted"], source=f"{p}/eval_dev_test.json")
    add("detector", "v3_test_full_reading_DIM_P_ci", "", ci="0.858–0.953", source="detector_v3/eval_dev_test.md")
    add("detector", "v3_test_full_reading_DIM_R_ci", "", ci="0.842–0.943", source="detector_v3/eval_dev_test.md")
    s = json.loads(Path("kidlit_dim/detector_v3/eval_dev_test.json").read_text())
    for lab in ["YOUNG", "HYP"]:
        d = s["test"]["part1"][lab]
        add("detector", f"v3_test_full_reading_{lab}_P", d["P"], n=f"{d['tp']}/{d['det']}", source="detector_v3/eval_dev_test.json")
        add("detector", f"v3_test_full_reading_{lab}_R", d["R"], n=f"{d['tp']}/{d['gold']}", source="detector_v3/eval_dev_test.json")
    # по группам на test
    spec = importlib.util.spec_from_file_location("ev", "kidlit_dim_eval.py")
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    p1, _ = ev.load_gold(Path("kidlit_dim/gold"), "pilot_orig/pilot_gold.csv")
    cand = pd.read_parquet("kidlit_dim/detector_v3/candidates.parquet").set_index(ev.KEY)
    g = p1[p1.split == "test"]
    for grp in ["RUS-O", "RUS-T"]:
        P, Rr = ev._pr(cand, g, grp)
        add("detector", f"v3_test_full_reading_DIM_P_{grp}", P, source="kidlit_dim_eval.py (_pr)")
        add("detector", f"v3_test_full_reading_DIM_R_{grp}", Rr, source="kidlit_dim_eval.py (_pr)")
    add("detector", "test_run_record", json.loads(Path("kidlit_dim/gold/TEST_RUN.json").read_text())["time"], source="gold/TEST_RUN.json")

    # корпус по v3.1
    c = pd.read_parquet("kidlit_dim/detector_v3.1/candidates.parquet")
    w = st.groupby("group").n_words.sum()
    for lab in ["DIM", "HYP", "YOUNG", "LEX"]:
        for grp in ["RUS-O", "RUS-T"]:
            k = int(((c.det_label == lab) & (c.group == grp)).sum())
            add("corpus_v3.1", f"{lab}_tokens_{grp}", k, source="detector_v3.1/candidates.parquet")
            add("corpus_v3.1", f"{lab}_per_1000w_pooled_{grp}", 1000 * k / w[grp], source="detector_v3.1/candidates.parquet")
    add("corpus_v3.1", "DIM_lemmas", int(c[c.det_label == "DIM"].lemma.nunique()), source="detector_v3.1/candidates.parquet")
    add("corpus_v3.1", "context_needs_review", int(c.needs_review.sum()), source="detector_v3.1/candidates.parquet")

    # 3.2
    t = pd.read_csv("kidlit_dim/analysis/group_tests.csv")
    for r in t.itertuples():
        src = "analysis/group_tests.csv"
        add("group_tests", f"{r.metric}_median_RUS-O", r.median_rus_o, n=r.n_rus_o, source=src)
        add("group_tests", f"{r.metric}_median_RUS-T", r.median_rus_t, n=r.n_rus_t, source=src)
        add("group_tests", f"{r.metric}_median_diff", r.median_diff, ci=r.median_diff_ci, source=src)
        add("group_tests", f"{r.metric}_cliffs_delta", r.cliffs_delta, ci=f"{f(r.delta_ci_lo)}–{f(r.delta_ci_hi)}", source=src)
        add("group_tests", f"{r.metric}_p_mw", r.p_mw, source=src)
        add("group_tests", f"{r.metric}_q_bh", r.q_bh, source=src, note=f"BH по {len(t)} метрикам")
        add("group_tests", f"{r.metric}_spearman_loglen", r.spearman_loglen_rho, source=src)
        add("group_tests", f"{r.metric}_ols_group_coef", r.ols_group_coef, ci=r.ols_group_ci, source=src,
            note="OLS (HC3): metric ~ group + log(words) + genre; RUS-O относительно RUS-T")
        add("group_tests", f"{r.metric}_ols_group_q_bh", r.q_bh_ols_group, source=src)
    s = pd.read_csv("kidlit_dim/analysis/sensitivity.csv")
    for r in s.itertuples():
        add("sensitivity", f"{r.variant}_{r.metric}_cliffs_delta", r.cliffs_delta, ci=f"{f(r.delta_ci_lo)}–{f(r.delta_ci_hi)}",
            source="analysis/sensitivity.csv", note=f"медианы {r.median_rus_o:.2f} / {r.median_rus_t:.2f}; q={r.q_bh:.4f}; OLS p={r.ols_group_p:.4f}")
    lg = pd.read_csv("kidlit_dim/analysis/rus_t_by_language.csv")
    for r in lg.itertuples():
        add("rus_t_language", f"{r.lang_group}_median_dim_per_1000w", r.median_dim_per_1000w, n=r.n_books,
            source="analysis/rus_t_by_language.csv", note="описательно")

    # 3.3 (keyness-классы)
    add("semantics", "keyness_classes_kappa", 0.634, n="30 lemmas", source="keyness/filled/keyness_classes_kappa30_expert2.xlsx",
        note="20 из 30 совпали; до обсуждения расхождений")
    k = pd.read_excel("kidlit_dim/keyness/filled/kidlit_vs_ref_top100.xlsx", "Разметка")
    for cl, v in k.sem_class.value_counts().items():
        add("semantics", f"top100_keywords_class_{cl}", int(v), source="keyness/filled/kidlit_vs_ref_top100.xlsx")
    sc = pd.read_csv("kidlit_dim/analysis/semantic_classes_by_group.csv", index_col=0)
    for cl, x in sc.iterrows():
        for g in ["RUS-O", "RUS-T"]:
            add("semantics", f"dim_class_{cl}_share_{g}", x[f"share_{g}"], n=int(x[f"n_{g}"]), source="analysis/semantic_classes_by_group.csv")
            add("semantics", f"dim_class_{cl}_per1000w_{g}", x[f"per1000w_{g}"], source="analysis/semantic_classes_by_group.csv")
        add("semantics", f"dim_class_{cl}_contribution_to_gap", x.contribution_to_diff, source="analysis/semantic_classes_by_group.csv")
    stt = pd.read_csv("kidlit_dim/analysis/semantic_class_tests.csv")
    for x in stt.itertuples():
        add("semantics", f"dim_class_{x.sem_class}_book_level_delta", x.cliffs_delta, source="analysis/semantic_class_tests.csv",
            note=f"p={x.p_mw:.3g}, q={x.q_bh:.3g}")
    add("semantics", "dim_class_coverage", 0.795, n="204 lemmas, 9 excluded", source="analysis/semantics_report.md")
    add("semantics", "dim_class_chi2_token_level_cramersV", 0.27, note="χ²=313.1, df=16; словоупотребления не независимы",
        source="analysis/semantics_report.md")
    se = pd.read_csv("kidlit_dim/analysis/sensitivity_excl.csv")
    for x in se.itertuples():
        add("sensitivity", f"excl_nondim_p20_{x.metric}_cliffs_delta", x.cliffs_delta, ci=f"{f(x.delta_ci_lo)}–{f(x.delta_ci_hi)}",
            source="analysis/sensitivity_excl.csv", note=f"медианы {x.median_rus_o:.2f} / {x.median_rus_t:.2f}; q={x.q_bh:.4f}")
    add("sensitivity", "no_name_dim_per_1000w_cliffs_delta", 0.309, note="без DIM с атрибутом name; медианы 24.1 / 14.8; MW p=0.0035",
        source="расчёт 21.10 (candidates v3.1, name == 0)")
    add("keyness", "kidlit_keywords_positive", 1098, source="keyness/keyness_report.md")
    add("keyness", "kidlit_keywords_negative", 140, source="keyness/keyness_report.md")
    add("keyness", "rus_o_keywords_ll", 24, note="12 устойчивы на уровне книг", source="keyness/keyness_report.md")
    add("keyness", "rus_t_keywords_ll", 45, note="0 устойчивы на уровне книг", source="keyness/keyness_report.md")

    # 3.4
    pl = pd.read_csv("kidlit_dim/analysis/pleonasm.csv")
    for r in pl.itertuples():
        tag = "size" if r.adjectives.startswith("размер") else "big"
        add("collocations", f"{tag}_{r.group}_per100_dim", r.dim_per_100, n=f"{r.dim_k}/{r.dim_n}", source="analysis/pleonasm.csv")
        add("collocations", f"{tag}_{r.group}_per100_base", r.base_per_100, n=f"{r.base_k}/{r.base_n}", source="analysis/pleonasm.csv")
        add("collocations", f"{tag}_{r.group}_per100_other", r.other_per_100, n=f"{r.other_k}/{r.other_n}", source="analysis/pleonasm.csv")
        add("collocations", f"{tag}_{r.group}_OR_dim_vs_base", r.OR_dim_vs_base, source="analysis/pleonasm.csv", note=f"Fisher p={r.p_fisher:.3g}")
    add("collocations", "size_dim_minus_base_boot_ci_all", "", ci=pl.boot_ci_dim_minus_base_size_all.iloc[0], source="analysis/pleonasm.csv")
    ad = pd.read_csv("kidlit_dim/analysis/address.csv")
    for r in ad.itertuples():
        add("collocations", f"address_share_{r.group}", r.address_share, n=f"{r.address}/{r.n_dim}", source="analysis/address.csv",
            note="эвристика; 24–26 из 30 срабатываний верны")
    add("collocations", "address_OR_O_vs_T", ad.OR_O_vs_T.iloc[0], source="analysis/address.csv", note=f"Fisher p={ad.p_fisher.iloc[0]:.3g}")

    # 3.5 (частично)
    si = pd.read_csv("kidlit_dim/analysis/sentiment_iaa_dim.csv").iloc[0]
    add("gold_link", "sentiment_disagreement_rate_dim_sent", si.rate_dim, n=int(si.sent_dim), source="analysis/sentiment_iaa_dim.csv",
        note="ЗАМЕНЕНО: см. gold30 (κ по двойной разметке); старая оценка по логу")
    add("gold_link", "sentiment_disagreement_rate_nodim_sent", si.rate_nodim, n=int(si.sent_nodim), source="analysis/sentiment_iaa_dim.csv")
    add("gold_link", "sentiment_disagreement_OR", si.OR, source="analysis/sentiment_iaa_dim.csv", note=f"Fisher p={si.p_fisher:.3g}; Wilcoxon по книгам p={si.wilcoxon_books_p:.3g}")

    sd = pd.read_csv("kidlit_dim/analysis/sentiment_dim.csv")
    for x in sd.itertuples():
        for lab in ["pos", "neu", "neg"]:
            add("gold_link", f"sentiment_{lab}_share_dim_{x.group}", getattr(x, f"{lab}_share_dim"), n=x.with_dim,
                source="analysis/sentiment_dim.csv", note="Label Studio, одна разметчица")
            add("gold_link", f"sentiment_{lab}_share_nodim_{x.group}", getattr(x, f"{lab}_share_nodim"), n=x.sentences - x.with_dim,
                source="analysis/sentiment_dim.csv")
        add("gold_link", f"sentiment_OR_pos_{x.group}", x.OR_pos, source="analysis/sentiment_dim.csv", note=f"Fisher p={x.p_pos:.3g}")
        add("gold_link", f"sentiment_OR_neg_{x.group}", x.OR_neg, source="analysis/sentiment_dim.csv", note=f"Fisher p={x.p_neg:.3g}")
    x = sd.iloc[0]
    add("gold_link", "sentiment_books", 32, note="RUS-O 15, RUS-T 17", source="analysis/goldlink_report.md")
    add("gold_link", "sentiment_logit_pos_has_dim_OR", x.logit_has_dim_OR, ci=x.logit_has_dim_OR_ci, source="analysis/sentiment_dim.csv",
        note=f"кластеры по книгам; p={x.logit_has_dim_p:.3g}")
    add("gold_link", "sentiment_logit_neg_has_dim_OR", x.logit_neg_has_dim_OR, ci=x.logit_neg_has_dim_OR_ci, source="analysis/sentiment_dim.csv",
        note=f"кластеры по книгам; p={x.logit_neg_has_dim_p:.3g}")
    add("gold_link", "sentiment_boot_pos_diff", "", ci=x.boot_ci_pos_diff_all, source="analysis/sentiment_dim.csv")
    nd = pd.read_csv("kidlit_dim/analysis/ner_dim.csv")
    for _, x in nd.iterrows():
        for g in ["RUS-O", "RUS-T"]:
            add("gold_link", f"ner_{x['label']}_dim_share_{g}", x[f"dim_share_{g}"], n=int(x[f"mentions_{g}"]), source="analysis/ner_dim.csv")
            add("gold_link", f"ner_{x['label']}_book_median_{g}", x[f"book_median_{g}"], source="analysis/ner_dim.csv")
        add("gold_link", f"ner_{x['label']}_OR_O_vs_T", x["OR_O_vs_T"], source="analysis/ner_dim.csv",
            note=f"Fisher p={x['p_fisher']:.3g}; MW по книгам p={x['p_mw_books']:.3g}")
    # 3.5 на двойной разметке KidLit-Gold (A и B, 30 книг) — основной источник для статьи
    ia = pd.read_csv("kidlit_dim/analysis/gold30_sentiment_iaa.csv")
    for x in ia.itertuples():
        add("gold30", f"sentiment_kappa_all_{x.group}", x.kappa_all, n=x.sentences, source="analysis/gold30_sentiment_iaa.csv")
        add("gold30", f"sentiment_kappa_dim_{x.group}", x.kappa_dim, n=x.with_dim, source="analysis/gold30_sentiment_iaa.csv")
        add("gold30", f"sentiment_kappa_nodim_{x.group}", x.kappa_nodim, n=x.sentences - x.with_dim, source="analysis/gold30_sentiment_iaa.csv")
    add("gold30", "sentiment_kappa_dim_minus_nodim_boot", "", ci=ia.boot_ci_kappa_dim_minus_nodim.iloc[0], source="analysis/gold30_sentiment_iaa.csv")
    sd30 = pd.read_csv("kidlit_dim/analysis/gold30_sentiment_dim.csv")
    for x in sd30.itertuples():
        for lab in ["pos", "neu", "neg"]:
            add("gold30", f"sentiment_{lab}_dim_{x.source}", getattr(x, f"{lab}_dim"), n=x.with_dim, source="analysis/gold30_sentiment_dim.csv")
            add("gold30", f"sentiment_{lab}_nodim_{x.source}", getattr(x, f"{lab}_nodim"), n=x.sentences - x.with_dim, source="analysis/gold30_sentiment_dim.csv")
        add("gold30", f"sentiment_OR_pos_{x.source}", x.OR_pos, ci=x.OR_pos_ci, source="analysis/gold30_sentiment_dim.csv", note=f"p={x.p_pos:.3g}; кластеры по книгам")
        add("gold30", f"sentiment_OR_neg_{x.source}", x.OR_neg, ci=x.OR_neg_ci, source="analysis/gold30_sentiment_dim.csv", note=f"p={x.p_neg:.3g}; кластеры по книгам")
    cv = pd.read_csv("kidlit_dim/analysis/gold30_log_coverage.csv")
    add("gold30", "log_resolved_share_10books", cv.resolved.sum() / cv.disagreements.sum(), n=f"{cv.resolved.sum()}/{cv.disagreements.sum()}",
        source="analysis/gold30_log_coverage.csv")
    nd30 = pd.read_csv("kidlit_dim/analysis/gold30_ner_dim.csv")
    add("gold30", "ner_F1_AB_all", nd30.ner_F1_all.iloc[0], source="analysis/gold30_ner_dim.csv", note="точные границы + метка")
    add("gold30", "ner_F1_AB_dim", nd30.ner_F1_dim.iloc[0], source="analysis/gold30_ner_dim.csv")
    add("gold30", "ner_F1_AB_nodim", nd30.ner_F1_nodim.iloc[0], source="analysis/gold30_ner_dim.csv")
    for _, x in nd30.iterrows():
        for g in ["RUS-O", "RUS-T"]:
            add("gold30", f"ner_{x['annotator']}_{x['label']}_dim_share_{g}", x[f"dim_share_{g}"], n=int(x[f"n_{g}"]), source="analysis/gold30_ner_dim.csv")
        add("gold30", f"ner_{x['annotator']}_{x['label']}_OR_O_vs_T", x["OR"], source="analysis/gold30_ner_dim.csv",
            note=f"Fisher p={x['p_fisher']:.3g}; MW по книгам p={x['p_mw_books']:.3g}")
    out = pd.DataFrame(R)
    out.insert(0, "id", [f"R{i:03d}" for i in range(1, len(out) + 1)])
    out.to_csv("kidlit_dim/analysis/results_master.csv", index=False)
    print(len(out), "строк →", "kidlit_dim/analysis/results_master.csv")


if __name__ == "__main__":
    main()
