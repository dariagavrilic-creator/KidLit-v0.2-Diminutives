#!/usr/bin/env python3
"""
KidLit by Numbers — шаг 3.3: семантические классы диминутивов (по производящему), RUS-O против RUS-T.

    python kidlit_dim_semantics.py

Вход: dim_lemma_classes_3.3_filled.xlsx (разметка 204 лемм, 80% употреблений DIM) + решения 21.10.2026.
Выход: dim_lemma_classes_3.3_final.csv, semantic_classes_by_group.csv, semantic_class_tests.csv, semantics_report.md,
       figures/fig4_semantic_classes.png; вариант чувствительности без исключённых лемм — в sensitivity_excl.csv.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

A = Path("kidlit_dim/analysis")
SEED = 2026
DECISIONS = {"енотик": "животные", "сумочка": "одежда", "пуговка": "одежда", "шляпка": "одежда",
             "шапочка": "одежда", "ботиночек": "одежда", "носочек": "одежда", "курточка": "одежда"}
# Не диминутивы (остаточные ошибки детектора, P-20 тип 0/2) — исключаются из анализа классов (решение 21.10.2026)
EXCLUDE = {"полка", "колонка", "духовка", "палатка", "крышка", "чашка", "нитка", "стенка", "грядка"}


def e(s):
    return s.replace("ё", "е") if isinstance(s, str) else s


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for rank, i in reversed(list(enumerate(o, start=1))):
        prev = min(prev, p[i] * n / rank); q[i] = prev
    return q


def main():
    d = pd.read_excel(A / "dim_lemma_classes_3.3_filled.xlsx", "Разметка", dtype=str)
    d["final_class"] = d.sem_class.fillna("").str.strip()
    for l, c in DECISIONS.items():
        d.loc[d.lemma == l, "final_class"] = c
    d["excluded"] = d.lemma.isin(EXCLUDE)
    d.loc[d.excluded, "final_class"] = ""
    d["decision_note"] = ["исключено: не диминутив (P-20)" if x else ("решение 21.10" if l in DECISIONS else "")
                          for l, x in zip(d.lemma, d.excluded)]
    assert (d[~d.excluded].final_class != "").all(), d[~d.excluded & (d.final_class == "")].lemma.tolist()
    d.to_csv(A / "dim_lemma_classes_3.3_final.csv", index=False)

    c = pd.read_parquet("kidlit_dim/detector_v3.1/candidates.parquet")
    dim = c[c.det_label == "DIM"].copy()
    dim["lem"] = dim.lemma.map(e)
    cls = dict(zip(d.lemma.map(e), d.final_class))
    excl = set(map(e, EXCLUDE))
    dim = dim[~dim.lem.isin(excl)]
    dim["sem_class"] = dim.lem.map(cls).fillna("не размечено")
    n_all = len(dim)
    cover = (dim.sem_class != "не размечено").mean()

    # распределение по группам (словоупотребления)
    lab = dim[dim.sem_class != "не размечено"]
    tab = pd.crosstab(lab.sem_class, lab.group)
    share = tab / tab.sum()
    w = pd.read_csv("kidlit_dim/data/book_stats.csv").groupby("group").n_words.sum()
    per1000 = tab / w * 1000
    out = pd.DataFrame({"n_RUS-O": tab["RUS-O"], "n_RUS-T": tab["RUS-T"], "share_RUS-O": share["RUS-O"],
                        "share_RUS-T": share["RUS-T"], "per1000w_RUS-O": per1000["RUS-O"], "per1000w_RUS-T": per1000["RUS-T"]})
    out["per1000w_diff"] = out["per1000w_RUS-O"] - out["per1000w_RUS-T"]
    out["contribution_to_diff"] = out.per1000w_diff / out.per1000w_diff.sum()
    out = out.sort_values("n_RUS-O", ascending=False)
    lemmas_per_class = lab.groupby("sem_class").lem.nunique()
    out["n_lemmas"] = lemmas_per_class
    out.to_csv(A / "semantic_classes_by_group.csv")
    chi2, p_chi, dof, _ = stats.chi2_contingency(tab.values)
    cramer = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))

    # по книгам: доля класса среди DIM книги; MW + BH (книги с ≥ 5 DIM)
    bk = lab.groupby(["kidlit_id", "group", "sem_class"]).size().unstack(fill_value=0)
    tot = bk.sum(axis=1)
    bk = bk[tot >= 5]
    bshare = bk.div(bk.sum(axis=1), axis=0).reset_index()
    rows = []
    for k in tab.index:
        o = bshare[bshare.group == "RUS-O"][k].values
        t = bshare[bshare.group == "RUS-T"][k].values
        u, p = stats.mannwhitneyu(o, t, alternative="two-sided")
        dlt = ((o[:, None] > t[None, :]).sum() - (o[:, None] < t[None, :]).sum()) / (len(o) * len(t))
        rows.append({"sem_class": k, "books_RUS-O": len(o), "books_RUS-T": len(t), "mean_share_RUS-O": o.mean(),
                     "mean_share_RUS-T": t.mean(), "cliffs_delta": dlt, "p_mw": p})
    bt = pd.DataFrame(rows)
    bt["q_bh"] = bh(bt.p_mw)
    bt = bt.sort_values("p_mw")
    bt.to_csv(A / "semantic_class_tests.csv", index=False)

    # чувствительность главной метрики к исключению 9 лемм
    spec = importlib.util.spec_from_file_location("an", "kidlit_dim_analysis.py")
    an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
    tok = pd.read_parquet("kidlit_dim/data/tokens.parquet", columns=["kidlit_id", "is_word", "pos", "lemma", "lemma_e"])
    tok["lemma_e"] = tok.lemma.map(e)
    books = pd.read_csv("kidlit_dim/data/books.csv"); books["genre"] = books.genre.fillna("проза")
    c2 = c.copy()
    c2.loc[c2.lemma.map(e).isin(excl) & (c2.det_label == "DIM"), "det_label"] = "NO"
    mx = an.book_metrics(tok, c2, books, "base")
    r = an.compare(mx, np.random.default_rng(SEED), {k: an.METRICS[k] for k in ["dim_per_1000w", "dim_noun_per_100n", "dim_choice"]})
    r["variant"] = "excl_nondim_p20"
    r.to_csv(A / "sensitivity_excl.csv", index=False)

    L = ["# Шаг 3.3. Семантические классы диминутивов\n",
         f"Размечено 204 леммы (класс по производящему, 21 класс: 20 из keyness + «одежда»); исключено как не диминутивы: "
         f"{', '.join(sorted(EXCLUDE))}. Покрытие: {cover:.1%} из {n_all} употреблений DIM (после исключения).\n",
         f"χ² (класс × группа, словоупотребления): χ² = {chi2:.1f}, df = {dof}, p = {p_chi:.2g}, V Крамера = {cramer:.2f}. "
         "Словоупотребления внутри книги не независимы, поэтому ниже — проверка на уровне книг.\n",
         "| Класс | RUS-O, % | RUS-T, % | RUS-O на 1000 слов | RUS-T на 1000 слов | вклад в разницу частот | лемм |",
         "|---|---|---|---|---|---|---|"]
    for k, x in out.iterrows():
        L.append(f"| {k} | {100 * x['share_RUS-O']:.1f} | {100 * x['share_RUS-T']:.1f} | {x['per1000w_RUS-O']:.2f} | "
                 f"{x['per1000w_RUS-T']:.2f} | {100 * x.contribution_to_diff:.0f}% | {int(x.n_lemmas)} |")
    L.append("\n## По книгам (доля класса среди DIM книги; книги с ≥ 5 DIM; Mann–Whitney, BH)\n")
    L.append("| Класс | RUS-O | RUS-T | Cliff's δ | p | q |\n|---|---|---|---|---|---|")
    for x in bt.itertuples():
        L.append(f"| {x.sem_class} | {x._4:.3f} | {x._5:.3f} | {x.cliffs_delta:.2f} | {x.p_mw:.3g} | {x.q_bh:.3g} |")
    L.append("\n## Чувствительность главной метрики к исключению 9 лемм\n")
    for x in r.itertuples():
        L.append(f"- {x.metric}: медианы {x.median_rus_o:.2f} / {x.median_rus_t:.2f}, δ = {x.cliffs_delta:.2f} "
                 f"[{x.delta_ci_lo:.2f}; {x.delta_ci_hi:.2f}], q = {x.q_bh:.3g}")
    (A / "semantics_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
