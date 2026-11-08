#!/usr/bin/env python3
"""
KidLit by Numbers — шаг 3.5: диминутивы и разметка сентимента / NER.

    python kidlit_dim_goldlink.py

Вход: data/labelstudio/project1.json, project2.json (одна разметчица; сентимент — 32 книги, NER — 60 книг).
Смещения переносятся на тексты корпуса по совпадению непробельных символов (тексты совпадают с точностью до пробелов).
Выход (analysis/): ls_alignment_check.csv, sentiment_by_sentence.csv, sentiment_dim.csv, ner_mentions.csv, ner_dim.csv,
                   goldlink_report.md
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

A = Path("kidlit_dim/analysis")
D = Path("kidlit_dim/data")
SEED = 2026
SENT = {"+1 POSITIVE": 1, "0 NEUTRAL": 0, "-1 NEGATIVE": -1}
NER = {"PER_NAME", "PER_ROLE", "PER_FAMILY"}


def nonws_index(s):
    return [i for i, ch in enumerate(s) if not ch.isspace()]


def build_map(task_text, our_text):
    """Соответствие непробельных символов текста задачи и текста корпуса. Если последовательности совпадают —
    один к одному; иначе (мягкие переносы, обрезанный конец, служебные вставки) — по совпадающим блокам difflib."""
    import difflib
    a, b = nonws_index(task_text), nonws_index(our_text)
    m = np.full(len(task_text) + 1, -1)
    if len(a) == len(b) and all(task_text[i] == our_text[j] for i, j in zip(a, b)):
        for i, j in zip(a, b):
            m[i] = j
        return m, a
    sa = "".join(task_text[i] for i in a); sb = "".join(our_text[j] for j in b)
    sm = difflib.SequenceMatcher(None, sa, sb, autojunk=False)
    keep = []
    for blk in sm.get_matching_blocks():
        for t in range(blk.size):
            m[a[blk.a + t]] = b[blk.b + t]
            keep.append(a[blk.a + t])
    return m, sorted(keep)


def map_span(m, a_idx, start, end):
    """[start, end) в тексте задачи → [s, e) в тексте корпуса по первому и последнему непробельному символу."""
    import bisect
    i = bisect.bisect_left(a_idx, start)
    k = bisect.bisect_left(a_idx, end) - 1
    if i >= len(a_idx) or k < i:
        return None
    return int(m[a_idx[i]]), int(m[a_idx[k]]) + 1


def main():
    books = pd.read_csv(D / "books.csv")
    tmap = {t.strip().lower(): (k, bid, g) for t, k, bid, g in zip(books.title, books.kidlit_id, books.book_id, books.group)}
    gold_ids = set(books[books.id_source == "задан в таблице проекта"].kidlit_id)
    tok = pd.read_parquet(D / "tokens.parquet", columns=["kidlit_id", "sent_id", "tok_id", "start", "end", "is_word", "token"])
    sents = pd.read_parquet(D / "sentences.parquet")
    cand = pd.read_parquet("kidlit_dim/detector_v3.1/candidates.parquet")
    dimk = cand[cand.det_label == "DIM"][["kidlit_id", "sent_id", "tok_id", "lemma", "name", "pos"]]
    tok = tok.merge(dimk.assign(is_dim=True), on=["kidlit_id", "sent_id", "tok_id"], how="left")
    tok["is_dim"] = tok.is_dim.fillna(False).astype(bool)

    align, sent_rows, ner_rows = [], [], []
    for proj in ["project1", "project2"]:
        for x in json.loads((D / "labelstudio" / f"{proj}.json").read_text()):
            dd = x["data"]
            k, bid, grp = tmap[dd["title"].strip().lower()]
            ours = (D / "texts" / f"{bid}.txt").read_text(encoding="utf-8")
            mp = build_map(dd["text"], ours)
            n_s = n_n = n_fail = 0
            for ann in x["annotations"]:
                for r in ann["result"]:
                    v = r["value"]; lab = v["labels"][0]
                    if mp is None:
                        n_fail += 1; continue
                    sp = map_span(mp[0], mp[1], v["start"], v["end"])
                    if sp is None:
                        n_fail += 1; continue
                    if lab in SENT:
                        n_s += 1
                        sent_rows.append({"kidlit_id": k, "group": grp, "project": proj, "s": sp[0], "e": sp[1], "label": SENT[lab]})
                    elif lab in NER:
                        n_n += 1
                        ner_rows.append({"kidlit_id": k, "group": grp, "project": proj, "s": sp[0], "e": sp[1], "label": lab, "text": v["text"]})
            align.append({"kidlit_id": k, "group": grp, "project": proj, "uid": dd["unique_id"], "aligned": mp is not None,
                          "sentiment_spans": n_s, "ner_spans": n_n, "failed": n_fail, "in_gold": k in gold_ids})
    al = pd.DataFrame(align)
    al.to_csv(A / "ls_alignment_check.csv", index=False)

    # --- сентимент: метка предложения корпуса = метка спана с наибольшим перекрытием
    sp = pd.DataFrame(sent_rows)
    S = sents[sents.kidlit_id.isin(sp.kidlit_id.unique())].copy()
    labs = []
    for b, g in S.groupby("kidlit_id"):
        spans = sp[sp.kidlit_id == b].sort_values("s")
        ss, ee, ll = spans.s.values, spans.e.values, spans.label.values
        for r in g.itertuples():
            ov = np.minimum(ee, r.end) - np.maximum(ss, r.start)
            j = np.argmax(ov) if len(ov) else -1
            labs.append((r.kidlit_id, r.sent_id, int(ll[j]) if j >= 0 and ov[j] > 0 else np.nan,
                         int((ov > 0).sum()) if len(ov) else 0))
    L_ = pd.DataFrame(labs, columns=["kidlit_id", "sent_id", "sentiment", "n_spans"])
    S = S.merge(L_, on=["kidlit_id", "sent_id"])
    ds = tok[tok.is_dim].groupby(["kidlit_id", "sent_id"]).agg(n_dim=("token", "size"), dim_name=("name", "max")).reset_index()
    S = S.merge(ds, on=["kidlit_id", "sent_id"], how="left")
    S["n_dim"] = S.n_dim.fillna(0).astype(int); S["has_dim"] = S.n_dim > 0
    S["dim_name"] = S.dim_name.fillna(0)
    S["has_dim_noname"] = S.has_dim & (S.dim_name == 0)   # грубо: в предложении есть DIM, но не имя
    S = S[S.sentiment.notna()].copy()
    S["pos"] = (S.sentiment == 1).astype(int); S["neg"] = (S.sentiment == -1).astype(int)
    S[["kidlit_id", "group", "sent_id", "sentiment", "n_spans", "n_dim", "has_dim"]].to_csv(A / "sentiment_by_sentence.csv", index=False)

    rows = []
    for grp in ["все", "RUS-O", "RUS-T"]:
        x = S if grp == "все" else S[S.group == grp]
        ct = pd.crosstab(x.has_dim, x.sentiment)
        share_dim = (x[x.has_dim].sentiment.value_counts(normalize=True)).to_dict()
        share_no = (x[~x.has_dim].sentiment.value_counts(normalize=True)).to_dict()
        orr, p = stats.fisher_exact(pd.crosstab(x.has_dim, x.pos).values)
        orn, pn = stats.fisher_exact(pd.crosstab(x.has_dim, x.neg).values)
        chi2, pc, _, _ = stats.chi2_contingency(ct.values)
        rows.append({"group": grp, "books": x.kidlit_id.nunique(), "sentences": len(x), "with_dim": int(x.has_dim.sum()),
                     "pos_share_dim": share_dim.get(1, 0), "pos_share_nodim": share_no.get(1, 0),
                     "neu_share_dim": share_dim.get(0, 0), "neu_share_nodim": share_no.get(0, 0),
                     "neg_share_dim": share_dim.get(-1, 0), "neg_share_nodim": share_no.get(-1, 0),
                     "OR_pos": orr, "p_pos": p, "OR_neg": orn, "p_neg": pn, "chi2_3x2_p": pc})
    sres = pd.DataFrame(rows)
    # bootstrap по книгам: разница доли +1 (с DIM − без DIM)
    rng = np.random.default_rng(SEED)
    bk = S.groupby(["kidlit_id", "has_dim"]).pos.agg(["sum", "size"]).unstack(fill_value=0)
    arr = bk.values  # [sum_F, sum_T, size_F, size_T]
    cols = list(bk.columns)
    iF, iT = cols.index(("sum", False)), cols.index(("sum", True))
    nF, nT = cols.index(("size", False)), cols.index(("size", True))
    diffs = []
    for _ in range(5000):
        z = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
        diffs.append(z[iT] / z[nT] - z[iF] / z[nF])
    sres["boot_ci_pos_diff_all"] = f"{np.percentile(diffs, 2.5):.3f}…{np.percentile(diffs, 97.5):.3f}"
    # логистическая регрессия с кластерными ошибками по книгам
    fit = smf.logit("pos ~ has_dim + C(group) + np.log(n_tokens)", data=S.assign(has_dim=S.has_dim.astype(int))).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(S.kidlit_id)[0]})
    sres["logit_has_dim_OR"] = float(np.exp(fit.params["has_dim"]))
    ci = np.exp(fit.conf_int().loc["has_dim"]).values
    sres["logit_has_dim_OR_ci"] = f"{ci[0]:.2f}…{ci[1]:.2f}"
    sres["logit_has_dim_p"] = float(fit.pvalues["has_dim"])
    fitn = smf.logit("neg ~ has_dim + C(group) + np.log(n_tokens)", data=S.assign(has_dim=S.has_dim.astype(int))).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(S.kidlit_id)[0]})
    cin = np.exp(fitn.conf_int().loc["has_dim"]).values
    sres["logit_neg_has_dim_OR"] = float(np.exp(fitn.params["has_dim"]))
    sres["logit_neg_has_dim_OR_ci"] = f"{cin[0]:.2f}…{cin[1]:.2f}"
    sres["logit_neg_has_dim_p"] = float(fitn.pvalues["has_dim"])
    sres.to_csv(A / "sentiment_dim.csv", index=False)

    # --- NER: упоминание диминутивное, если любой его токен — DIM
    nr = pd.DataFrame(ner_rows)
    tw = tok[tok.is_word]
    out = []
    for b, g in nr.groupby("kidlit_id"):
        t = tw[tw.kidlit_id == b]
        ts, te, td = t.start.values, t.end.values, t.is_dim.values
        tl = t.lemma.values
        for r in g.itertuples():
            m = (ts < r.e) & (te > r.s)
            out.append({**r._asdict(), "n_tokens": int(m.sum()), "is_dim": bool(td[m].any()),
                        "dim_lemmas": ";".join(sorted(set(str(v) for v in tl[m & td])))})
    nm = pd.DataFrame(out).drop(columns=["Index"])
    nm.to_csv(A / "ner_mentions.csv", index=False)
    nrow = []
    for lab in ["PER_NAME", "PER_ROLE", "PER_FAMILY", "все"]:
        x = nm if lab == "все" else nm[nm.label == lab]
        o, t = x[x.group == "RUS-O"], x[x.group == "RUS-T"]
        orr, p = stats.fisher_exact([[o.is_dim.sum(), (~o.is_dim).sum()], [t.is_dim.sum(), (~t.is_dim).sum()]])
        # по книгам
        bb = x.groupby(["kidlit_id", "group"]).is_dim.mean().reset_index()
        bo, bt = bb[bb.group == "RUS-O"].is_dim, bb[bb.group == "RUS-T"].is_dim
        pb = stats.mannwhitneyu(bo, bt).pvalue if len(bo) > 2 and len(bt) > 2 else np.nan
        nrow.append({"label": lab, "mentions_RUS-O": len(o), "mentions_RUS-T": len(t), "dim_share_RUS-O": o.is_dim.mean(),
                     "dim_share_RUS-T": t.is_dim.mean(), "OR_O_vs_T": orr, "p_fisher": p,
                     "book_median_RUS-O": bo.median(), "book_median_RUS-T": bt.median(), "p_mw_books": pb,
                     "top_dim_RUS-O": ", ".join(o[o.is_dim].dim_lemmas.value_counts().head(6).index),
                     "top_dim_RUS-T": ", ".join(t[t.is_dim].dim_lemmas.value_counts().head(6).index)})
    nres = pd.DataFrame(nrow)
    dimm = nm[nm.is_dim]
    nres["share_of_dim_mentions_RUS-O"] = [dimm[(dimm.group == "RUS-O") & (dimm.label == l)].shape[0] / max((dimm.group == "RUS-O").sum(), 1) if l != "все" else 1 for l in nres.label]
    nres["share_of_dim_mentions_RUS-T"] = [dimm[(dimm.group == "RUS-T") & (dimm.label == l)].shape[0] / max((dimm.group == "RUS-T").sum(), 1) if l != "все" else 1 for l in nres.label]
    nres.to_csv(A / "ner_dim.csv", index=False)
    # доля DIM-токенов (существительных), попавших в какое-либо упоминание персонажа
    dim_nouns = tw[tw.is_dim & tw.kidlit_id.isin(nm.kidlit_id.unique())]
    covered = 0
    for b, g in dim_nouns.groupby("kidlit_id"):
        x = nm[nm.kidlit_id == b]
        for s0, e0 in zip(g.start, g.end):
            covered += bool(((x.s < e0) & (x.e > s0)).any())
    dim_in_ner = covered / len(dim_nouns)

    L = ["# Шаг 3.5. Диминутивы и разметка сентимента / NER\n",
         "Данные: экспорт Label Studio (project-1, project-2). Разметка одной разметчицы; итоговых решений адъюдикации "
         "в экспорте нет, второго разметчика нет — согласие по этим файлам не считается.\n",
         f"Перенос смещений: {int(al.aligned.sum())} из {len(al)} книг; не перенесено спанов: {int(al.failed.sum())}.\n",
         "## Сентимент\n",
         f"Книг с разметкой сентимента: {S.kidlit_id.nunique()} (RUS-O {S[S.group == 'RUS-O'].kidlit_id.nunique()}, "
         f"RUS-T {S[S.group == 'RUS-T'].kidlit_id.nunique()}); предложений с меткой: {len(S)}, из них с DIM: {int(S.has_dim.sum())}.\n",
         "| Группа | Предложений (с DIM) | +1: с DIM / без | 0: с DIM / без | −1: с DIM / без | OR(+1) | p | OR(−1) | p |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in sres.itertuples():
        L.append(f"| {r.group} | {r.sentences} ({r.with_dim}) | {r.pos_share_dim:.2f} / {r.pos_share_nodim:.2f} | "
                 f"{r.neu_share_dim:.2f} / {r.neu_share_nodim:.2f} | {r.neg_share_dim:.2f} / {r.neg_share_nodim:.2f} | "
                 f"{r.OR_pos:.2f} | {r.p_pos:.3g} | {r.OR_neg:.2f} | {r.p_neg:.3g} |")
    L.append(f"\nBootstrap по книгам (5000), разница доли +1 «с DIM − без DIM»: {sres.boot_ci_pos_diff_all.iloc[0]}.")
    L.append(f"Логистическая регрессия «+1 ~ есть DIM + группа + log(длина предложения)», ошибки кластеризованы по книгам: "
             f"OR(есть DIM) = {sres.logit_has_dim_OR.iloc[0]:.2f} [{sres.logit_has_dim_OR_ci.iloc[0]}], p = {sres.logit_has_dim_p.iloc[0]:.3g}.\n")
    L.append(f"То же для −1: OR(есть DIM) = {sres.logit_neg_has_dim_OR.iloc[0]:.2f} [{sres.logit_neg_has_dim_OR_ci.iloc[0]}], "
             f"p = {sres.logit_neg_has_dim_p.iloc[0]:.3g}.\n")
    L.append("## NER\n")
    L.append(f"Упоминаний персонажей: {len(nm)} в {nm.kidlit_id.nunique()} книгах. Доля DIM-словоупотреблений, входящих в упоминание "
             f"персонажа: {dim_in_ner:.1%}.\n")
    L.append("| Тип | Упоминаний O / T | доля в диминутивной форме O / T | OR (O/T) | p | по книгам (медианы) | p (MW) |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in nres.iterrows():
        L.append(f"| {r['label']} | {r['mentions_RUS-O']} / {r['mentions_RUS-T']} | {r['dim_share_RUS-O']:.3f} / {r['dim_share_RUS-T']:.3f} | "
                 f"{r['OR_O_vs_T']:.2f} | {r['p_fisher']:.3g} | {r['book_median_RUS-O']:.3f} / {r['book_median_RUS-T']:.3f} | {r['p_mw_books']:.3g} |")
    L.append("\nЧастые диминутивы в упоминаниях: " + "; ".join(
        f"{r['label']}: RUS-O — {r['top_dim_RUS-O']}; RUS-T — {r['top_dim_RUS-T']}" for _, r in nres.iterrows() if r['label'] != "все"))
    (A / "goldlink_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(al.groupby(["project", "aligned"]).size())


if __name__ == "__main__":
    main()
