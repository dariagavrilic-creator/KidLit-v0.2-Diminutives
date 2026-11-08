#!/usr/bin/env python3
"""
KidLit by Numbers — шаг 3.5 на двойной разметке KidLit-Gold (30 книг, разметчицы A и B).

    python kidlit_dim_gold30.py

Вход: data/combined.json (A и B: сентимент и NER), data/KidLit_Adjudication_Log.csv (решения по 10 книгам),
      детектор v3.1.
Выход (analysis/): gold30_sentences.csv, gold30_sentiment_iaa.csv, gold30_sentiment_dim.csv, gold30_ner_dim.csv,
                   gold30_log_coverage.csv, gold30_report.md
"""
import importlib.util
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

spec = importlib.util.spec_from_file_location("gl", "kidlit_dim_goldlink.py")
gl = importlib.util.module_from_spec(spec); spec.loader.exec_module(gl)


def kappa(a, b):
    a, b = list(a), list(b); n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def sentence_labels(S_book, spans):
    """Метка предложения корпуса = метка спана с наибольшим перекрытием (NaN — предложение не размечено)."""
    if len(spans) == 0:
        return [np.nan] * len(S_book)
    ss, ee, ll = spans.s.values, spans.e.values, spans.label.values
    out = []
    for st, en in zip(S_book.start, S_book.end):
        ov = np.minimum(ee, en) - np.maximum(ss, st)
        j = int(np.argmax(ov))
        out.append(ll[j] if ov[j] > 0 else np.nan)
    return out


def main():
    data = json.loads((D / "labelstudio" / "combined_4.json").read_text())
    books = pd.read_csv(D / "books.csv")
    bid = dict(zip(books.kidlit_id, books.book_id)); grp = dict(zip(books.kidlit_id, books.group))
    sents = pd.read_parquet(D / "sentences.parquet")
    tok = pd.read_parquet(D / "tokens.parquet", columns=["kidlit_id", "sent_id", "tok_id", "start", "end", "is_word", "token"])
    cand = pd.read_parquet("kidlit_dim/detector_v3.1/candidates.parquet")
    dimk = cand[cand.det_label == "DIM"][["kidlit_id", "sent_id", "tok_id", "lemma", "name"]]
    tok = tok.merge(dimk.assign(is_dim=True), on=["kidlit_id", "sent_id", "tok_id"], how="left")
    tok["is_dim"] = tok.is_dim.fillna(False).astype(bool)

    sent_spans, ner_spans, fails = [], [], 0
    for x in data:
        k = x["book_id"]
        ours = (D / "texts" / f"{bid[k]}.txt").read_text(encoding="utf-8")
        mp = gl.build_map(x["text"], ours)
        for lay, out, lab_map in (("sentiment_annotations", sent_spans, SENT), ("ner_annotations", ner_spans, None)):
            for a in x[lay]:
                sp = gl.map_span(mp[0], mp[1], a["start"], a["end"])
                if sp is None:
                    fails += 1; continue
                out.append({"kidlit_id": k, "group": grp[k], "ann": x["annotator"], "s": sp[0], "e": sp[1],
                            "label": lab_map[a["label"]] if lab_map else a["label"], "text": a["text"]})
    sp = pd.DataFrame(sent_spans); nr = pd.DataFrame(ner_spans)

    # предложения корпуса с метками A и B
    S = sents[sents.kidlit_id.isin(sp.kidlit_id.unique())].copy()
    for ann in ["A", "B"]:
        S[ann] = np.nan
        for k, g in S.groupby("kidlit_id"):
            S.loc[g.index, ann] = sentence_labels(g, sp[(sp.kidlit_id == k) & (sp.ann == ann)].sort_values("s"))
    ds = tok[tok.is_dim].groupby(["kidlit_id", "sent_id"]).size().rename("n_dim").reset_index()
    S = S.merge(ds, on=["kidlit_id", "sent_id"], how="left")
    S["n_dim"] = S.n_dim.fillna(0).astype(int); S["has_dim"] = S.n_dim > 0

    # решения адъюдикации (лог): метка для предложений, где A и B расходятся
    log = pd.read_csv(D / "KidLit_Adjudication_Log.csv")
    ls = log[log.type == "Sentiment_label"].copy()
    ls["target"] = ls.context.str.extract(r"\[\[(.*?)\]\]", flags=re.S)[0]
    norm = lambda t: re.sub(r"\W+", "", str(t).lower().replace("ё", "е"))
    S["n"] = S.text.map(norm)
    S["adj"] = np.nan
    for r in ls.itertuples():
        tn = norm(r.target)
        idx = S[(S.kidlit_id == r.book_id).values & np.array([(tn in v) or (len(v) > 15 and v in tn) for v in S.n])].index
        S.loc[idx, "adj"] = SENT.get(r.decision, np.nan)
    S["both"] = S.A.notna() & S.B.notna()
    S["agree"] = S.both & (S.A == S.B)
    log_books = set(ls.book_id)
    S["gold"] = np.where(S.agree, S.A, np.where(S.kidlit_id.isin(log_books) & S.both, S.adj, np.nan))
    S.drop(columns=["n"]).to_csv(A / "gold30_sentences.csv", index=False)

    # покрытие лога: сколько расхождений A/B в книгах лога закрыто решением
    cov = (S[S.kidlit_id.isin(log_books) & S.both & ~S.agree].groupby("kidlit_id")
           .agg(disagreements=("adj", "size"), resolved=("adj", lambda v: int(v.notna().sum()))).reset_index())
    cov["resolved_share"] = cov.resolved / cov.disagreements
    cov.to_csv(A / "gold30_log_coverage.csv", index=False)

    # IAA сентимента: κ на предложениях с DIM и без; bootstrap по книгам для разницы
    B_ = S[S.both]
    rows = []
    for g in ["все", "RUS-O", "RUS-T"]:
        x = B_ if g == "все" else B_[B_.group == g]
        kd, kn = kappa(x[x.has_dim].A, x[x.has_dim].B), kappa(x[~x.has_dim].A, x[~x.has_dim].B)
        rows.append({"group": g, "books": x.kidlit_id.nunique(), "sentences": len(x), "with_dim": int(x.has_dim.sum()),
                     "kappa_all": kappa(x.A, x.B), "kappa_dim": kd, "kappa_nodim": kn,
                     "agree_dim": (x[x.has_dim].A == x[x.has_dim].B).mean(), "agree_nodim": (x[~x.has_dim].A == x[~x.has_dim].B).mean()})
    iaa = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    bl = list(B_.kidlit_id.unique()); diffs = []
    for _ in range(2000):
        pick = rng.choice(bl, len(bl))
        x = pd.concat([B_[B_.kidlit_id == b] for b in pick])
        diffs.append(kappa(x[x.has_dim].A, x[x.has_dim].B) - kappa(x[~x.has_dim].A, x[~x.has_dim].B))
    iaa["boot_ci_kappa_dim_minus_nodim"] = f"{np.nanpercentile(diffs, 2.5):.3f}…{np.nanpercentile(diffs, 97.5):.3f}"
    iaa.to_csv(A / "gold30_sentiment_iaa.csv", index=False)

    # сентимент и DIM: отдельно A, B, согласованные (A = B) и gold (10 книг лога)
    res = []
    for src in ["A", "B", "agreed", "gold_10books"]:
        if src in ("A", "B"):
            x = S[S[src].notna()].assign(lab=S[src])
        elif src == "agreed":
            x = S[S.agree].assign(lab=S.A)
        else:
            x = S[S.gold.notna() & S.kidlit_id.isin(log_books)].assign(lab=S.gold)
        x = x.assign(pos=(x.lab == 1).astype(int), neg=(x.lab == -1).astype(int), hd=x.has_dim.astype(int))
        r = {"source": src, "books": x.kidlit_id.nunique(), "sentences": len(x), "with_dim": int(x.has_dim.sum())}
        for lab, nm in ((1, "pos"), (0, "neu"), (-1, "neg")):
            r[f"{nm}_dim"] = (x[x.has_dim].lab == lab).mean(); r[f"{nm}_nodim"] = (x[~x.has_dim].lab == lab).mean()
        for y in ["pos", "neg"]:
            f = f"{y} ~ hd + np.log(n_tokens)" + (" + C(group)" if x.group.nunique() > 1 else "")
            fit = smf.logit(f, data=x).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(x.kidlit_id)[0]})
            ci = np.exp(fit.conf_int().loc["hd"]).values
            r[f"OR_{y}"] = float(np.exp(fit.params["hd"])); r[f"OR_{y}_ci"] = f"{ci[0]:.2f}…{ci[1]:.2f}"; r[f"p_{y}"] = float(fit.pvalues["hd"])
        res.append(r)
    sres = pd.DataFrame(res); sres.to_csv(A / "gold30_sentiment_dim.csv", index=False)

    # NER: доля упоминаний в диминутивной форме, по разметчицам; согласие по спанам (точное совпадение границ и метки)
    tw = tok[tok.is_word]
    isd = []
    for k, g in nr.groupby("kidlit_id"):
        t = tw[tw.kidlit_id == k]; ts, te, td = t.start.values, t.end.values, t.is_dim.values
        for i, r in g.iterrows():
            m = (ts < r.e) & (te > r.s); isd.append((i, bool(td[m].any())))
    nr["is_dim"] = pd.Series(dict(isd))
    nrows = []
    for ann in ["A", "B"]:
        for lab in ["PER_NAME", "PER_ROLE", "PER_FAMILY", "все"]:
            x = nr[nr.ann == ann]; x = x if lab == "все" else x[x.label == lab]
            o, t = x[x.group == "RUS-O"], x[x.group == "RUS-T"]
            orr, p = stats.fisher_exact([[o.is_dim.sum(), (~o.is_dim).sum()], [t.is_dim.sum(), (~t.is_dim).sum()]])
            bb = x.groupby(["kidlit_id", "group"]).is_dim.mean().reset_index()
            pb = stats.mannwhitneyu(bb[bb.group == "RUS-O"].is_dim, bb[bb.group == "RUS-T"].is_dim).pvalue
            nrows.append({"annotator": ann, "label": lab, "n_RUS-O": len(o), "n_RUS-T": len(t),
                          "dim_share_RUS-O": o.is_dim.mean(), "dim_share_RUS-T": t.is_dim.mean(), "OR": orr, "p_fisher": p,
                          "book_median_RUS-O": bb[bb.group == "RUS-O"].is_dim.median(),
                          "book_median_RUS-T": bb[bb.group == "RUS-T"].is_dim.median(), "p_mw_books": pb})
    ndim = pd.DataFrame(nrows)
    # согласие NER: F1 по точным спанам (с меткой), отдельно для диминутивных и прочих упоминаний
    def spans(ann, dim=None):
        x = nr[nr.ann == ann]
        if dim is not None:
            x = x[x.is_dim == dim]
        return set(zip(x.kidlit_id, x.s, x.e, x.label))
    f1 = {}
    for nm, dim in (("all", None), ("dim", True), ("nodim", False)):
        a_, b_ = spans("A", dim), spans("B", dim)
        f1[nm] = 2 * len(a_ & b_) / (len(a_) + len(b_)) if a_ or b_ else float("nan")
    ndim["ner_F1_all"], ndim["ner_F1_dim"], ndim["ner_F1_nodim"] = f1["all"], f1["dim"], f1["nodim"]
    ndim.to_csv(A / "gold30_ner_dim.csv", index=False)

    # отчёт
    L = ["# Шаг 3.5 на двойной разметке KidLit-Gold (30 книг, A и B)\n",
         f"Файл: KidLit_NER_Sentiment_Combined (4).json — 30 книг × 2 разметчицы, статус complete. Перенос смещений: не перенесено спанов {fails}.\n",
         "## Что есть в файле\n",
         f"- Сентимент и NER у A и B для всех 30 книг KidLit-Gold; тексты A и B совпадают.",
         f"- Итоговых (адъюдицированных) меток в файле нет. Лог адъюдикации покрывает {len(log_books)} книг; "
         f"в них закрыто решением {cov.resolved.sum()} из {cov.disagreements.sum()} расхождений на уровне предложений "
         f"({cov.resolved.sum() / max(cov.disagreements.sum(), 1):.0%}).",
         f"- Предложений корпуса с метками обеих: {int(S.both.sum())} из {len(S)}; совпадение A и B: {S[S.both].agree.mean():.1%}.\n",
         "## Согласие по сентименту: затрудняют ли диминутивы разметку?\n",
         "| Группа | Предложений (с DIM) | κ все | κ с DIM | κ без DIM | совпадение с DIM / без |", "|---|---|---|---|---|---|"]
    for r in iaa.itertuples():
        L.append(f"| {r.group} | {r.sentences} ({r.with_dim}) | {r.kappa_all:.3f} | {r.kappa_dim:.3f} | {r.kappa_nodim:.3f} | {r.agree_dim:.3f} / {r.agree_nodim:.3f} |")
    L.append(f"\nBootstrap по книгам (2000): κ(с DIM) − κ(без DIM), 95% ДИ {iaa.boot_ci_kappa_dim_minus_nodim.iloc[0]}.\n")
    L.append("## Сентимент предложений с DIM и без\n")
    L.append("Логистическая регрессия «метка ~ есть DIM + log(длина предложения) + группа», ошибки кластеризованы по книгам.\n")
    L.append("| Источник меток | Книг | Предл. (с DIM) | +1 с/без | 0 с/без | −1 с/без | OR(+1) [ДИ], p | OR(−1) [ДИ], p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in sres.itertuples():
        L.append(f"| {r.source} | {r.books} | {r.sentences} ({r.with_dim}) | {r.pos_dim:.2f}/{r.pos_nodim:.2f} | {r.neu_dim:.2f}/{r.neu_nodim:.2f} | "
                 f"{r.neg_dim:.2f}/{r.neg_nodim:.2f} | {r.OR_pos:.2f} [{r.OR_pos_ci}], {r.p_pos:.3g} | {r.OR_neg:.2f} [{r.OR_neg_ci}], {r.p_neg:.3g} |")
    L.append("\n## NER: персонажи в диминутивной форме\n")
    L.append(f"Согласие A и B по упоминаниям (точные границы и метка), F1: все — {f1['all']:.3f}; диминутивные — {f1['dim']:.3f}; прочие — {f1['nodim']:.3f}.\n")
    L.append("| Разметчица | Тип | Упоминаний O / T | доля DIM O / T | OR | p | медианы по книгам O / T | p (MW) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in ndim.iterrows():
        L.append(f"| {r['annotator']} | {r['label']} | {r['n_RUS-O']} / {r['n_RUS-T']} | {r['dim_share_RUS-O']:.3f} / {r['dim_share_RUS-T']:.3f} | "
                 f"{r['OR']:.2f} | {r['p_fisher']:.3g} | {r['book_median_RUS-O']:.3f} / {r['book_median_RUS-T']:.3f} | {r['p_mw_books']:.3g} |")
    (A / "gold30_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(cov.to_string())


if __name__ == "__main__":
    main()
