#!/usr/bin/env python3
"""
KidLit by Numbers — шаги 3.1–3.2: метрики по книгам и сравнение RUS-O и RUS-T.

    python kidlit_dim_analysis.py --data kidlit_dim --cand kidlit_dim/detector_v3.1/candidates.parquet --out kidlit_dim/analysis

3.1 → book_metrics.csv (одна строка на книгу) и book_metrics_variants.csv (варианты для анализа чувствительности).
3.2 → group_tests.csv: Mann–Whitney U, Cliff's δ, 95% bootstrap-интервал (по книгам) для δ и для разницы медиан,
      поправка Бенджамини–Хохберга на все метрики; Spearman с log(длины); регрессия metric ~ group + log(n_words) + genre
      (HC3); поправка на ошибки детектора (P/R по группам на test); чувствительность к решениям P-01, P-19, CONTEXT.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

SEED = 2026
N_BOOT = 5000
P01 = {"бабушка", "дедушка", "девочка", "мальчик"}
P19 = {"дочка", "сынок", "сыночек", "детка", "деточка"}

METRICS = {
    # имя: (описание, направление в статье)
    "dim_per_1000w": "DIM на 1000 слов",
    "dim_noun_per_100n": "DIM-существительные на 100 существительных",
    "dim_lemma_share": "доля DIM-лемм среди лемм существительных",
    "dim_lemmas_n": "число разных DIM-лемм",
    "dim_choice": "доля диминутивного выбора (пары база/диминутив в одной книге)",
    "hyp_per_1000w": "HYP на 1000 слов",
    "young_per_1000w": "YOUNG на 1000 слов",
    "dim_adj_per_1000w": "DIM-прилагательные на 1000 слов",
    "dim_adv_per_1000w": "DIM-наречия на 1000 слов",
    "pej_share": "доля pej среди DIM",
    "name_share": "доля DIM, употреблённых как имя (name)",
}


def e(s):
    return s.replace("ё", "е") if isinstance(s, str) else s


# ---------------------------------------------------------------------------
# 3.1
# ---------------------------------------------------------------------------
def book_metrics(tok, cand, books, variant="base"):
    c = cand.copy()
    c["lem_e"] = c.lemma.map(e)
    if variant == "p01_dim":          # P-01 (бабушка, дедушка, девочка, мальчик) считаем DIM
        c.loc[c.lem_e.isin(P01), "det_label"] = "DIM"
    elif variant == "p19_lex":        # P-19 (дочка, сынок, детка…) считаем LEX
        c.loc[c.lem_e.isin(P19) & (c.det_label == "DIM"), "det_label"] = "LEX"
    elif variant == "no_context":     # контекстные слова (needs_review) не считаем DIM
        c.loc[(c.needs_review == 1) & (c.det_label == "DIM"), "det_label"] = "CTX"
    w = tok[tok.is_word]
    n_words = w.groupby("kidlit_id").size()
    nouns = w[w.pos == "NOUN"]
    n_nouns = nouns.groupby("kidlit_id").size()
    n_noun_lemmas = nouns.groupby("kidlit_id").lemma_e.nunique()
    dim = c[c.det_label == "DIM"]
    rows = []
    lemma_counts = w.groupby(["kidlit_id", "lemma_e"]).size()
    for b in books.kidlit_id:
        d = dim[dim.kidlit_id == b]
        nw, nn = n_words.get(b, 0), n_nouns.get(b, 0)
        dn = d[d.pos == "NOUN"]
        # доля диминутивного выбора: база встречается в книге сама по себе и в диминутиве
        lc = lemma_counts.get(b, pd.Series(dtype=int))
        dnb = dn.assign(base_e=dn.base_lemma.map(e))
        pairs = dnb[dnb.base_e.notna() & (dnb.base_e != dnb.lem_e)].groupby("base_e").size()
        pairs = pairs[[bb in lc.index for bb in pairs.index]]
        n_dim_p, n_base_p = int(pairs.sum()), int(sum(lc[bb] for bb in pairs.index))
        cb = c[c.kidlit_id == b]
        rows.append({
            "kidlit_id": b, "n_words": int(nw), "n_nouns": int(nn), "n_noun_lemmas": int(n_noun_lemmas.get(b, 0)),
            "n_dim": len(d), "n_dim_noun": len(dn), "n_dim_adj": int(d.pos.isin(["ADJF", "ADJS"]).sum()),
            "n_dim_adv": int((d.pos == "ADVB").sum()),
            "n_hyp": int((cb.det_label == "HYP").sum()), "n_young": int((cb.det_label == "YOUNG").sum()),
            "n_lex": int((cb.det_label == "LEX").sum()),
            "dim_lemmas_n": d.lem_e.nunique(),
            "dim_per_1000w": 1000 * len(d) / nw if nw else np.nan,
            "dim_noun_per_100n": 100 * len(dn) / nn if nn else np.nan,
            "dim_lemma_share": dn.lem_e.nunique() / n_noun_lemmas.get(b, np.nan),
            "dim_choice_pairs": len(pairs), "dim_choice_ndim": n_dim_p, "dim_choice_nbase": n_base_p,
            "dim_choice": n_dim_p / (n_dim_p + n_base_p) if (n_dim_p + n_base_p) else np.nan,
            "hyp_per_1000w": 1000 * (cb.det_label == "HYP").sum() / nw if nw else np.nan,
            "young_per_1000w": 1000 * (cb.det_label == "YOUNG").sum() / nw if nw else np.nan,
            "dim_adj_per_1000w": 1000 * d.pos.isin(["ADJF", "ADJS"]).sum() / nw if nw else np.nan,
            "dim_adv_per_1000w": 1000 * (d.pos == "ADVB").sum() / nw if nw else np.nan,
            "pej_share": d.pej.mean() if len(d) else np.nan,
            "name_share": d.name.mean() if len(d) else np.nan,
            "n_context": int(((cb.needs_review == 1) & (cb.det_label == "DIM")).sum()),
        })
    m = pd.DataFrame(rows).merge(books[["kidlit_id", "group", "corpus_type", "genre", "language", "year", "author_gender",
                                        "is_translation"]], on="kidlit_id")
    m["log_words"] = np.log(m.n_words)
    m["variant"] = variant
    return m


# ---------------------------------------------------------------------------
# 3.2
# ---------------------------------------------------------------------------
def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return (gt - lt) / (len(x) * len(y))


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def compare(m, rng, metrics=METRICS):
    rows = []
    for k, desc in metrics.items():
        o = m[m.group == "RUS-O"][k].dropna().values
        t = m[m.group == "RUS-T"][k].dropna().values
        if len(o) < 3 or len(t) < 3:
            continue
        u, p = stats.mannwhitneyu(o, t, alternative="two-sided")
        dlt = cliffs_delta(o, t)
        bd, bm = [], []
        for _ in range(N_BOOT):
            oo, tt = rng.choice(o, len(o)), rng.choice(t, len(t))
            bd.append(cliffs_delta(oo, tt))
            bm.append(np.median(oo) - np.median(tt))
        sub = m[[k, "group", "log_words", "genre"]].dropna()
        try:
            fit = smf.ols(f"{k} ~ C(group, Treatment('RUS-T')) + log_words + C(genre)", data=sub).fit(cov_type="HC3")
            gname = [n for n in fit.params.index if n.startswith("C(group")][0]
            b_g, p_g = fit.params[gname], fit.pvalues[gname]
            ci_g = fit.conf_int().loc[gname].tolist()
            b_l, p_l = fit.params["log_words"], fit.pvalues["log_words"]
        except Exception:
            b_g = p_g = b_l = p_l = np.nan
            ci_g = [np.nan, np.nan]
        rho, prho = stats.spearmanr(sub[k], sub.log_words)
        rows.append({"metric": k, "description": desc, "n_rus_o": len(o), "n_rus_t": len(t),
                     "median_rus_o": np.median(o), "median_rus_t": np.median(t),
                     "iqr_rus_o": f"{np.percentile(o, 25):.3g}–{np.percentile(o, 75):.3g}",
                     "iqr_rus_t": f"{np.percentile(t, 25):.3g}–{np.percentile(t, 75):.3g}",
                     "median_diff": np.median(o) - np.median(t),
                     "median_diff_ci": f"{np.percentile(bm, 2.5):.3g}…{np.percentile(bm, 97.5):.3g}",
                     "U": u, "p_mw": p, "cliffs_delta": dlt,
                     "delta_ci_lo": np.percentile(bd, 2.5), "delta_ci_hi": np.percentile(bd, 97.5),
                     "spearman_loglen_rho": rho, "spearman_loglen_p": prho,
                     "ols_group_coef": b_g, "ols_group_ci": f"{ci_g[0]:.3g}…{ci_g[1]:.3g}", "ols_group_p": p_g,
                     "ols_loglen_coef": b_l, "ols_loglen_p": p_l})
    r = pd.DataFrame(rows)
    r["q_bh"] = bh(r.p_mw.values)
    r["q_bh_ols_group"] = bh(r.ols_group_p.fillna(1).values)
    return r


def detector_correction(m, eval_json):
    """Поправка частоты DIM на ошибки детектора: истинная ≈ наблюдаемая × P / R (P, R — по группе на test)."""
    ev = json.loads(Path(eval_json).read_text()) if eval_json and Path(eval_json).exists() else None
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kidlit_dim")
    ap.add_argument("--cand", default="kidlit_dim/detector_v3.1/candidates.parquet")
    ap.add_argument("--out", default="kidlit_dim/analysis")
    ap.add_argument("--pr", default="", help="P/R по группам на test: RUS-O:P:R,RUS-T:P:R")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    data = Path(args.data)
    tok = pd.read_parquet(data / "data" / "tokens.parquet",
                          columns=["kidlit_id", "is_word", "pos", "lemma", "lemma_e"])
    tok["lemma_e"] = tok.lemma.map(e)
    books = pd.read_csv(data / "data" / "books.csv")
    books["genre"] = books.genre.fillna("проза")
    cand = pd.read_parquet(args.cand)

    ms = {v: book_metrics(tok, cand, books, v) for v in ["base", "p01_dim", "p19_lex", "no_context"]}
    ms["base"].to_csv(out / "book_metrics.csv", index=False)
    pd.concat(ms.values()).to_csv(out / "book_metrics_variants.csv", index=False)

    res = compare(ms["base"], rng)
    res.to_csv(out / "group_tests.csv", index=False)

    # чувствительность: главная метрика в вариантах
    sens = []
    for v, mv in ms.items():
        r = compare(mv, np.random.default_rng(SEED), {k: METRICS[k] for k in ["dim_per_1000w", "dim_noun_per_100n", "dim_choice"]})
        r["variant"] = v
        sens.append(r)
    sens = pd.concat(sens)
    # поправка на ошибки детектора (по P/R групп на test)
    corr = []
    if args.pr:
        pr = {g: (float(p), float(r)) for g, p, r in (x.split(":") for x in args.pr.split(","))}
        mc = ms["base"].copy()
        for col in ["dim_per_1000w", "dim_noun_per_100n"]:
            mc[col] = [v * pr[g][0] / pr[g][1] for v, g in zip(mc[col], mc.group)]
        r = compare(mc, np.random.default_rng(SEED), {k: METRICS[k] for k in ["dim_per_1000w", "dim_noun_per_100n"]})
        r["variant"] = "detector_corrected"
        sens = pd.concat([sens, r])
    sens.to_csv(out / "sensitivity.csv", index=False)

    # языки оригинала (описательно)
    t = ms["base"][ms["base"].group == "RUS-T"].copy()
    vc = t.language.value_counts()
    t["lang_group"] = t.language.where(t.language.map(vc) >= 5, "другие")
    lang = (t.groupby("lang_group").agg(n_books=("kidlit_id", "size"), median_dim_per_1000w=("dim_per_1000w", "median"),
                                         median_dim_noun_per_100n=("dim_noun_per_100n", "median"),
                                         median_dim_choice=("dim_choice", "median"), median_words=("n_words", "median"))
            .reset_index().sort_values("n_books", ascending=False))
    lang.to_csv(out / "rus_t_by_language.csv", index=False)
    pd.set_option("display.width", 250)
    print(res[["metric", "median_rus_o", "median_rus_t", "cliffs_delta", "delta_ci_lo", "delta_ci_hi", "p_mw", "q_bh",
               "spearman_loglen_rho", "ols_group_coef", "ols_group_p", "q_bh_ols_group"]].round(3).to_string())
    print(sens[["variant", "metric", "median_rus_o", "median_rus_t", "cliffs_delta", "q_bh", "ols_group_p"]].round(3).to_string())
    print(lang.round(2).to_string())


if __name__ == "__main__":
    main()
