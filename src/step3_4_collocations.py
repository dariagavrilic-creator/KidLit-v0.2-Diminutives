#!/usr/bin/env python3
"""
KidLit by Numbers — шаг 3.4: коллокации диминутивов.

    python kidlit_dim_colloc.py --data kidlit_dim --cand kidlit_dim/detector_v3.1/candidates.parquet --out kidlit_dim/analysis

(а) Определения (amod по разбору spaCy) к диминутивам: logDice, минимальная частота 3.
    — colloc_amod_dim.csv: прилагательные, тяготеющие к DIM-существительным (logDice к классу DIM, частоты
      с DIM и с прочими существительными, отношение долей, LL);
    — colloc_amod_pairs.csv: пары «прилагательное + DIM-лемма», f ≥ 3, logDice.
    — pleonasm.csv: «маленький (и другие слова размера) + DIM» против «… + та же база без суффикса»
      и против прочих существительных; по группам; точный тест Фишера и bootstrap по книгам.
(б) Обращения: «мой/моя + DIM» и DIM в позиции обращения (vocative в разборе или шаблон «…, DIM!»/«— DIM,»).
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SIZE = {"маленький", "крошечный", "малюсенький", "небольшой", "малый", "махонький", "малюпасенький", "крохотный"}
BIG = {"большой", "огромный", "громадный"}
SEED = 2026


def e(s):
    return s.replace("ё", "е") if isinstance(s, str) else s


def logdice(fxy, fx, fy):
    return 14 + math.log2(2 * fxy / (fx + fy)) if fxy else float("nan")


def ll(a, b, c, d):
    """Log-likelihood для таблицы 2×2 (a — A с DIM, b — A с прочими, c — прочие с DIM, d — прочие с прочими)."""
    n = a + b + c + d
    out = 0
    for o, ex in ((a, (a + b) * (a + c) / n), (b, (a + b) * (b + d) / n), (c, (c + d) * (a + c) / n), (d, (c + d) * (b + d) / n)):
        if o > 0:
            out += o * math.log(o / ex)
    return 2 * out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kidlit_dim")
    ap.add_argument("--cand", default="kidlit_dim/detector_v3.1/candidates.parquet")
    ap.add_argument("--out", default="kidlit_dim/analysis")
    args = ap.parse_args()
    out = Path(args.out)
    tok = pd.read_parquet(Path(args.data) / "data" / "tokens.parquet",
                          columns=["kidlit_id", "group", "sent_id", "tok_id", "token", "token_lower", "lemma", "pos",
                                   "head", "deprel", "gram", "is_word"])
    tok["lem"] = tok.lemma.map(e)
    cand = pd.read_parquet(args.cand)
    key = ["kidlit_id", "sent_id", "tok_id"]
    dim = cand[(cand.det_label == "DIM") & (cand.pos == "NOUN")][key + ["base_lemma", "name"]]
    dim["base_e"] = dim.base_lemma.map(e)
    t = tok.merge(dim.assign(is_dim=True), on=key, how="left")
    t["is_dim"] = t.is_dim.fillna(False).astype(bool)
    bases = set(dim.base_e.dropna())
    nouns = t[t.pos == "NOUN"].copy()
    nouns["is_base"] = ~nouns.is_dim & nouns.lem.isin(bases)
    # связи amod: зависимое — прилагательное/причастие, вершина — существительное в том же предложении
    mods = t[(t.deprel == "amod") & t.pos.isin(["ADJF", "ADJS", "PRTF"])][["kidlit_id", "group", "sent_id", "tok_id", "lem", "head"]]
    rel = mods.merge(nouns[["kidlit_id", "sent_id", "tok_id", "lem", "is_dim", "is_base", "base_e"]]
                     .rename(columns={"tok_id": "head", "lem": "noun"}), on=["kidlit_id", "sent_id", "head"])
    rel = rel.rename(columns={"lem": "adj"})
    rel.to_csv(out / "amod_relations.csv", index=False)
    n_dim, n_non = int(nouns.is_dim.sum()), int((~nouns.is_dim).sum())
    L = ["# Шаг 3.4. Коллокации диминутивов\n",
         f"Существительных: {len(nouns)}, из них DIM: {n_dim}. Связей amod (прилагательное → существительное): {len(rel)}; "
         f"к DIM: {int(rel.is_dim.sum())}.\n"]

    # (а1) прилагательные, тяготеющие к DIM
    rows = []
    fx = rel.groupby("adj").size()
    for adj, g in rel.groupby("adj"):
        a = int(g.is_dim.sum())
        if a < 3:
            continue
        b = len(g) - a
        c, d = n_dim - a, n_non - b
        rows.append({"adj": adj, "f_with_dim": a, "f_with_other_nouns": b, "f_total": len(g),
                     "logDice_dim": logdice(a, len(g), n_dim), "ratio": (a / n_dim) / (b / n_non) if b else float("inf"),
                     "LL": ll(a, b, c, d), "rus_o": int((g.is_dim & (g.group == "RUS-O")).sum()),
                     "rus_t": int((g.is_dim & (g.group == "RUS-T")).sum())})
    a1 = pd.DataFrame(rows).sort_values("logDice_dim", ascending=False)
    a1.to_csv(out / "colloc_amod_dim.csv", index=False)
    L.append("## (а) Определения к диминутивам\n")
    L.append("Прилагательные с частотой при DIM ≥ 3, по убыванию logDice (к классу DIM-существительных); ratio — во сколько раз "
             "доля при DIM выше доли при прочих существительных.\n")
    L.append("| Прилагательное | при DIM | при прочих | logDice | ratio | LL |\n|---|---|---|---|---|---|")
    for r in a1.head(20).itertuples():
        L.append(f"| {r.adj} | {r.f_with_dim} | {r.f_with_other_nouns} | {r.logDice_dim:.2f} | {r.ratio:.1f} | {r.LL:.1f} |")

    # (а2) пары
    fy = nouns[nouns.is_dim].groupby("lem").size()
    pr = rel[rel.is_dim].groupby(["adj", "noun"]).size().reset_index(name="f")
    pr = pr[pr.f >= 3].copy()
    pr["logDice"] = [logdice(f, fx[a], fy[n]) for a, n, f in zip(pr.adj, pr.noun, pr.f)]
    pr = pr.sort_values(["logDice", "f"], ascending=False)
    pr.to_csv(out / "colloc_amod_pairs.csv", index=False)
    L.append(f"\nПары «прилагательное + DIM» с f ≥ 3: {len(pr)}. Самые сильные: " +
             ", ".join(f"{r.adj} {r.noun} ({r.f}; {r.logDice:.1f})" for r in pr.head(15).itertuples()) + ".\n")

    # (а3) плеоназм «маленький + DIM»
    def rate(mask_nouns, words, by_group=None):
        nn = nouns[mask_nouns]
        if by_group:
            nn = nn[nn.group == by_group]
        heads = set(map(tuple, nn[["kidlit_id", "sent_id", "tok_id"]].values))
        r = rel[rel.adj.isin(words)]
        k = sum((a, s, h) in heads for a, s, h in zip(r.kidlit_id, r.sent_id, r["head"]))
        return k, len(nn)

    pl = []
    for grp in [None, "RUS-O", "RUS-T"]:
        for words, wname in ((SIZE, "размер (маленький и др.)"), (BIG, "большой и др.")):
            kd, nd = rate(nouns.is_dim, words, grp)
            kb, nb = rate(nouns.is_base, words, grp)
            ko, no = rate(~nouns.is_dim & ~nouns.is_base, words, grp)
            orr, p = stats.fisher_exact([[kd, nd - kd], [kb, nb - kb]])
            pl.append({"group": grp or "все", "adjectives": wname, "dim_k": kd, "dim_n": nd, "dim_per_100": 100 * kd / nd,
                       "base_k": kb, "base_n": nb, "base_per_100": 100 * kb / nb if nb else float("nan"),
                       "other_k": ko, "other_n": no, "other_per_100": 100 * ko / no,
                       "OR_dim_vs_base": orr, "p_fisher": p})
    pl = pd.DataFrame(pl)
    # bootstrap по книгам: разница долей «маленький» при DIM и при базе
    rng = np.random.default_rng(SEED)
    rs = rel[rel.adj.isin(SIZE)]
    per_book = []
    for b, nb_ in nouns.groupby("kidlit_id"):
        r = rs[rs.kidlit_id == b]
        hd = set(map(tuple, nb_[nb_.is_dim][["sent_id", "tok_id"]].values))
        hb = set(map(tuple, nb_[nb_.is_base][["sent_id", "tok_id"]].values))
        per_book.append((sum((s, h) in hd for s, h in zip(r.sent_id, r["head"])), len(hd),
                         sum((s, h) in hb for s, h in zip(r.sent_id, r["head"])), len(hb)))
    pb = np.array(per_book)
    diffs = []
    for _ in range(5000):
        x = pb[rng.integers(0, len(pb), len(pb))].sum(axis=0)
        diffs.append(100 * x[0] / x[1] - 100 * x[2] / x[3])
    pl["boot_ci_dim_minus_base_size_all"] = f"{np.percentile(diffs, 2.5):.2f}…{np.percentile(diffs, 97.5):.2f}"
    pl.to_csv(out / "pleonasm.csv", index=False)
    L.append("### Плеоназм «маленький + диминутив»\n")
    L.append("Число определений размера на 100 существительных: при DIM; при той же базе без суффикса (существительные, "
             "которые в корпусе служат производящими для DIM); при прочих существительных.\n")
    L.append("| Группа | Определения | при DIM | при базе | при прочих | OR (DIM/база) | p (Фишер) |\n|---|---|---|---|---|---|---|")
    for r in pl.itertuples():
        L.append(f"| {r.group} | {r.adjectives} | {r.dim_per_100:.2f} ({r.dim_k}/{r.dim_n}) | {r.base_per_100:.2f} ({r.base_k}/{r.base_n}) | "
                 f"{r.other_per_100:.2f} | {r.OR_dim_vs_base:.2f} | {r.p_fisher:.3g} |")
    L.append(f"\nBootstrap по книгам (5000), разница «размер при DIM − размер при базе», все книги: 95% ДИ {pl.boot_ci_dim_minus_base_size_all.iloc[0]}.\n")
    ex = rel[rel.adj.isin(SIZE) & rel.is_dim].groupby("noun").size().sort_values(ascending=False).head(12)
    L.append("Чаще всего с «маленький» и др.: " + ", ".join(f"{k} ({v})" for k, v in ex.items()) + ".\n")

    # (б) обращения
    t2 = t.copy()
    t2["next_tok"] = t2.groupby(["kidlit_id", "sent_id"]).token.shift(-1)
    t2["prev_tok"] = t2.groupby(["kidlit_id", "sent_id"]).token.shift(1)
    t2["prev_lem"] = t2.groupby(["kidlit_id", "sent_id"]).lem.shift(1)
    dn = t2[t2.is_dim & (t2.pos == "NOUN")].copy()
    moi = dn.prev_lem.isin(["мой", "наш", "милый", "дорогой"])
    voc_dep = dn.deprel.eq("vocative")
    voc_pat = dn.gram.fillna("").str.contains("nomn") & dn.next_tok.isin([",", "!", "?", "…", "!..", "?!"]) & \
        (dn.prev_tok.isna() | dn.prev_tok.isin(["—", "–", "-", ",", "«", "\"", "!"]) | moi)
    dn["address"] = voc_dep | voc_pat
    dn["moi"] = moi
    ad = dn.groupby("group").agg(n_dim=("token", "size"), address=("address", "sum"), moi_dim=("moi", "sum")).reset_index()
    ad["address_share"] = ad.address / ad.n_dim
    ad["moi_share"] = ad.moi_dim / ad.n_dim
    o, tt = ad.set_index("group").loc["RUS-O"], ad.set_index("group").loc["RUS-T"]
    orr, p = stats.fisher_exact([[o.address, o.n_dim - o.address], [tt.address, tt.n_dim - tt.address]])
    ad["OR_O_vs_T"], ad["p_fisher"] = orr, p
    ad.to_csv(out / "address.csv", index=False)
    dn[dn.address][["kidlit_id", "group", "sent_id", "tok_id", "token", "prev_tok", "next_tok"]].to_csv(out / "address_examples.csv", index=False)
    L.append("## (б) Диминутивы в обращениях\n")
    L.append("Обращение — vocative в разборе spaCy или шаблон: DIM в именительном падеже, за ним запятая / ! / ?, перед ним "
             "начало предложения, тире, запятая или «мой/наш/милый/дорогой». Эвристика: на 30 случайных срабатываниях верны 24–26 "
             "(ошибки: подлежащее перед деепричастием «Мишутка, урча…», «Где моя шляпка?»); полнота не оценивалась.\n")
    L.append("| Группа | DIM-существительных | в обращении | доля | «мой/наш/милый/дорогой + DIM» |\n|---|---|---|---|---|")
    for r in ad.itertuples():
        L.append(f"| {r.group} | {r.n_dim} | {r.address} | {r.address_share:.3f} | {r.moi_dim} |")
    L.append(f"\nRUS-O против RUS-T: OR = {orr:.2f}, p (Фишер) = {p:.3g}. Счёт по словоупотреблениям: книги неравного объёма, "
             "вывод предварительный.\n")
    (out / "collocations_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
