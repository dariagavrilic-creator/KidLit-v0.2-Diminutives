#!/usr/bin/env python3
"""
KidLit by Numbers — шаг 2.4: оценка детектора на золотой выборке.

    python kidlit_dim_eval.py --cand kidlit_dim/detector_v3/candidates.parquet            # только dev
    python kidlit_dim_eval.py --cand ... --test                                           # test: ОДИН раз

Dev = пилот (2 книги, сплошное чтение) + dev-часть KidLit-Gold (2 книги + 200 кандидатов).
Test = 4 книги сплошного чтения + 200 кандидатов. Test-цифры не печатаются без --test; первый прогон
на test записывается в gold/TEST_RUN.json (хеш файла кандидатов, время). Повторный прогон с другим
детектором печатает предупреждение: такие цифры уже нельзя выдавать за независимую оценку.

Часть 1 (сплошное чтение): точность, полнота, F1 по DIM / YOUNG / HYP — прямо по всем словам книг.
Часть 2 (кандидаты): выборка стратифицирована по меткам детектора v2. Для новой версии детектора
доля верных и точность DIM оцениваются с весами N_страты / n_страты (каждое слово выборки представляет
свою страту v2). Слово, которое новая версия больше не считает кандидатом, получает метку «—».
"""

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

KEY = ["kidlit_id", "sent_id", "tok_id"]
LAB3 = ["DIM", "YOUNG", "HYP"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def f3(x):
    return "—" if x != x else f"{x:.3f}".replace(".", ",")


def load_gold(gdir, pilot):
    g = pd.read_csv(gdir / "gold_2.3.csv", keep_default_na=False, dtype={"id": str})
    p = pd.read_csv(pilot, keep_default_na=False)
    p = p[p.part == 1].copy()
    p["gold"] = p.gold.where(p.gold.isin(LAB3), "")
    p["split"], p["id"], p["source"] = "dev", "pilot", "pilot_v0.3"
    p1 = pd.concat([g[g.part == 1][["split"] + KEY + ["token", "gold"]].assign(origin="gold"),
                    p[["split"] + KEY + ["token", "gold"]].assign(origin="pilot")])
    p2 = g[g.part == 2].copy()
    return p1, p2


def eval_part1(cand, p1, split, L, err_rows):
    g = p1[p1.split == split]
    books = sorted(g.kidlit_id.unique())
    c = cand[cand.kidlit_id.isin(books)].set_index(KEY)
    tok = set(map(tuple, g[KEY].values))
    L.append(f"### Часть 1. Сплошное чтение ({len(books)} книг, {len(g)} слов)\n")
    L.append("| Метка | Gold | Детектор | Верно | Точность (95% ДИ) | Полнота (95% ДИ) | F1 |\n|---|---|---|---|---|---|---|")
    res = {}
    for lab in LAB3:
        gp = set(map(tuple, g[g.gold == lab][KEY].values))
        dp = set(k for k in c.index[c.det_label == lab] if k in tok)
        tp = len(gp & dp)
        P, R = wilson(tp, len(dp)), wilson(tp, len(gp))
        f1 = 2 * P[0] * R[0] / (P[0] + R[0]) if (P[0] == P[0] and R[0] == R[0] and P[0] + R[0]) else float("nan")
        res[lab] = (P, R, f1, tp, len(dp), len(gp))
        L.append(f"| {lab} | {len(gp)} | {len(dp)} | {tp} | {f3(P[0])} ({f3(P[1])}–{f3(P[2])}) | "
                 f"{f3(R[0])} ({f3(R[1])}–{f3(R[2])}) | {f3(f1)} |")
        gi = g.set_index(KEY)
        for k in dp - gp:
            err_rows.append({"split": split, "part": 1, "error": f"FP {lab}", **dict(zip(KEY, k)), "token": gi.token.get(k),
                             "gold": gi.gold.get(k) or "—", "det": lab, "lemma": c.lemma.get(k), "base": c.base_lemma.get(k),
                             "rule": c.rule_id.get(k), "context": c.context.get(k)})
        for k in gp - dp:
            err_rows.append({"split": split, "part": 1, "error": f"FN {lab}", **dict(zip(KEY, k)), "token": gi.token.get(k),
                             "gold": lab, "det": c.det_label.get(k, "не кандидат"), "lemma": c.lemma.get(k),
                             "base": c.base_lemma.get(k), "rule": c.rule_id.get(k), "context": c.context.get(k)})
    # по группам
    L.append("\nDIM по группам: " + "; ".join(
        f"{grp} — P {f3(_pr(c, g, grp)[0])}, R {f3(_pr(c, g, grp)[1])}" for grp in ["RUS-O", "RUS-T"]) + ".\n")
    return res


def _pr(c, g, grp):
    g = g[g.kidlit_id.str.startswith(grp)]
    tok = set(map(tuple, g[KEY].values))
    gp = set(map(tuple, g[g.gold == "DIM"][KEY].values))
    dp = set(k for k in c.index[c.det_label == "DIM"] if k in tok)
    tp = len(gp & dp)
    return (tp / len(dp) if dp else float("nan")), (tp / len(gp) if gp else float("nan"))


def eval_part2(cand, p2, key2, pop, split, L, err_rows):
    s = p2[p2.split == split].merge(key2[["item_id", "group", "det_label"]].rename(columns={"det_label": "v2"}),
                                    left_on="id", right_on="item_id")
    c = cand.set_index(KEY)
    s["det"] = [c.det_label.get(tuple(k), "—") for k in s[KEY].values]
    s["det_base"] = [c.base_lemma.get(tuple(k)) for k in s[KEY].values]
    pp = pop[pop.split == split].set_index(["group", "det_label"])
    s["w"] = [pp.population.get((g, v), 0) / max(pp.sampled.get((g, v), 1), 1) for g, v in zip(s.group, s.v2)]
    L.append(f"### Часть 2. Кандидаты ({len(s)}; веса по стратам детектора v2)\n")
    acc = (s.w * (s.det.replace("—", "NO") == s.gold.replace("", "NO"))).sum() / s.w.sum()
    dim = s[s.det == "DIM"]
    Pw = (dim.w * (dim.gold == "DIM")).sum() / dim.w.sum() if len(dim) else float("nan")
    gdim = s[s.gold == "DIM"]
    Rw = (gdim.w * (gdim.det == "DIM")).sum() / gdim.w.sum() if len(gdim) else float("nan")
    Pu = wilson(int((dim.gold == "DIM").sum()), len(dim))
    L.append("| Мера | Значение |\n|---|---|")
    L.append(f"| Точность DIM, взвешенная | {f3(Pw)} |")
    L.append(f"| Точность DIM, по выборке (95% ДИ) | {f3(Pu[0])} ({int((dim.gold == 'DIM').sum())}/{len(dim)}; {f3(Pu[1])}–{f3(Pu[2])}) |")
    L.append(f"| Доля gold-DIM, которые детектор назвал DIM (среди кандидатов), взвешенная | {f3(Rw)} |")
    L.append(f"| Доля верных меток (5 классов), взвешенная | {f3(acc)} |")
    cm = pd.crosstab(s.det, s.gold.replace("", "—"))
    L.append("\nМатрица (строки — детектор, столбцы — gold; невзвешенная):\n")
    L.append("| детектор \\ gold | " + " | ".join(cm.columns) + " |\n|---|" + "---|" * len(cm.columns))
    for k, r in cm.iterrows():
        L.append(f"| {k} | " + " | ".join(str(v) for v in r.values) + " |")
    L.append("")
    for r in s[s.det.replace("—", "NO") != s.gold.replace("", "NO")].itertuples():
        err_rows.append({"split": split, "part": 2, "error": f"{r.det}→{r.gold or '—'}", "kidlit_id": r.kidlit_id,
                         "sent_id": r.sent_id, "tok_id": r.tok_id, "token": r.token, "gold": r.gold, "det": r.det,
                         "lemma": c.lemma.get((r.kidlit_id, r.sent_id, r.tok_id)), "base": r.det_base,
                         "rule": c.rule_id.get((r.kidlit_id, r.sent_id, r.tok_id)),
                         "context": c.context.get((r.kidlit_id, r.sent_id, r.tok_id))})
    return {"P_w": Pw, "P_u": Pu, "acc_w": acc, "n_dim": len(dim)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--gold", default="kidlit_dim/gold")
    ap.add_argument("--pilot", default="pilot_orig/pilot_gold.csv")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    gdir = Path(args.gold)
    cand = pd.read_parquet(args.cand)
    ver = ",".join(sorted(cand.detector_version.astype(str).unique()))
    h = hashlib.sha256(Path(args.cand).read_bytes()).hexdigest()[:16]
    p1, p2 = load_gold(gdir, args.pilot)
    key2 = pd.read_csv(gdir / "_key" / "candidates_sample_key_DO_NOT_SHARE.csv")
    pop = pd.read_csv(gdir / "_key" / "candidate_strata_population.csv")
    out = Path(args.out) if args.out else Path(args.cand).parent
    L = [f"# Оценка детектора {ver} на KidLit-Gold\n", f"Файл кандидатов: `{args.cand}` (sha256 {h}).\n"]
    splits = ["dev"] + (["test"] if args.test else [])
    if args.test:
        lock = gdir / "TEST_RUN.json"
        if lock.exists():
            prev = json.loads(lock.read_text())
            if prev["sha256_16"] != h:
                L.append(f"> ВНИМАНИЕ: test уже оценивался {prev['time']} на детекторе {prev['version']} (sha {prev['sha256_16']}). "
                         "Эти цифры НЕ являются независимой оценкой; в статью идут цифры первого прогона.\n")
        else:
            lock.write_text(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "version": ver,
                                        "sha256_16": h, "cand": str(args.cand)}, ensure_ascii=False, indent=1))
    err, summary = [], {}
    for sp in splits:
        L.append(f"## {sp}\n")
        r1 = eval_part1(cand, p1, sp, L, err)
        r2 = eval_part2(cand, p2, key2, pop, sp, L, err)
        summary[sp] = {"part1": {k: {"P": v[0][0], "R": v[1][0], "F1": v[2], "tp": v[3], "det": v[4], "gold": v[5]} for k, v in r1.items()},
                       "part2": {"P_dim_weighted": r2["P_w"], "P_dim_sample": r2["P_u"][0], "acc_weighted": r2["acc_w"]}}
    e = pd.DataFrame(err)
    name = "eval_" + "_".join(splits)
    e.to_csv(out / f"{name}_errors.csv", index=False)
    (out / f"{name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out / f"{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n".join(L))
    print(f"Ошибки: {out / (name + '_errors.csv')} ({len(e)})")


if __name__ == "__main__":
    main()
