#!/usr/bin/env python3
"""
KidLit by Numbers — шаг 2.3: согласие разметчиц, адъюдикация, золотые метки, оценка детектора.

    python kidlit_dim_gold_score.py --gold kidlit_dim/gold --filled kidlit_dim/gold/filled
    python kidlit_dim_gold_score.py ... --eval-test      # шаг 2.4: ОДИН раз, после доработки детектора на dev

1. Проверки: строки не потеряны, метки допустимы, в части 2 нет пустых меток, атрибуты стоят только там,
   где положено (pej/name/nonce — DIM, foreign — HYP), base заполнен для размеченных слов.
2. Согласие на двойной части: часть 1 — две книги (по строкам), часть 2 — 100 кандидатов.
3. Файл адъюдикации adjudication_2.3.xlsx (создаётся один раз; если в нём уже есть решения, не перезаписывается):
     label       — разные метки на двойной части;
     base        — одинаковая метка, разное производящее;
     attr        — одинаковая метка DIM/HYP, разные атрибуты;
     unsure      — разметчица отметила unsure = 1;
     consistency — одна и та же лемма в одиночной части размечена по-разному (у одной или у разных разметчиц).
   Решение — колонки final_*. Пустой final_label в части 1 = «не DIM/HYP/YOUNG».
4. Когда все final_label заполнены: gold_2.3.csv (золотые метки всех строк) и оценка детектора на dev.
   Test считается только с флагом --eval-test.
Исходные файлы разметчиц не изменяются: все решения живут только в файле адъюдикации.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

LABELS5 = ["DIM", "LEX", "HYP", "YOUNG", "NO"]
LABELS_FULL = ["DIM", "HYP", "YOUNG"]
KEY = ["kidlit_id", "sent_id", "tok_id"]
BIN = ["pej", "name", "nonce", "foreign"]
ADJ_COLS_FINAL = ["final_label", "final_base", "final_pej", "final_name", "final_nonce", "final_foreign",
                  "precedent", "decision_comment"]


# ---------------------------------------------------------------------------
def norm_base(x):
    return "" if pd.isna(x) else str(x).strip().lower().replace("ё", "е")


def read_filled(path, labels, idc):
    df = pd.read_excel(path, sheet_name="Разметка", dtype=str)
    df["label"] = df.label.fillna("").str.strip().str.upper()
    for c in BIN + ["unsure"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["precedent"] = (df.precedent.fillna("").str.strip().str.replace("Р", "P").str.replace("р", "P")
                       .str.replace(" ", ""))
    df["base"] = df.base.fillna("").str.strip()
    df["comment"] = df.comment.fillna("").str.strip()
    for c in ["sent_id", "tok_id"]:
        df[c] = df[c].astype(int)
    if idc == "row_id":
        df["row_id"] = df.row_id.astype(int)
    return df


def check(df, name, part, orig, idc):
    msgs = []
    if len(df) != len(orig) or set(df[idc]) != set(orig[idc]):
        sys.exit(f"{name}: строки не совпадают с выданным файлом")
    m = df.merge(orig[[idc, "token"]], on=idc, suffixes=("", "_orig"))
    if (m.token.fillna("") != m.token_orig.fillna("")).any():
        sys.exit(f"{name}: строки перепутаны (token не совпадает с исходным для того же {idc})")
    labels = LABELS_FULL if part == 1 else LABELS5
    bad = df[(df.label != "") & ~df.label.isin(labels)]
    if len(bad):
        sys.exit(f"{name}: недопустимые метки: {bad[idc].tolist()[:10]}")
    if part == 2 and (df.label == "").any():
        msgs.append(f"{name}: пустые метки в {(df.label == '').sum()} строках")
    lab = df.label != ""
    nb = df[lab & (df.base == "")]
    if len(nb):
        msgs.append(f"{name}: нет base у {len(nb)} размеченных слов")
    for c in ["pej", "name", "nonce"]:
        x = df[(df[c] == 1) & (df.label != "DIM")]
        if len(x):
            msgs.append(f"{name}: {c}=1 не у DIM в {len(x)} строках ({', '.join(x.token.head(5))})")
    x = df[(df.foreign == 1) & (df.label != "HYP")]
    if len(x):
        msgs.append(f"{name}: foreign=1 не у HYP в {len(x)} строках")
    return msgs


def kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def pos_agree(a, b, lab):
    sa, sb = int((a == lab).sum()), int((b == lab).sum())
    both = int(((a == lab) & (b == lab)).sum())
    return (2 * both / (sa + sb) if sa + sb else float("nan")), sa, sb, both


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


# ---------------------------------------------------------------------------
def build(args):
    gold, filled = Path(args.gold), Path(args.filled)
    man = json.loads((gold / "gold_manifest.json").read_text(encoding="utf-8"))
    split_book = {b: sp for sp, bl in man["full_reading"].items() for b in bl}
    P1, P2, msgs = {}, {}, []
    for a in ["A", "B"]:
        o1 = pd.read_excel(gold / f"gold_{a}_part1_full_reading.xlsx", sheet_name="Разметка", dtype=str)
        o1["row_id"] = o1.row_id.astype(int)
        o2 = pd.read_excel(gold / f"gold_{a}_part2_candidates.xlsx", sheet_name="Разметка", dtype=str)
        P1[a] = read_filled(filled / f"gold_{a}_part1_full_reading.xlsx", LABELS_FULL, "row_id")
        P2[a] = read_filled(filled / f"gold_{a}_part2_candidates.xlsx", LABELS5, "item_id")
        msgs += check(P1[a], f"{a}, часть 1", 1, o1, "row_id")
        msgs += check(P2[a], f"{a}, часть 2", 2, o2, "item_id")
        # правки в нередактируемых колонках (лемма и т. п.) — только сообщаем
        m = P2[a].merge(o2[["item_id", "lemma"]], on="item_id", suffixes=("", "_orig"))
        ch = m[m.lemma.fillna("") != m.lemma_orig.fillna("")]
        for r in ch.itertuples():
            msgs.append(f"{a}, часть 2: лемма {r.item_id} исправлена разметчицей: {r.lemma_orig} → {r.lemma} (учтём при доработке лемматизации)")
    for a in ["A", "B"]:
        P1[a]["split"] = P1[a].kidlit_id.map(split_book)
    key2 = pd.read_csv(gold / "_key" / "candidates_sample_key_DO_NOT_SHARE.csv")
    meta2 = key2[["item_id", "split", "group", "double", "annotator"]]
    for a in ["A", "B"]:
        P2[a] = P2[a].merge(meta2, on="item_id")
    return man, P1, P2, msgs


def iaa(P1, P2, man):
    L = []
    dbl = man["double_books"]
    A = P1["A"][P1["A"].kidlit_id.isin(dbl)].set_index("row_id")
    B = P1["B"][P1["B"].kidlit_id.isin(dbl)].set_index("row_id").loc[A.index]
    la, lb = A.label.replace("", "—"), B.label.replace("", "—")
    L.append("## Часть 1. Сплошное чтение, двойная разметка\n")
    L.append(f"Книги: {', '.join(dbl)}; слов: {len(A)}.\n")
    L.append("| Мера | Все | " + " | ".join(dbl) + " |\n|---|---|" + "---|" * len(dbl))
    rows = [("κ, 4 класса (—/DIM/HYP/YOUNG)", lambda x, y: f3(kappa(x, y))),
            ("κ, DIM / не DIM", lambda x, y: f3(kappa(x == "DIM", y == "DIM")))]
    for lab in LABELS_FULL:
        rows.append((f"Положительное согласие F1, {lab}",
                     lambda x, y, lab=lab: "{} (A {}, B {}, общих {})".format(f3(pos_agree(x, y, lab)[0]), *pos_agree(x, y, lab)[1:])))
    for name, fn in rows:
        cells = [fn(la, lb)] + [fn(la[A.kidlit_id == b], lb[A.kidlit_id == b]) for b in dbl]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    both = (A.label == B.label) & (A.label != "")
    if both.any():
        L.append(f"\nСовпадение base, где метки совпали: {f3((A.base[both].map(norm_base) == B.base[both].map(norm_base)).mean())} "
                 f"({both.sum()} слов).")
    L.append("\nκ по всем словам завышен: почти все слова не диминутивы и совпадают тривиально; "
             "основная мера для сплошного чтения — положительное согласие F1.\n")

    A2 = P2["A"][P2["A"].double].set_index("item_id").sort_index()
    B2 = P2["B"][P2["B"].double].set_index("item_id").loc[A2.index]
    L.append("## Часть 2. Кандидаты, двойная разметка\n")
    L.append(f"Кандидатов: {len(A2)} (dev {int((A2.split == 'dev').sum())}, test {int((A2.split == 'test').sum())}).\n")
    L.append("| Мера | Все | dev | test | RUS-O | RUS-T |\n|---|---|---|---|---|---|")
    masks = [slice(None), A2.split == "dev", A2.split == "test", A2.group == "RUS-O", A2.group == "RUS-T"]
    L.append("| κ, 5 меток | " + " | ".join(f3(kappa(A2.label[m], B2.label[m])) for m in masks) + " |")
    L.append("| κ, DIM / не DIM | " + " | ".join(f3(kappa(A2.label[m] == "DIM", B2.label[m] == "DIM")) for m in masks) + " |")
    L.append("| Доля совпадений | " + " | ".join(f3((A2.label[m] == B2.label[m]).mean()) for m in masks) + " |")
    bd = (A2.label == B2.label) & A2.label.isin(["DIM", "LEX", "HYP", "YOUNG"])
    L.append(f"\nСовпадение base, где метки совпали (DIM/LEX/HYP/YOUNG): {f3((A2.base[bd].map(norm_base) == B2.base[bd].map(norm_base)).mean())} ({bd.sum()}).")
    dd = (A2.label == "DIM") & (B2.label == "DIM")
    L.append("Атрибуты, где обе поставили DIM: " + ", ".join(
        f"{c} — совпадение {f3((A2[c][dd] == B2[c][dd]).mean())}" for c in ["pej", "name", "nonce"]) + ".\n")
    cm = pd.crosstab(A2.label, B2.label)
    L.append("Матрица (строки — A, столбцы — B):\n")
    L.append("| A \\ B | " + " | ".join(cm.columns) + " |\n|---|" + "---|" * len(cm.columns))
    for k, r in cm.iterrows():
        L.append(f"| {k} | " + " | ".join(str(v) for v in r.values) + " |")
    k5 = kappa(A2.label, B2.label)
    L.append("\nОриентир инструкции: κ ≥ 0,8." + (" Выполнен." if k5 >= 0.8 else " Не выполнен — разобрать расхождения."))
    return L, (A, B, A2, B2)


def adjudication_rows(P1, P2, man, pairs):
    A, B, A2, B2 = pairs
    rows = []

    def add(typ, part, idv, a, b, extra=None):
        r = {"type": typ, "part": part, "id": idv, "split": (a if a is not None else b)["split"],
             "kidlit_id": (a if a is not None else b)["kidlit_id"], "token": (a if a is not None else b)["token"],
             "lemma": (a if a is not None else b)["lemma"], "context": (a if a is not None else b)["context"],
             "context_more": (a if a is not None else b).get("context_more", "")}
        for who, x in (("A", a), ("B", b)):
            if x is None:
                r.update({f"label_{who}": "", f"base_{who}": "", f"attrs_{who}": "", f"prec_{who}": "", f"comment_{who}": ""})
            else:
                r.update({f"label_{who}": x["label"] or "—", f"base_{who}": x["base"],
                          f"attrs_{who}": " ".join(c for c in BIN if x[c] == 1),
                          f"prec_{who}": x["precedent"], f"comment_{who}": x["comment"]})
        if extra:
            r.update(extra)
        rows.append(r)

    # двойная часть
    for part, X, Y in ((1, A, B), (2, A2, B2)):
        for i in X.index:
            a, b = X.loc[i], Y.loc[i]
            if a.label != b.label:
                add("label", part, i, a, b)
            elif a.label and norm_base(a.base) != norm_base(b.base):
                add("base", part, i, a, b)
            elif a.label in ("DIM", "HYP") and any(a[c] != b[c] for c in BIN):
                add("attr", part, i, a, b)
    # unsure в одиночной части
    dbl_rows = set(A.index)
    for a_ in ["A", "B"]:
        x1 = P1[a_][(P1[a_].unsure == 1) & ~P1[a_].row_id.isin(dbl_rows)]
        for _, r in x1.iterrows():
            add("unsure", 1, r.row_id, r if a_ == "A" else None, r if a_ == "B" else None)
        x2 = P2[a_][(P2[a_].unsure == 1) & ~P2[a_].double]
        for _, r in x2.iterrows():
            add("unsure", 2, r.item_id, r if a_ == "A" else None, r if a_ == "B" else None)
    # согласованность по леммам в одиночной части (часть 2 — 5 меток; часть 1 — помеченное vs непомеченное не сравниваем)
    s2 = pd.concat([P2["A"][~P2["A"].double].assign(who="A"), P2["B"][~P2["B"].double].assign(who="B")])
    done = {(r["part"], r["id"]) for r in rows}
    for lem, g in s2.groupby(s2.lemma.str.lower().str.replace("ё", "е")):
        if g.label.nunique() > 1:
            for _, r in g.iterrows():
                if (2, r.item_id) in done:
                    continue
                add("consistency", 2, r.item_id, r if r.who == "A" else None, r if r.who == "B" else None,
                    {"note": f"лемма «{lem}»: " + ", ".join(f"{w}:{l}" for w, l in g.groupby('who').label.agg(lambda s: '/'.join(sorted(set(s)))).items())})
    s1 = pd.concat([P1["A"][~P1["A"].row_id.isin(dbl_rows)].assign(who="A"),
                    P1["B"][~P1["B"].row_id.isin(dbl_rows)].assign(who="B")])
    s1 = s1[s1.label != ""]
    for lem, g in s1.groupby(s1.lemma.str.lower().str.replace("ё", "е")):
        if g.label.nunique() > 1:
            for _, r in g.iterrows():
                add("consistency", 1, r.row_id, r if r.who == "A" else None, r if r.who == "B" else None,
                    {"note": f"лемма «{lem}»: " + "/".join(sorted(set(g.label)))})
    adj = pd.DataFrame(rows)
    if "note" not in adj:
        adj["note"] = ""
    adj["note"] = adj.note.fillna("")
    order = {"label": 0, "base": 1, "attr": 2, "unsure": 3, "consistency": 4}
    adj = adj.sort_values(["type", "part", "lemma"], key=lambda s: s.map(order) if s.name == "type" else s).reset_index(drop=True)
    adj.insert(0, "adj_id", [f"J{i:03d}" for i in range(1, len(adj) + 1)])
    for c in ["suggestion", "suggestion_reason"] + ADJ_COLS_FINAL:
        adj[c] = ""
    return adj


def write_adj(adj, path):
    from openpyxl.styles import Alignment, Font, PatternFill
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        adj.to_excel(xw, sheet_name="Адъюдикация", index=False)
        ws = xw.sheets["Адъюдикация"]
        widths = {"adj_id": 7, "type": 11, "part": 5, "id": 7, "split": 6, "kidlit_id": 16, "token": 14, "lemma": 13,
                  "context": 60, "context_more": 40, "note": 28, "suggestion": 11, "suggestion_reason": 45,
                  "final_label": 10, "final_base": 13, "decision_comment": 35}
        fill = PatternFill("solid", fgColor="FFF8E1")
        for j, c in enumerate(adj.columns, start=1):
            L = ws.cell(1, j).column_letter
            ws.column_dimensions[L].width = widths.get(c, 9)
            ws.cell(1, j).font = Font(bold=True)
            for i in range(2, len(adj) + 2):
                if c in ("context", "context_more", "suggestion_reason", "note", "comment_A", "comment_B"):
                    ws.cell(i, j).alignment = Alignment(wrap_text=True, vertical="top")
                if c in ADJ_COLS_FINAL:
                    ws.cell(i, j).fill = fill
        ws.freeze_panes = "H2"
        ws.auto_filter.ref = ws.dimensions
        info = xw.book.create_sheet("Как заполнять", 0)
        info.column_dimensions["A"].width = 120
        for line in [
            "Адъюдикация KidLit-Gold, шаг 2.3",
            "",
            "type: label — разные метки на двойной части; base — разное производящее; attr — разные атрибуты;",
            "      unsure — разметчица сомневалась; consistency — одна лемма в одиночной части размечена по-разному.",
            "suggestion / suggestion_reason — предложение по инструкции v0.3 (без меток детектора). Это не решение.",
            "Решение — жёлтые колонки final_*: final_label (в части 1 пустая метка = «не DIM/HYP/YOUNG»), final_base,",
            "final_pej/final_name/final_nonce (для DIM), final_foreign (для HYP), precedent, decision_comment.",
            "Для строк consistency: если обе метки правильные (разные контексты), впишите в final_label ту же метку, что стоит у разметчицы.",
            "Новые правила — в журнал прецедентов (инструкция в Docs) и затем в dim_lexicon.csv.",
            "Файлы разметчиц не меняем.",
        ]:
            info.append([line])
        info["A1"].font = Font(bold=True)
        xw.book.active = 1


# ---------------------------------------------------------------------------
def apply_adjudication(P1, P2, adj, man):
    """Золотая метка для каждой строки. Двойная часть: совпадение или решение; одиночная: метка разметчицы или решение."""
    dec = {}
    for r in adj.itertuples():
        dec[(int(r.part), str(r.id))] = r
    g1 = pd.concat([P1["A"].assign(who="A"), P1["B"][~P1["B"].row_id.isin(P1["A"].row_id)].assign(who="B")])
    dbl = set(P1["A"].row_id) & set(P1["B"].row_id)
    g1["double"] = g1.row_id.isin(dbl)
    g2 = pd.concat([P2["A"].assign(who="A"), P2["B"][~P2["B"].item_id.isin(P2["A"].item_id)].assign(who="B")])
    out = []
    for part, g, idc in ((1, g1, "row_id"), (2, g2, "item_id")):
        for _, r in g.iterrows():
            d = dec.get((part, str(r[idc])))
            lab, base = r.label, r.base
            attrs = {c: r[c] for c in BIN}
            src = "agreed" if r.get("double", False) else f"single_{r.who}"
            if d is not None:
                fl = str(d.final_label).strip().upper()
                lab = "" if fl in ("", "—", "NAN") else fl
                base = d.final_base if str(d.final_base).strip() not in ("", "nan") else base
                for c in BIN:
                    v = str(getattr(d, f"final_{c}")).strip()
                    if v in ("0", "1"):
                        attrs[c] = int(v)
                src = f"adjudicated_{d.type}"
            out.append({"part": part, "id": r[idc], "split": r.split, "kidlit_id": r.kidlit_id, "sent_id": r.sent_id,
                        "tok_id": r.tok_id, "token": r.token, "gold": lab, "gold_base": base, **{f"gold_{c}": attrs[c] for c in BIN},
                        "source": src})
    return pd.DataFrame(out)


def eval_detector(gold, gdir, split):
    det_full = pd.read_csv(gdir / "_key" / "full_reading_detector_key_DO_NOT_SHARE.csv")
    det_samp = pd.read_csv(gdir / "_key" / "candidates_sample_key_DO_NOT_SHARE.csv")
    pop = pd.read_csv(gdir / "_key" / "candidate_strata_population.csv")
    L = [f"## Детектор на {split}\n"]
    g1 = gold[(gold.part == 1) & (gold.split == split)]
    d1 = det_full[det_full.split == split]
    tok = set(map(tuple, g1[KEY].values))
    det_pos = {lab: set(map(tuple, d1[d1.det_label == lab][KEY].values)) & tok for lab in LABELS_FULL}
    L.append("Часть 1 (сплошное чтение; книги без пилотных):\n")
    L.append("| Метка | Gold | Детектор | Верно | Точность | Полнота (95% ДИ) | F1 |\n|---|---|---|---|---|---|---|")
    for lab in LABELS_FULL:
        gp = set(map(tuple, g1[g1.gold == lab][KEY].values))
        tp = len(gp & det_pos[lab])
        P = wilson(tp, len(det_pos[lab]))
        R = wilson(tp, len(gp))
        f1 = 2 * P[0] * R[0] / (P[0] + R[0]) if P[0] + R[0] else float("nan")
        L.append(f"| {lab} | {len(gp)} | {len(det_pos[lab])} | {tp} | {f3(P[0])} | {f3(R[0])} ({f3(R[1])}–{f3(R[2])}) | {f3(f1)} |")
    gpd = set(map(tuple, g1[g1.gold == "DIM"][KEY].values))
    missed = g1[g1[KEY].apply(tuple, axis=1).isin(gpd - det_pos["DIM"])]
    fp = d1[d1[KEY].apply(tuple, axis=1).isin(det_pos["DIM"] - gpd)]
    L.append("\nПропуски DIM: " + (", ".join(f"{t} ({k})" for t, k in zip(missed.token, missed.kidlit_id)) or "нет"))
    L.append("Ложные DIM: " + (", ".join(f"{t} ({k})" for t, k in zip(fp.token, fp.kidlit_id)) or "нет") + "\n")

    g2 = gold[(gold.part == 2) & (gold.split == split)].merge(det_samp[["item_id", "group", "det_label"]], left_on="id", right_on="item_id")
    L.append("Часть 2 (кандидаты): точность по метке детектора.\n")
    L.append("| Метка детектора | n | Совпало с gold | Точность (95% ДИ) |\n|---|---|---|---|")
    for lab in LABELS5:
        x = g2[g2.det_label == lab]
        if len(x):
            p = wilson(int((x.gold == lab).sum()), len(x))
            L.append(f"| {lab} | {len(x)} | {int((x.gold == lab).sum())} | {f3(p[0])} ({f3(p[1])}–{f3(p[2])}) |")
    # взвешенная доля верных решений по всем кандидатам
    pp = pop[pop.split == split]
    acc = 0.0
    tot = pp.population.sum()
    for r in pp.itertuples():
        x = g2[(g2.group == r.group) & (g2.det_label == r.det_label)]
        if len(x):
            acc += r.population * (x.gold == r.det_label).mean()
    L.append(f"\nДоля верных меток детектора по всем кандидатам (взвешено по размерам страт): {f3(acc / tot)}.")
    cm = pd.crosstab(g2.det_label, g2.gold.replace("", "—"))
    L.append("\nМатрица (строки — детектор, столбцы — gold):\n")
    L.append("| детектор \\ gold | " + " | ".join(cm.columns) + " |\n|---|" + "---|" * len(cm.columns))
    for k, r in cm.iterrows():
        L.append(f"| {k} | " + " | ".join(str(v) for v in r.values) + " |")
    err = g2[g2.det_label != g2.gold]
    err.to_csv(gdir / f"detector_errors_{split}.csv", index=False)
    L.append(f"\nОшибки: detector_errors_{split}.csv ({len(err)}).\n")
    return L


def main():
    ap = argparse.ArgumentParser(description="KidLit: шаг 2.3 — согласие, адъюдикация, золотые метки")
    ap.add_argument("--gold", default="kidlit_dim/gold")
    ap.add_argument("--filled", default="kidlit_dim/gold/filled")
    ap.add_argument("--eval-test", action="store_true", help="оценить детектор на test (шаг 2.4, один раз)")
    args = ap.parse_args()
    gdir = Path(args.gold)
    man, P1, P2, msgs = build(args)
    L = ["# KidLit-Gold: согласие разметчиц и адъюдикация (шаг 2.3)\n", "## Проверка файлов\n"]
    L += [f"- {m}" for m in msgs] or ["- Замечаний нет."]
    L.append("")
    Li, pairs = iaa(P1, P2, man)
    L += Li
    adj_path = gdir / "adjudication_2.3.xlsx"
    adj_new = adjudication_rows(P1, P2, man, pairs)
    if adj_path.exists():
        adj = pd.read_excel(adj_path, sheet_name="Адъюдикация", dtype=str).fillna("")
        if set(zip(adj.part.astype(str), adj.id.astype(str))) != set(zip(adj_new.part.astype(str), adj_new.id.astype(str))):
            L.append("\n> ВНИМАНИЕ: набор строк для адъюдикации изменился по сравнению с файлом adjudication_2.3.xlsx. "
                     "Файлы разметчиц менялись после выдачи адъюдикации?\n")
    else:
        write_adj(adj_new, adj_path)
        adj = adj_new.astype(str)
    L.append("\n## Адъюдикация\n")
    L.append("| Тип | Часть 1 | Часть 2 |\n|---|---|---|")
    for t in ["label", "base", "attr", "unsure", "consistency"]:
        L.append(f"| {t} | {int(((adj.type == t) & (adj.part.astype(str) == '1')).sum())} | {int(((adj.type == t) & (adj.part.astype(str) == '2')).sum())} |")
    pending = adj[adj.final_label.str.strip() == ""]
    # в части 1 пустой final_label — допустимое решение, поэтому готовность определяем по decision_comment или final_label
    pending = adj[(adj.final_label.str.strip() == "") & (adj.decision_comment.str.strip() == "")]
    if len(pending):
        L.append(f"\nНе решено строк: {len(pending)}. Золотые метки и оценка детектора — после адъюдикации.")
    else:
        gold = apply_adjudication(P1, P2, adj, man)
        gold.to_csv(gdir / "gold_2.3.csv", index=False)
        L.append(f"\nАдъюдикация завершена. gold_2.3.csv: {len(gold)} строк.\n")
        L += eval_detector(gold, gdir, "dev")
        if args.eval_test:
            L += eval_detector(gold, gdir, "test")
            L.append("> Test посчитан. Детектор после этого не меняется; эти цифры идут в статью.")
    (gdir / "iaa_report_2.3.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
