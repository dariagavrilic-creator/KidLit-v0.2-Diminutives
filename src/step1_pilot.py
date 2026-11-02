#!/usr/bin/env python3
"""
KidLit by Numbers — шаг 1.3: пилотная разметка диминутивов и согласие разметчиков.

1) Сделать таблицы разметки (раздел 7 инструкции v0.2):
    python kidlit_dim_pilot.py make --data kidlit_dim --books RUS-O_medium_05 RUS-T_short_03 --out pilot

   Для каждого разметчика (A, B) создаются два файла:
     pilot_<X>_part1_full_reading.xlsx — сплошное чтение: все слова книг, размечаются DIM / HYP / YOUNG
     pilot_<X>_part2_candidates.xlsx   — проверка кандидатов детектора: каждому одна из пяти меток
   Метки детектора в файлах не показаны (разметка вслепую). Часть 2 выдаётся после того, как сдана часть 1:
   список кандидатов подсказывает, какие слова «подозрительные».

2) Посчитать согласие и собрать файл адъюдикации:
    python kidlit_dim_pilot.py score --data kidlit_dim --out pilot

   Читает заполненные файлы из папки pilot, пишет pilot/iaa_report.md и pilot/adjudication.xlsx.

3) После адъюдикации (заполнена колонка final_label в adjudication.xlsx) тот же score
   дополнительно собирает pilot/pilot_gold.csv и считает точность и полноту детектора на пилоте.
   Пилотные книги после этого относятся к dev-части золотой выборки (шаг 2.1), а не к test.
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

LABELS5 = ["DIM", "LEX", "HYP", "YOUNG", "NO"]
LABELS_FULL = ["DIM", "HYP", "YOUNG"]
KEY = ["kidlit_id", "sent_id", "tok_id"]
SCRIPT_COLS = ["row_id", "kidlit_id", "book_id", "sent_id", "tok_id", "token", "lemma", "context"]
ANNOT_COLS = ["label", "base", "pej", "name", "nonce", "precedent", "unsure", "comment"]
HEAD_FILL = PatternFill("solid", fgColor="DCE6F1")
EDIT_FILL = PatternFill("solid", fgColor="FFF8E1")
ALT_FILL = PatternFill("solid", fgColor="F5F5F5")

HELP = {
    "part1": [
        "ЧАСТЬ 1. СПЛОШНОЕ ЧТЕНИЕ (полнота детектора)",
        "Читайте текст подряд, строка = слово. Контекст — предложение, слово выделено [скобками].",
        "Отметьте ВСЕ диминутивы (DIM), уменьшительные формы имён людей (HYP) и детёнышей (YOUNG).",
        "LEX и NO здесь НЕ ставим: пустая метка = «не DIM, не HYP, не YOUNG».",
        "Для DIM заполните base (производящее) и атрибуты pej / name / nonce (0 или 1).",
        "Применили прецедент из журнала — впишите его ID в precedent (например, P-04).",
        "Сомневаетесь — всё равно поставьте лучшую метку, unsure = 1 и комментарий.",
        "Не меняйте колонки A–H, не удаляйте и не сортируйте строки.",
        "Работаем независимо: не обсуждаем примеры с коллегой до сдачи обеих частей.",
    ],
    "part2": [
        "ЧАСТЬ 2. ПРОВЕРКА КАНДИДАТОВ (точность детектора)",
        "Открывайте только после того, как сдана часть 1.",
        "Каждая строка — слово, которое детектор счёл кандидатом. Метки детектора скрыты.",
        "Каждому слову поставьте ровно одну метку: DIM / LEX / HYP / YOUNG / NO. Пустых меток не оставляйте.",
        "Порядок решения — схема из раздела 4 инструкции (сверху вниз, первая подходящая метка).",
        "base заполняется для DIM, LEX, YOUNG, HYP; атрибуты pej / name / nonce — только для DIM.",
        "Сомневаетесь — лучшая метка, unsure = 1 и комментарий.",
        "Не меняйте колонки A–H, не удаляйте и не сортируйте строки.",
    ],
}


# ---------------------------------------------------------------------------
# make
# ---------------------------------------------------------------------------
def bracket(sent_text, sent_start, start, end):
    a, b = start - sent_start, end - sent_start
    return (sent_text[:a] + "[" + sent_text[a:b] + "]" + sent_text[b:]).replace("\n", " ")


def write_sheet(path, rows: pd.DataFrame, labels, part, annotator, books):
    wb = Workbook()
    ws = wb.active
    ws.title = "Разметка"
    cols = SCRIPT_COLS + ANNOT_COLS
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = HEAD_FILL
    prev_sent, shade = None, False
    for r in rows.itertuples(index=False):
        ws.append([getattr(r, c) for c in SCRIPT_COLS] + [None] * len(ANNOT_COLS))
        sk = (r.kidlit_id, r.sent_id)
        if sk != prev_sent:
            shade, prev_sent = not shade, sk
        i = ws.max_row
        for j in range(1, len(SCRIPT_COLS) + 1):
            if shade:
                ws.cell(i, j).fill = ALT_FILL
        for j in range(len(SCRIPT_COLS) + 1, len(cols) + 1):
            ws.cell(i, j).fill = EDIT_FILL
        ws.cell(i, cols.index("context") + 1).alignment = Alignment(wrap_text=False)
    n = ws.max_row
    col = {c: chr(ord("A") + k) for k, c in enumerate(cols)}
    dv = DataValidation(type="list", formula1='"' + ",".join(labels) + '"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Метка", error="Допустимо: " + ", ".join(labels))
    ws.add_data_validation(dv)
    dv.add(f"{col['label']}2:{col['label']}{n}")
    dv01 = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
    ws.add_data_validation(dv01)
    for c in ["pej", "name", "nonce", "unsure"]:
        dv01.add(f"{col[c]}2:{col[c]}{n}")
    widths = {"row_id": 7, "kidlit_id": 17, "book_id": 10, "sent_id": 7, "tok_id": 6, "token": 16, "lemma": 16,
              "context": 90, "label": 9, "base": 14, "pej": 5, "name": 6, "nonce": 7, "precedent": 10,
              "unsure": 7, "comment": 30}
    for c, w in widths.items():
        ws.column_dimensions[col[c]].width = w
    ws.freeze_panes = "I2"
    ws.auto_filter.ref = f"A1:{col['comment']}{n}"

    info = wb.create_sheet("Инструкция", 0)
    info.column_dimensions["A"].width = 110
    info.append([f"Пилот KidLit, разметчик {annotator}. Книги: {', '.join(books)}. Строк: {n - 1}."])
    info["A1"].font = Font(bold=True)
    info.append([""])
    for line in HELP[part]:
        info.append([line])
    info.append([""])
    info.append(["Метки: DIM — живой диминутив; LEX — лексикализованный; HYP — уменьшительная форма имени человека;"])
    info.append(["YOUNG — детёныш на -онок/-ёнок (-ата/-ята); NO — суффикс только похож на диминутивный."])
    info.append(["Полные правила и журнал прецедентов — в «Инструкции по разметке диминутивов KidLit»."])
    info["A3"].font = Font(bold=True)
    wb.active = 1
    wb.save(path)


def cmd_make(args):
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = pd.read_parquet(data / "data" / "tokens.parquet")
    sents = pd.read_parquet(data / "data" / "sentences.parquet").set_index(["book_id", "sent_id"])
    cand = pd.read_parquet(data / "detector" / "candidates.parquet")
    missing = [b for b in args.books if b not in set(tok.kidlit_id)]
    if missing:
        sys.exit(f"Нет таких книг: {missing}")

    w = tok[tok.kidlit_id.isin(args.books) & tok.is_word].copy()
    w["order"] = w.kidlit_id.map({b: i for i, b in enumerate(args.books)})
    w = w.sort_values(["order", "sent_id", "tok_id"])
    w["context"] = [bracket(sents.loc[(b, s)].text, sents.loc[(b, s)].start, st, en)
                    for b, s, st, en in zip(w.book_id, w.sent_id, w.start, w.end)]
    w["row_id"] = range(1, len(w) + 1)

    c = cand[cand.kidlit_id.isin(args.books)].merge(w[KEY + ["order"]], on=KEY)
    c = c.sort_values(["order", "sent_id", "tok_id"]).copy()
    c["context"] = [bracket(sents.loc[(b, s)].text, sents.loc[(b, s)].start, st, en)
                    for b, s, st, en in zip(c.book_id, c.sent_id, c.start, c.end)]
    c["row_id"] = range(1, len(c) + 1)

    for a in args.annotators:
        write_sheet(out / f"pilot_{a}_part1_full_reading.xlsx", w, LABELS_FULL, "part1", a, args.books)
        write_sheet(out / f"pilot_{a}_part2_candidates.xlsx", c, LABELS5, "part2", a, args.books)
    # Ключ для подсчёта: какие строки — кандидаты и что сказал детектор (разметчикам не передаётся)
    c[KEY + ["row_id", "token", "lemma", "det_label", "base_lemma", "rule_id", "needs_review", "pej", "name", "nonce"]] \
        .to_csv(out / "_detector_key_DO_NOT_SHARE.csv", index=False)
    w[KEY + ["row_id", "token"]].to_csv(out / "_tokens_index.csv", index=False)
    print(f"[1.3] Книги: {', '.join(args.books)}. Часть 1: {len(w)} слов; часть 2: {len(c)} кандидатов. → {out}")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def read_filled(path, labels):
    df = pd.read_excel(path, sheet_name="Разметка", dtype={"label": str, "base": str, "precedent": str, "comment": str})
    df["label"] = df.label.fillna("").str.strip().str.upper()
    bad = df[(df.label != "") & ~df.label.isin(labels)]
    if len(bad):
        sys.exit(f"{path.name}: недопустимые метки в строках {bad.row_id.tolist()[:10]}")
    for c in ["pej", "name", "nonce", "unsure"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def cohen_kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in cats)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def fmt(x):
    return "—" if x != x else f"{x:.3f}"


def cmd_score(args):
    out = Path(args.out)
    L = ["# Пилот: согласие разметчиков\n"]
    adj_rows = []
    a1, b1 = out / "pilot_A_part1_full_reading.xlsx", out / "pilot_B_part1_full_reading.xlsx"
    a2, b2 = out / "pilot_A_part2_candidates.xlsx", out / "pilot_B_part2_candidates.xlsx"
    key = pd.read_csv(out / "_detector_key_DO_NOT_SHARE.csv")

    if a1.exists() and b1.exists():
        A, B = read_filled(a1, LABELS_FULL), read_filled(b1, LABELS_FULL)
        if len(A) != len(B) or (A.row_id.values != B.row_id.values).any():
            sys.exit("Часть 1: файлы A и B не совпадают по строкам (удалены или отсортированы строки?)")
        la, lb = A.label.replace("", "—"), B.label.replace("", "—")
        L.append("## Часть 1. Сплошное чтение\n")
        L.append(f"Слов: {len(A)}. Отмечено A: {(A.label != '').sum()}, B: {(B.label != '').sum()}.\n")
        L.append("| Мера | Значение |\n|---|---|")
        L.append(f"| κ по всем словам, 4 класса (—/DIM/HYP/YOUNG) | {fmt(cohen_kappa(la, lb))} |")
        L.append(f"| κ по всем словам, DIM / не DIM | {fmt(cohen_kappa(la == 'DIM', lb == 'DIM'))} |")
        for lab in LABELS_FULL:
            sa, sb = set(A.row_id[A.label == lab]), set(B.row_id[B.label == lab])
            pa = 2 * len(sa & sb) / (len(sa) + len(sb)) if (sa or sb) else float("nan")
            L.append(f"| Положительное согласие (F1 между A и B), {lab} | {fmt(pa)} (A {len(sa)}, B {len(sb)}, общих {len(sa & sb)}) |")
        L.append("\nκ по всем словам завышен: подавляющее большинство слов не диминутивы и совпадают тривиально. "
                 "Для статьи основная мера сплошного чтения — положительное согласие (F1).\n")
        dis = A[la.values != lb.values]
        for i in dis.index:
            adj_rows.append({"part": 1, "row_id": A.row_id[i], "kidlit_id": A.kidlit_id[i], "sent_id": A.sent_id[i],
                             "tok_id": A.tok_id[i], "token": A.token[i], "context": A.context[i],
                             "label_A": A.label[i], "base_A": A.base[i], "comment_A": A.comment[i],
                             "label_B": B.label[i], "base_B": B.base[i], "comment_B": B.comment[i]})
    else:
        L.append("## Часть 1\n\nФайлы части 1 ещё не заполнены.\n")

    if a2.exists() and b2.exists():
        A, B = read_filled(a2, LABELS5), read_filled(b2, LABELS5)
        if len(A) != len(B) or (A.row_id.values != B.row_id.values).any():
            sys.exit("Часть 2: файлы A и B не совпадают по строкам")
        empty = int((A.label == "").sum() + (B.label == "").sum())
        L.append("## Часть 2. Проверка кандидатов\n")
        L.append(f"Кандидатов: {len(A)}." + (f" Пустых меток: {empty} — их нужно заполнить." if empty else "") + "\n")
        L.append("| Мера | Значение |\n|---|---|")
        L.append(f"| κ, 5 меток | {fmt(cohen_kappa(A.label, B.label))} |")
        L.append(f"| κ, DIM / не DIM | {fmt(cohen_kappa(A.label == 'DIM', B.label == 'DIM'))} |")
        both_dim = (A.label == "DIM") & (B.label == "DIM")
        if both_dim.any():
            base_eq = (A.base[both_dim].fillna("").str.lower().str.replace("ё", "е").str.strip()
                       == B.base[both_dim].fillna("").str.lower().str.replace("ё", "е").str.strip()).mean()
            L.append(f"| Совпадение base, где оба поставили DIM | {base_eq:.3f} |")
            for att in ["pej", "name", "nonce"]:
                L.append(f"| κ атрибута {att} (где оба DIM) | {fmt(cohen_kappa(A[att][both_dim], B[att][both_dim]))} |")
        L.append("\nМатрица ошибок (строки — A, столбцы — B):\n")
        cm = pd.crosstab(A.label.replace("", "пусто"), B.label.replace("", "пусто"))
        L.append("| A \\ B | " + " | ".join(cm.columns) + " |\n|---|" + "---|" * len(cm.columns))
        for k, r in cm.iterrows():
            L.append(f"| {k} | " + " | ".join(str(v) for v in r.values) + " |")
        L.append("")
        dis = A[A.label.values != B.label.values]
        for i in dis.index:
            adj_rows.append({"part": 2, "row_id": A.row_id[i], "kidlit_id": A.kidlit_id[i], "sent_id": A.sent_id[i],
                             "tok_id": A.tok_id[i], "token": A.token[i], "context": A.context[i],
                             "label_A": A.label[i], "base_A": A.base[i], "comment_A": A.comment[i],
                             "label_B": B.label[i], "base_B": B.base[i], "comment_B": B.comment[i]})
        k5 = cohen_kappa(A.label, B.label)
        L.append("Ориентир инструкции: κ ≥ 0,8; при κ < 0,7 схема сужается до «DIM / не DIM»." +
                 ("" if k5 != k5 else (" Сейчас κ ≥ 0,8." if k5 >= 0.8 else
                                        " Сейчас κ между 0,7 и 0,8 — разобрать расхождения и уточнить инструкцию." if k5 >= 0.7 else
                                        " Сейчас κ < 0,7 — план Б.")) + "\n")
    else:
        L.append("## Часть 2\n\nФайлы части 2 ещё не заполнены.\n")

    # Файл адъюдикации (не перезаписываем, если там уже есть решения)
    adj_path = out / "adjudication.xlsx"
    if adj_rows:
        adj = pd.DataFrame(adj_rows)
        adj["final_label"], adj["final_base"], adj["precedent"], adj["decision_comment"] = "", "", "", ""
        if adj_path.exists():
            old = pd.read_excel(adj_path, dtype=str)
            if old.final_label.fillna("").str.strip().ne("").any():
                print(f"[1.3] {adj_path.name} уже содержит решения — не перезаписываю")
            else:
                adj.to_excel(adj_path, index=False)
        else:
            adj.to_excel(adj_path, index=False)
        L.append(f"## Адъюдикация\n\nРасхождений: {len(adj)} (часть 1 — {(adj.part == 1).sum()}, часть 2 — {(adj.part == 2).sum()}). "
                 f"Файл `{adj_path.name}`: заполните final_label (для части 1 пустая метка допустима — «не DIM/HYP/YOUNG»), "
                 "final_base, precedent и decision_comment. Новые решения переносятся в журнал прецедентов и в dim_lexicon.csv.\n")

    # Детектор на пилоте — только когда адъюдикация завершена
    if a2.exists() and b2.exists() and a1.exists() and b1.exists() and adj_path.exists():
        adj = pd.read_excel(adj_path, dtype=str).fillna("")
        need = adj[(adj.part.astype(int) == 2) & (adj.final_label.str.strip() == "")]
        if len(need) == 0 and len(adj):
            A1 = read_filled(a1, LABELS_FULL)
            A2 = read_filled(a2, LABELS5)
            fin1 = dict(zip(A1.row_id, A1.label))
            fin2 = dict(zip(A2.row_id, A2.label))
            for r in adj.itertuples():
                tgt = fin1 if int(r.part) == 1 else fin2
                tgt[int(r.row_id)] = r.final_label.strip().upper()
            A1["gold"] = A1.row_id.map(fin1)
            A2["gold"] = A2.row_id.map(fin2)
            gold_dim = set(map(tuple, A1[A1.gold == "DIM"][KEY].values))
            key2 = key.merge(A2[KEY + ["gold"]], on=KEY)
            det_dim = set(map(tuple, key[key.det_label == "DIM"][KEY].values))
            tp_r = len(gold_dim & det_dim)
            p_k = int(((key2.det_label == "DIM") & (key2.gold == "DIM")).sum())
            p_n = int((key2.det_label == "DIM").sum())
            P = wilson(p_k, p_n)
            R = wilson(tp_r, len(gold_dim))
            f1 = 2 * P[0] * R[0] / (P[0] + R[0]) if P[0] + R[0] else float("nan")
            L.append("## Детектор на пилоте (после адъюдикации)\n")
            L.append("| Мера | Значение | 95% ДИ (Уилсон) |\n|---|---|---|")
            L.append(f"| Точность DIM (часть 2) | {fmt(P[0])} ({p_k}/{p_n}) | {fmt(P[1])}–{fmt(P[2])} |")
            L.append(f"| Полнота DIM (часть 1) | {fmt(R[0])} ({tp_r}/{len(gold_dim)}) | {fmt(R[1])}–{fmt(R[2])} |")
            L.append(f"| F1 | {fmt(f1)} | |")
            missed = A1[(A1.gold == "DIM") & ~A1[KEY].apply(tuple, axis=1).isin(det_dim)]
            if len(missed):
                L.append("\nПропущенные детектором DIM: " + ", ".join(f"{t} ({k})" for t, k in zip(missed.token, missed.kidlit_id)))
            L.append("\nЭто dev-оценка: по этим ошибкам дорабатывается детектор, итоговые цифры — только на test (шаг 2.4).\n")
            gold = pd.concat([A1.assign(part=1)[["part"] + KEY + ["token", "gold"]],
                              A2.assign(part=2)[["part"] + KEY + ["token", "gold"]]])
            gold.to_csv(out / "pilot_gold.csv", index=False)
    (out / "iaa_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[1.3] Отчёт: {out / 'iaa_report.md'}")


def main():
    ap = argparse.ArgumentParser(description="KidLit: пилотная разметка диминутивов (шаг 1.3)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--data", default="kidlit_dim")
    m.add_argument("--books", nargs="+", default=["RUS-O_medium_05", "RUS-T_short_03"])
    m.add_argument("--annotators", nargs="+", default=["A", "B"])
    m.add_argument("--out", default="pilot")
    s = sub.add_parser("score")
    s.add_argument("--data", default="kidlit_dim")
    s.add_argument("--out", default="pilot")
    args = ap.parse_args()
    cmd_make(args) if args.cmd == "make" else cmd_score(args)


if __name__ == "__main__":
    main()
