#!/usr/bin/env python3
"""
KidLit by Numbers — шаг 2.1: золотая выборка диминутивов (dev / test) и таблицы разметки.

    python kidlit_dim_gold.py make --data kidlit_dim --out gold --seed 2026

Что делает
  1. Сплошное чтение (полнота). 8 книг KidLit-Gold, по 4 из RUS-O и RUS-T:
       dev  = 2 пилотные книги (уже размечены) + по одной длинной книге из каждой группы;
       test = по одной длинной и одной средней книге из каждой группы, хотя бы одна книга в стихах,
              объём RUS-O и RUS-T в test различается не больше чем в 1,5 раза.
     Книги, из которых взяты контрольные примеры инструкции и примеры для разработки детектора, не берутся.
  2. Кандидаты (точность). Остальные книги корпуса (без 8 книг сплошного чтения и без книг контрольных
     примеров) делятся ПО КНИГАМ на dev и test внутри каждой группы. Из каждой ячейки split × group
     берётся 100 кандидатов детектора, стратифицированно по метке детектора (квоты QUOTA).
     Итого 400 кандидатов: 200 dev, 200 test.
  3. Двойная разметка: 2 новые книги (одна dev, одна test, из разных групп) и 100 кандидатов
     (по 25 из каждой ячейки). Остальное делится между A и B.
  4. Таблицы для A и B (формат раздела 7 инструкции v0.3, с колонкой foreign). Метки детектора скрыты,
     ключ лежит отдельно в _key/ и разметчицам не передаётся.

Всё, что выбрано, записано в gold_manifest.json (seed, списки книг, квоты, размеры совокупностей).
Повторный запуск с тем же seed и теми же данными даёт ту же выборку.
"""

import argparse
import hashlib
import itertools
import json
import random
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

PILOT_BOOKS = ["RUS-O_medium_05", "RUS-T_short_03"]
# Книги, из которых взяты контрольные примеры инструкции / примеры для разработки детектора
EXCLUDED = ["RUS-O_long_06", "RUS-O_long_07", "RUS-O_long_08", "RUS-O_long_09", "RUS-O_long_15",
            "RUS-O_long_25", "RUS-O_long_26", "RUS-O_short_02", "RUS-O_medium_01"]
GROUPS = ["RUS-O", "RUS-T"]
QUOTA = {"RUS-O": {"DIM": 53, "HYP": 5, "LEX": 15, "NO": 15, "YOUNG": 12},
         "RUS-T": {"DIM": 58, "HYP": 0, "LEX": 15, "NO": 15, "YOUNG": 12}}   # в RUS-T детектор не дал ни одного HYP
DOUBLE_PER_CELL = 25
ANNOTATORS = ["A", "B"]

LABELS5 = ["DIM", "LEX", "HYP", "YOUNG", "NO"]
LABELS_FULL = ["DIM", "HYP", "YOUNG"]
KEY = ["kidlit_id", "sent_id", "tok_id"]
ANNOT_COLS = ["label", "base", "pej", "name", "nonce", "foreign", "precedent", "unsure", "comment"]
BIN_COLS = ["pej", "name", "nonce", "foreign", "unsure"]
HEAD_FILL = PatternFill("solid", fgColor="DCE6F1")
EDIT_FILL = PatternFill("solid", fgColor="FFF8E1")
ALT_FILL = PatternFill("solid", fgColor="F5F5F5")

HELP = {
    "part1": [
        "ЧАСТЬ 1. СПЛОШНОЕ ЧТЕНИЕ (полнота детектора)",
        "Читайте текст подряд, строка = слово. Контекст — предложение, слово выделено [скобками].",
        "Отметьте ВСЕ диминутивы (DIM), уменьшительные формы имён людей (HYP) и детёнышей (YOUNG).",
        "LEX и NO здесь НЕ ставим: пустая метка = «не DIM, не HYP, не YOUNG».",
        "base — по правилу P-17: DIM — ближайшее производящее; YOUNG — взрослое животное; HYP — полное имя.",
        "Атрибуты (0 или 1): pej / name / nonce — для DIM; foreign — только для HYP (иноязычное имя, P-14).",
        "Применили прецедент из журнала — впишите его ID в precedent (латинская P, например P-04).",
        "Сомневаетесь — всё равно поставьте лучшую метку, unsure = 1 и комментарий.",
        "Книги идут подряд; первыми стоят книги, которые размечаете не только вы, — начните с них.",
        "Заполняются только жёлтые колонки. Не удаляйте и не сортируйте строки (фильтр использовать можно).",
        "Работаем независимо: не обсуждаем примеры с коллегой, пока обе не сдали обе части.",
        "Сданный файл больше не меняем: решения по расхождениям записываются только в файл адъюдикации.",
    ],
    "part2": [
        "ЧАСТЬ 2. ПРОВЕРКА КАНДИДАТОВ (точность детектора)",
        "Каждая строка — слово, которое детектор счёл кандидатом, из разных книг корпуса. Метки детектора скрыты.",
        "Каждому слову поставьте ровно одну метку: DIM / LEX / HYP / YOUNG / NO. Пустых меток не оставляйте.",
        "Порядок решения — схема из раздела 4 инструкции (сверху вниз, первая подходящая метка).",
        "context — предложение со словом в [скобках]; context_more — соседние предложения, если нужно больше.",
        "base — для DIM, LEX, YOUNG, HYP по правилу P-17 (у LEX base = само слово).",
        "Атрибуты (0 или 1): pej / name / nonce — для DIM; foreign — только для HYP.",
        "Сомневаетесь — лучшая метка, unsure = 1 и комментарий.",
        "Заполняются только жёлтые колонки. Не удаляйте и не сортируйте строки (фильтр использовать можно).",
        "Сданный файл больше не меняем: решения по расхождениям записываются только в файл адъюдикации.",
    ],
}


def bracket(text, s0, start, end):
    a, b = start - s0, end - s0
    return (text[:a] + "[" + text[a:b] + "]" + text[b:]).replace("\n", " ")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Отбор
# ---------------------------------------------------------------------------
def pick_full_reading(books, rng):
    gold = books[books.id_source == "задан в таблице проекта"].set_index("kidlit_id")
    elig = gold[~gold.index.isin(EXCLUDED + PILOT_BOOKS)]
    by = {(g, L): sorted(elig[(elig.group == g) & (elig.corpus_type == f"{g}_{L}")].index)
          for g in GROUPS for L in ["long", "medium"]}
    for attempt in range(1, 1001):
        sel = {}
        for g in GROUPS:
            dev_long, test_long = rng.sample(by[(g, "long")], 2)
            sel[g] = {"dev": [dev_long], "test": [test_long, rng.choice(by[(g, "medium")])]}
        test = sel["RUS-O"]["test"] + sel["RUS-T"]["test"]
        w = {g: gold.loc[sel[g]["test"], "n_words_counted"].sum() for g in GROUPS}
        verse = (gold.loc[test, "genre"] == "стихи").any()
        if verse and max(w.values()) / min(w.values()) <= 1.5:
            return sel, attempt
    sys.exit("Не удалось подобрать книги под ограничения")


def assign_books(new_books, sel, words, rng):
    """Двойная разметка: одна новая dev-книга и одна test-книга из другой группы. Остальные — A/B поровну по словам."""
    g_dev = rng.choice(GROUPS)
    g_test = [g for g in GROUPS if g != g_dev][0]
    double = [sel[g_dev]["dev"][0], rng.choice(sel[g_test]["test"])]
    rest = [b for b in new_books if b not in double]
    best = None
    for combo in itertools.combinations(rest, len(rest) // 2):
        a = list(combo)
        b = [x for x in rest if x not in a]
        diff = abs(sum(words[x] for x in a) - sum(words[x] for x in b))
        key = (diff, rng.random())
        if best is None or key < best[0]:
            best = (key, a, b)
    return double, {"A": best[1], "B": best[2]}


def split_pool(books, cand, rng):
    used = set(EXCLUDED) | set(PILOT_BOOKS)
    pool = books[~books.kidlit_id.isin(used)]
    split = {}
    hyp = cand[cand.det_label == "HYP"].kidlit_id.value_counts()
    for g in GROUPS:
        ids = sorted(pool[pool.group == g].kidlit_id)
        # Книги с HYP редки (HYP сосредоточены в нескольких книгах) — раскладываем их по очереди в dev и test,
        # иначе в одной из частей HYP может не оказаться вовсе
        h = sorted([b for b in ids if b in hyp.index], key=lambda b: (-hyp[b], b))
        rest = [b for b in ids if b not in hyp.index]
        rng.shuffle(rest)
        first = rng.choice(["dev", "test"])
        other = "test" if first == "dev" else "dev"
        for i, b in enumerate(h):
            split[b] = first if i % 2 == 0 else other
        half = len(ids) // 2
        for b in rest:
            n_dev = sum(1 for x in ids if split.get(x) == "dev")
            split[b] = "dev" if n_dev < half else "test"
    return split


def sample_candidates(cand, split, seed):
    c = cand[cand.kidlit_id.isin(split)].copy()
    c["split"] = c.kidlit_id.map(split)
    parts, pop = [], []
    for sp in ["dev", "test"]:
        for g in GROUPS:
            cell = c[(c.split == sp) & (c.group == g)]
            quota = dict(QUOTA[g])
            short = 0
            for lab in quota:
                n = int((cell.det_label == lab).sum())
                if n < quota[lab]:
                    print(f"[2.1] {sp}/{g}/{lab}: в совокупности {n} < квоты {quota[lab]}, недостача уходит в DIM")
                    short += quota[lab] - n
                    quota[lab] = n
            quota["DIM"] += short
            for lab, q in quota.items():
                stratum = cell[cell.det_label == lab]
                pop.append({"split": sp, "group": g, "det_label": lab, "population": len(stratum), "sampled": min(q, len(stratum))})
                if q == 0 or len(stratum) == 0:
                    continue
                parts.append(stratum.sample(n=min(q, len(stratum)), random_state=seed))
    s = pd.concat(parts)
    s = s.sample(frac=1, random_state=seed + 1).reset_index(drop=True)          # перемешать: порядок не выдаёт метку
    s["item_id"] = [f"C{i:03d}" for i in range(1, len(s) + 1)]
    return s, pd.DataFrame(pop)


def assign_candidates(s, rng):
    s["double"] = False
    s["annotator"] = ""
    for (sp, g), cell in s.groupby(["split", "group"]):
        ids = list(cell.item_id)
        rng.shuffle(ids)
        dbl, rest = ids[:DOUBLE_PER_CELL], ids[DOUBLE_PER_CELL:]
        s.loc[s.item_id.isin(dbl), ["double", "annotator"]] = [True, "AB"]
        for i, it in enumerate(rest):
            s.loc[s.item_id == it, "annotator"] = ANNOTATORS[i % 2]
    return s


# ---------------------------------------------------------------------------
# Таблицы
# ---------------------------------------------------------------------------
def write_sheet(path, rows, cols, labels, part, title_line):
    wb = Workbook()
    ws = wb.active
    ws.title = "Разметка"
    ws.append(cols)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = HEAD_FILL
    prev, shade = None, False
    for r in rows.itertuples(index=False):
        ws.append([None if c in ANNOT_COLS else getattr(r, c) for c in cols])
        sk = (r.kidlit_id, r.sent_id) if part == "part1" else r.item_id
        if sk != prev:
            shade, prev = not shade, sk
        i = ws.max_row
        for j, cname in enumerate(cols, start=1):
            if cname in ANNOT_COLS:
                ws.cell(i, j).fill = EDIT_FILL
            elif shade:
                ws.cell(i, j).fill = ALT_FILL
        for cname in ("context", "context_more"):
            if cname in cols:
                ws.cell(i, cols.index(cname) + 1).alignment = Alignment(wrap_text=True, vertical="top")
    n = ws.max_row
    col = {c: ws.cell(1, k + 1).column_letter for k, c in enumerate(cols)}
    dv = DataValidation(type="list", formula1='"' + ",".join(labels) + '"', allow_blank=True,
                        showErrorMessage=True, errorTitle="Метка", error="Допустимо: " + ", ".join(labels))
    ws.add_data_validation(dv)
    dv.add(f"{col['label']}2:{col['label']}{n}")
    dv01 = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
    ws.add_data_validation(dv01)
    for c in BIN_COLS:
        dv01.add(f"{col[c]}2:{col[c]}{n}")
    widths = {"row_id": 7, "item_id": 7, "kidlit_id": 17, "book_id": 10, "sent_id": 7, "tok_id": 6, "token": 16,
              "lemma": 16, "context": 70, "context_more": 45, "label": 9, "base": 14, "pej": 5, "name": 6,
              "nonce": 7, "foreign": 7, "precedent": 10, "unsure": 7, "comment": 30}
    for c, w in widths.items():
        if c in col:
            ws.column_dimensions[col[c]].width = w
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = f"A1:{col[cols[-1]]}{n}"

    info = wb.create_sheet("Инструкция", 0)
    info.column_dimensions["A"].width = 115
    info.append([title_line + f" Строк: {n - 1}."])
    info["A1"].font = Font(bold=True)
    info.append([""])
    for line in HELP[part]:
        info.append([line])
    info["A3"].font = Font(bold=True)
    info.append([""])
    info.append(["Метки: DIM — живой диминутив; LEX — лексикализованный; HYP — уменьшительная форма имени человека;"])
    info.append(["YOUNG — детёныш на -онок/-ёнок (-ата/-ята); NO — суффикс только похож на диминутивный."])
    info.append(["Правила и журнал прецедентов P-01–P-18 — в «Инструкции по разметке диминутивов KidLit v0.3»."])
    wb.active = 1
    wb.save(path)


def cmd_make(args):
    data, out = Path(args.data), Path(args.out)
    (out / "_key").mkdir(parents=True, exist_ok=True)
    books = pd.read_csv(data / "data" / "books.csv")
    tok = pd.read_parquet(data / "data" / "tokens.parquet")
    sents = pd.read_parquet(data / "data" / "sentences.parquet")
    cand_path = data / "detector" / "candidates.parquet"
    cand = pd.read_parquet(cand_path)
    ver = sorted(cand.detector_version.unique())
    words = dict(zip(books.kidlit_id, books.n_words_counted))
    rng = random.Random(args.seed)

    # 1. Книги сплошного чтения
    sel, attempts = pick_full_reading(books, rng)
    new_books = [b for g in GROUPS for sp in ["dev", "test"] for b in sel[g][sp]]
    double_books, own_books = assign_books(new_books, sel, words, rng)
    split_full = {b: "dev" for b in PILOT_BOOKS}
    for g in GROUPS:
        for sp in ["dev", "test"]:
            for b in sel[g][sp]:
                split_full[b] = sp

    # 2. Кандидаты из остальных книг
    split_pool_books = split_pool(books[~books.kidlit_id.isin(new_books)], cand, rng)
    s, pop = sample_candidates(cand, split_pool_books, args.seed)
    s = assign_candidates(s, rng)

    # Контексты
    sidx = sents.set_index(["book_id", "sent_id"])

    def ctx(b, si, st, en):
        r = sidx.loc[(b, si)]
        return bracket(r.text, r.start, st, en)

    def ctx_more(b, si):
        parts = []
        for d in (-1, 1):
            k = (b, si + d)
            if k in sidx.index:
                parts.append(("← " if d < 0 else "→ ") + sidx.loc[k].text.replace("\n", " "))
        return "\n".join(parts)

    # Часть 1: строки
    w = tok[tok.kidlit_id.isin(new_books) & tok.is_word].copy()
    order_all = double_books + [b for a in ANNOTATORS for b in own_books[a]]
    w["order"] = w.kidlit_id.map({b: i for i, b in enumerate(order_all)})
    w = w.sort_values(["order", "sent_id", "tok_id"])
    w["context"] = [ctx(b, si, st, en) for b, si, st, en in zip(w.book_id, w.sent_id, w.start, w.end)]
    w["row_id"] = range(1, len(w) + 1)
    cols1 = ["row_id", "kidlit_id", "token", "lemma", "context"] + ANNOT_COLS + ["book_id", "sent_id", "tok_id"]

    # Часть 2: строки
    s["context"] = [ctx(b, si, st, en) for b, si, st, en in zip(s.book_id, s.sent_id, s.start, s.end)]
    s["context_more"] = [ctx_more(b, si) for b, si in zip(s.book_id, s.sent_id)]
    s = s.sort_values("item_id")
    cols2 = ["item_id", "kidlit_id", "token", "lemma", "context", "context_more"] + ANNOT_COLS + ["book_id", "sent_id", "tok_id"]

    files = {}
    for a in ANNOTATORS:
        bl = double_books + own_books[a]
        rows1 = w[w.kidlit_id.isin(bl)]
        p1 = out / f"gold_{a}_part1_full_reading.xlsx"
        write_sheet(p1, rows1, cols1, LABELS_FULL, "part1",
                    f"KidLit-Gold, шаг 2.2, разметчица {a}. Сплошное чтение. Книги: {', '.join(bl)}.")
        rows2 = s[s.annotator.str.contains(a)]
        p2 = out / f"gold_{a}_part2_candidates.xlsx"
        write_sheet(p2, rows2, cols2, LABELS5, "part2", f"KidLit-Gold, шаг 2.2, разметчица {a}. Проверка кандидатов.")
        files[a] = {"part1": p1.name, "part1_books": bl, "part1_words": int(len(rows1)),
                    "part2": p2.name, "part2_items": int(len(rows2))}

    # Ключи (не передаются разметчицам)
    kcols = ["det_label", "base_lemma", "rule_id", "needs_review", "pej", "name", "nonce", "cand_suffix", "detector_version"]
    full_key = cand[cand.kidlit_id.isin(new_books + PILOT_BOOKS)][KEY + ["token", "lemma"] + kcols].copy()
    full_key["split"] = full_key.kidlit_id.map(split_full)
    full_key = full_key.merge(w[KEY + ["row_id"]], on=KEY, how="left")
    full_key.to_csv(out / "_key" / "full_reading_detector_key_DO_NOT_SHARE.csv", index=False)
    w[KEY + ["row_id", "token"]].assign(split=w.kidlit_id.map(split_full)).to_csv(out / "_key" / "full_reading_tokens_index.csv", index=False)
    s[["item_id", "split", "group", "double", "annotator"] + KEY + ["token", "lemma"] + kcols] \
        .to_csv(out / "_key" / "candidates_sample_key_DO_NOT_SHARE.csv", index=False)
    pop.to_csv(out / "_key" / "candidate_strata_population.csv", index=False)

    # Таблица книг
    rows = []
    for b, sp in split_full.items():
        r = books.set_index("kidlit_id").loc[b]
        who = "уже размечена (пилот)" if b in PILOT_BOOKS else ("A+B" if b in double_books else
                                                                ("A" if b in own_books["A"] else "B"))
        rows.append({"kidlit_id": b, "split": sp, "role": "full_reading", "annotation": who,
                     "title": r.title, "genre": r.genre, "n_words": int(r.n_words_counted),
                     "n_candidates": int((cand.kidlit_id == b).sum()),
                     "det_DIM": int(((cand.kidlit_id == b) & (cand.det_label == "DIM")).sum())})
    for b, sp in sorted(split_pool_books.items()):
        rows.append({"kidlit_id": b, "split": sp, "role": "candidate_pool", "annotation": "",
                     "title": books.set_index("kidlit_id").loc[b].title, "genre": books.set_index("kidlit_id").loc[b].genre,
                     "n_words": int(words[b]), "n_candidates": int((cand.kidlit_id == b).sum()),
                     "n_sampled": int((s.kidlit_id == b).sum())})
    for b in EXCLUDED:
        rows.append({"kidlit_id": b, "split": "—", "role": "excluded (контрольные примеры)", "annotation": "",
                     "title": books.set_index("kidlit_id").loc[b].title, "n_words": int(words[b])})
    gb = pd.DataFrame(rows)
    gb.to_csv(out / "gold_books.csv", index=False)

    manifest = {
        "step": "2.1", "created": str(date.today()), "seed": args.seed, "selection_attempts": attempts,
        "detector_version": ver, "candidates_sha256_16": sha(cand_path),
        "pilot_books_dev": PILOT_BOOKS, "excluded_books": EXCLUDED,
        "full_reading": {"dev": [b for b, sp in split_full.items() if sp == "dev"],
                         "test": [b for b, sp in split_full.items() if sp == "test"]},
        "double_books": double_books, "own_books": own_books,
        "candidate_pool": {sp: sorted(b for b, x in split_pool_books.items() if x == sp) for sp in ["dev", "test"]},
        "quota_per_cell": QUOTA, "double_per_cell": DOUBLE_PER_CELL,
        "n_candidates_sampled": int(len(s)), "n_double_candidates": int(s.double.sum()),
        "files": files,
        "rules": ["test full-reading: 1 long + 1 medium per group, >=1 verse book, RUS-O/RUS-T words ratio <= 1.5",
                  "candidate pool split by book within group (no book in both dev and test)",
                  "candidates stratified by detector label; per-label precision needs no weights; "
                  "overall figures are weighted by candidate_strata_population.csv"],
    }
    (out / "gold_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Короткий отчёт
    fr = gb[gb.role == "full_reading"]
    L = ["# KidLit-Gold: состав золотой выборки (шаг 2.1)\n",
         f"seed = {args.seed}, детектор {', '.join(ver)}, дата {date.today()}.\n",
         "## Сплошное чтение\n", "| Книга | split | Кто размечает | Жанр | Слов | Кандидатов (DIM) |", "|---|---|---|---|---|---|"]
    for r in fr.sort_values(["split", "kidlit_id"]).itertuples():
        L.append(f"| {r.kidlit_id} «{r.title}» | {r.split} | {r.annotation} | {r.genre} | {r.n_words} | {int(r.n_candidates)} ({int(r.det_DIM)}) |")
    for sp in ["dev", "test"]:
        x = fr[fr.split == sp]
        L.append(f"\n{sp}: {len(x)} книг, {x.n_words.sum()} слов, DIM детектора {int(x.det_DIM.sum())}." +
                 "  ".join(f" {g}: {x[x.kidlit_id.str.startswith(g)].n_words.sum()} слов." for g in GROUPS))
    L.append("\n## Кандидаты\n")
    pv = pd.crosstab([s.split, s.group], s.det_label)
    L.append("| split | group | " + " | ".join(pv.columns) + " | всего |\n|---|---|" + "---|" * (len(pv.columns) + 1))
    for (sp, g), r in pv.iterrows():
        L.append(f"| {sp} | {g} | " + " | ".join(str(v) for v in r.values) + f" | {r.sum()} |")
    L.append(f"\nКниг в пуле: dev {sum(v == 'dev' for v in split_pool_books.values())}, "
             f"test {sum(v == 'test' for v in split_pool_books.values())}; "
             f"кандидаты взяты из {s.kidlit_id.nunique()} книг. Двойная разметка: {int(s.double.sum())} кандидатов.\n")
    L.append("## Нагрузка\n\n| Разметчица | Часть 1: книги | Слов | Часть 2: кандидатов |\n|---|---|---|---|")
    for a in ANNOTATORS:
        f = files[a]
        L.append(f"| {a} | {', '.join(f['part1_books'])} | {f['part1_words']} | {f['part2_items']} |")
    (out / "gold_selection_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description="KidLit: золотая выборка диминутивов (шаг 2.1)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--data", default="kidlit_dim")
    m.add_argument("--out", default="gold")
    m.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    cmd_make(args)


if __name__ == "__main__":
    main()
