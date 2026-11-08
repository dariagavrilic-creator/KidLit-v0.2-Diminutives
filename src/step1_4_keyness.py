#!/usr/bin/env python3
"""
KidLit by Numbers — шаг 1.4: keyness v1.

Два сравнения на уровне лемм (лемма с ё → е + часть речи по тегам словаря Ляшевской–Шарова):
  1) KidLit (все книги) против частотного словаря Ляшевской–Шарова (НКРЯ);
  2) RUS-O против RUS-T.

Меры: log-likelihood (Rayson, Garside 2000) и log ratio (Hardie 2014).
Порог ключевого слова: LL ≥ 15.13 (p < 0.0001), |log ratio| ≥ 1, лемма минимум в 5 книгах.
Для RUS-O / RUS-T дополнительно — проверка на уровне книг (Mann–Whitney по частотам на 1000 слов,
поправка Бенджамини–Хохберга): одна длинная книга не должна «делать» ключевое слово группы.

Запуск:
    python kidlit_dim_keyness.py --data kidlit_dim

Результат (<data>/keyness/):
    kidlit_vs_ref_all.csv          все леммы KidLit с мерами
    kidlit_vs_ref_top100.xlsx      топ-100 ключевых слов KidLit — для разметки семантических классов
    rus_o_vs_rus_t_all.csv         все леммы с мерами для сравнения групп
    rus_o_vs_rus_t_keywords.xlsx   ключевые слова каждой группы (+ устойчивость по книгам) — для разметки классов
    keyness_report.md              сводка

Разметка семантических классов (колонка sem_class) делается ПОСЛЕ сдачи пилотной разметки.
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from scipy.stats import mannwhitneyu

RNC_CORPUS_SIZE = 92_000_000      # объём корпуса словаря (freqrnc_readme.txt)
LL_CRIT = 15.13                   # p < 0.0001, 1 степень свободы
LR_MIN = 1.0
MIN_BOOKS = 5
MIN_BOOKS_GROUP = 3               # для RUS-O / RUS-T: в группе-«владельце» ключевого слова
ZERO = 0.5                        # поправка на ноль для log ratio
SEM_CLASSES = ["еда", "животные", "тело", "дом и быт", "игрушки и игра", "природа и погода", "семья и люди",
               "эмоции и оценка", "звукоподражание", "транспорт и техника", "время", "грамматическое слово",
               "имя собственное", "современная лексика", "прочее"]
PROPER = {"Name", "Surn", "Patr", "Geox", "Orgn", "Trad"}


def e_norm(s):
    return s.replace("ё", "е") if isinstance(s, str) else s


def pos_key(pos, gram, pos_ud=None):
    """Часть речи pymorphy3 → тег словаря; имена собственные (s.PROP) объединены с s:
    «Мышка»-персонаж и «мышка» — одна лемма, имя отмечается отдельным флагом."""
    g = set(re.split(r"[ ,]", gram or ""))
    if pos == "NOUN":
        return "s"
    if pos in ("ADJF", "ADJS", "COMP"):
        if "Apro" in g:
            return "apro"
        if "Anum" in g:
            return "anum"
        return "adv" if (pos == "COMP" and pos_ud == "ADV") else "a"
    if pos in ("VERB", "INFN", "PRTF", "PRTS", "GRND"):
        return "v"
    if pos == "ADVB":
        return "advpro" if ("Ques" in g or "Dmns" in g) else "adv"
    if pos == "PRED":
        return "adv"
    return {"NPRO": "spro", "NUMR": "num", "PREP": "pr", "CONJ": "conj", "PRCL": "part", "INTJ": "intj"}.get(pos, "x")


def log_likelihood(a, b, c, d):
    """a, b — частоты в корпусах 1 и 2; c, d — их объёмы."""
    e1 = c * (a + b) / (c + d)
    e2 = d * (a + b) / (c + d)
    ll = 0.0
    if a > 0:
        ll += a * math.log(a / e1)
    if b > 0:
        ll += b * math.log(b / e2)
    return 2 * ll


def log_ratio(a, b, c, d):
    return math.log2(((a if a > 0 else ZERO) / c) / ((b if b > 0 else ZERO) / d))


def bh(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def write_xlsx(df, path, title_note):
    df = df.copy()
    df["sem_class"] = ""
    df["morph_check"] = ""
    df["comment"] = ""
    df.to_excel(path, index=False)
    wb = load_workbook(path)
    ws = wb.active
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DCE6F1")
    cols = [c.value for c in ws[1]]
    edit = PatternFill("solid", fgColor="FFF8E1")
    letters = {name: ws.cell(1, i + 1).column_letter for i, name in enumerate(cols)}
    n = ws.max_row
    for name in ["sem_class", "morph_check", "comment"]:
        for r in range(2, n + 1):
            ws[f"{letters[name]}{r}"].fill = edit
    dv = DataValidation(type="list", formula1='"' + ",".join(SEM_CLASSES) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{letters['sem_class']}2:{letters['sem_class']}{n}")
    dv2 = DataValidation(type="list", formula1='"DIM,YOUNG,HYP,LEX,—"', allow_blank=True)
    ws.add_data_validation(dv2)
    dv2.add(f"{letters['morph_check']}2:{letters['morph_check']}{n}")
    for name, wdt in {"lemma": 16, "example": 60, "sem_class": 20, "morph_check": 12, "comment": 30}.items():
        if name in letters:
            ws.column_dimensions[letters[name]].width = wdt
    ws.freeze_panes = "C2"
    info = wb.create_sheet("Как размечать", 0)
    info.column_dimensions["A"].width = 110
    for line in [title_note, "",
                 "sem_class — семантическое поле слова (для диминутива — поле производящего: домик → «дом и быт»).",
                 "morph_check — проверка морфологической метки детектора v1 (det_label_v1): подтвердить или исправить.",
                 "Колонку sem_class размечаем ПОСЛЕ сдачи пилотной разметки диминутивов.",
                 "Классы: " + ", ".join(SEM_CLASSES) + "."]:
        info.append([line])
    info["A1"].font = Font(bold=True)
    wb.active = 1
    wb.save(path)


def main():
    ap = argparse.ArgumentParser(description="KidLit: keyness v1 (шаг 1.4)")
    ap.add_argument("--data", default="kidlit_dim")
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()
    data = Path(args.data)
    out = data / "keyness"
    out.mkdir(exist_ok=True)

    tok = pd.read_parquet(data / "data" / "tokens.parquet")
    sents = pd.read_parquet(data / "data" / "sentences.parquet").set_index(["book_id", "sent_id"]).text
    w = tok[tok.is_word & tok.lemma.notna()].copy()
    w["key_lemma"] = w.lemma.str.lower().map(e_norm)
    w["key_pos"] = [pos_key(p, g, u) for p, g, u in zip(w.pos, w.gram, w.pos_ud)]
    w["mid_cap"] = w.is_cap & (w.tok_id > 0)
    w["proper_tag"] = w.gram.fillna("").map(lambda g: bool(set(re.split(r"[ ,]", g)) & PROPER))
    sizes = w.groupby("group").size()
    N_kid = len(w)

    # Метка детектора v1 по лемме (большинство употреблений)
    cand_path = data / "detector" / "candidates.parquet"
    det = {}
    if cand_path.exists():
        cand = pd.read_parquet(cand_path, columns=["lemma", "det_label"])
        cand["key_lemma"] = cand.lemma.str.lower().map(e_norm)
        det = cand.groupby("key_lemma").det_label.agg(lambda s: s.value_counts().index[0]).to_dict()

    g = (w.groupby(["key_lemma", "key_pos"])
           .agg(lemma=("lemma", lambda s: s.value_counts().index[0]), freq=("token", "size"),
                n_books=("book_id", "nunique"),
                freq_rus_o=("group", lambda s: int((s == "RUS-O").sum())),
                freq_rus_t=("group", lambda s: int((s == "RUS-T").sum())),
                books_rus_o=("book_id", lambda s: s[s.str.startswith("RUS-O")].nunique()),
                books_rus_t=("book_id", lambda s: s[s.str.startswith("RUS-T")].nunique()),
                cap_share=("mid_cap", "mean"), proper_share=("proper_tag", "mean"),
                ex_book=("book_id", "first"), ex_sent=("sent_id", "first"))
           .reset_index())
    g["is_proper"] = (g.proper_share >= 0.5) | (g.cap_share >= 0.8)
    g["example"] = [sents.loc[(b, s)].replace("\n", " ")[:200] for b, s in zip(g.ex_book, g.ex_sent)]
    g["det_label_v1"] = g.key_lemma.map(det).fillna("—")

    # Референс: тот же ключ (s.PROP → s), абсолютные частоты = ipm × 92 млн
    ref = pd.read_csv(data / "reference" / "freqrnc2011_clean.csv")
    ref["key_pos"] = ref.pos_rnc.replace({"s.PROP": "s"})
    ref_lp = ref.groupby(["lemma_e", "key_pos"]).approx_count.sum()
    ref_l = ref.groupby("lemma_e").approx_count.sum()
    rc, how = [], []
    for lem, pos in zip(g.key_lemma, g.key_pos):
        if (lem, pos) in ref_lp.index:
            rc.append(int(ref_lp[(lem, pos)])); how.append("lemma+pos")
        elif lem in ref_l.index:
            rc.append(int(ref_l[lem])); how.append("lemma_only")
        else:
            rc.append(0); how.append("none")
    g["ref_freq"], g["ref_match"] = rc, how

    # 1) KidLit против референса
    k = g.copy()
    k["ipm_kidlit"] = (k.freq / N_kid * 1e6).round(1)
    k["ipm_ref"] = (k.ref_freq / RNC_CORPUS_SIZE * 1e6).round(2)
    k["LL"] = [log_likelihood(a, b, N_kid, RNC_CORPUS_SIZE) for a, b in zip(k.freq, k.ref_freq)]
    k["log_ratio"] = [log_ratio(a, b, N_kid, RNC_CORPUS_SIZE) for a, b in zip(k.freq, k.ref_freq)]
    k["direction"] = np.where(k.log_ratio > 0, "KidLit", "референс")
    k["is_keyword"] = (k.LL >= LL_CRIT) & (k.log_ratio.abs() >= LR_MIN) & (k.n_books >= MIN_BOOKS)
    k = k.sort_values("LL", ascending=False)
    cols1 = ["lemma", "key_pos", "freq", "n_books", "ipm_kidlit", "ref_freq", "ipm_ref", "ref_match", "LL", "log_ratio",
             "direction", "is_keyword", "is_proper", "det_label_v1", "freq_rus_o", "freq_rus_t", "example"]
    k[cols1].round({"LL": 1, "log_ratio": 2}).to_csv(out / "kidlit_vs_ref_all.csv", index=False)
    pos_kw = k[k.is_keyword & (k.direction == "KidLit")]
    top = pos_kw.head(args.top)
    write_xlsx(top[cols1].round({"LL": 1, "log_ratio": 2}), out / "kidlit_vs_ref_top100.xlsx",
               f"Топ-{args.top} ключевых слов KidLit против словаря Ляшевской–Шарова (по LL, при LR ≥ 1 и ≥ {MIN_BOOKS} книгах)")

    # 2) RUS-O против RUS-T
    c, d = int(sizes["RUS-O"]), int(sizes["RUS-T"])
    r = g.copy()
    r["ipm_rus_o"] = (r.freq_rus_o / c * 1e6).round(1)
    r["ipm_rus_t"] = (r.freq_rus_t / d * 1e6).round(1)
    r["LL"] = [log_likelihood(a, b, c, d) for a, b in zip(r.freq_rus_o, r.freq_rus_t)]
    r["log_ratio"] = [log_ratio(a, b, c, d) for a, b in zip(r.freq_rus_o, r.freq_rus_t)]
    r["owner"] = np.where(r.log_ratio > 0, "RUS-O", "RUS-T")
    r["owner_books"] = np.where(r.owner == "RUS-O", r.books_rus_o, r.books_rus_t)
    r["is_keyword_ll"] = ((r.LL >= LL_CRIT) & (r.log_ratio.abs() >= LR_MIN) &
                          (r.n_books >= MIN_BOOKS) & (r.owner_books >= MIN_BOOKS_GROUP))
    # проверка на уровне книг для кандидатов в ключевые слова
    words_per_book = w.groupby(["book_id", "group"]).size().rename("n").reset_index()
    cnt = w.groupby(["key_lemma", "key_pos", "book_id"]).size().rename("k").reset_index()
    kw = r[r.is_keyword_ll][["key_lemma", "key_pos"]]
    pvals = []
    for lem, pos in zip(kw.key_lemma, kw.key_pos):
        x = words_per_book.merge(cnt[(cnt.key_lemma == lem) & (cnt.key_pos == pos)][["book_id", "k"]], on="book_id", how="left")
        x["rate"] = x.k.fillna(0) / x.n * 1000
        pvals.append(mannwhitneyu(x[x.group == "RUS-O"].rate, x[x.group == "RUS-T"].rate, alternative="two-sided").pvalue)
    r["p_books"], r["q_books"] = np.nan, np.nan
    if len(kw):
        idx = r.index[r.is_keyword_ll]
        r.loc[idx, "p_books"] = pvals
        r.loc[idx, "q_books"] = bh(pvals)
    r["robust"] = r.is_keyword_ll & (r.q_books < 0.05)
    r = r.sort_values("LL", ascending=False)
    cols2 = ["lemma", "key_pos", "owner", "freq_rus_o", "freq_rus_t", "ipm_rus_o", "ipm_rus_t", "books_rus_o", "books_rus_t",
             "LL", "log_ratio", "is_keyword_ll", "p_books", "q_books", "robust", "is_proper", "det_label_v1", "example"]
    r[cols2].round({"LL": 1, "log_ratio": 2, "p_books": 4, "q_books": 4}).to_csv(out / "rus_o_vs_rus_t_all.csv", index=False)
    write_xlsx(r[r.is_keyword_ll][cols2].round({"LL": 1, "log_ratio": 2, "p_books": 4, "q_books": 4}),
               out / "rus_o_vs_rus_t_keywords.xlsx",
               "Ключевые слова RUS-O и RUS-T (owner — группа, где слово частотнее; robust — подтверждено на уровне книг, q < 0.05)")

    # Отчёт
    L = ["# Keyness v1: сводка\n",
         f"KidLit: {N_kid} словоупотреблений (RUS-O {c}, RUS-T {d}), {len(g)} пар «лемма + часть речи». "
         f"Референс: словарь Ляшевской–Шарова, {RNC_CORPUS_SIZE / 1e6:.0f} млн словоупотреблений (частоты = ipm × 92 млн).\n",
         f"Порог: LL ≥ {LL_CRIT}, |log ratio| ≥ {LR_MIN}, лемма минимум в {MIN_BOOKS} книгах. Ноль заменяется на {ZERO} для log ratio.\n",
         "## 1. KidLit против словаря\n",
         f"Ключевых слов KidLit (частотнее, чем в словаре): {len(pos_kw)}; негативных (реже, чем в словаре): "
         f"{int((k.is_keyword & (k.direction == 'референс')).sum())}.\n"]
    for name, part in [(f"топ-{args.top}", top), ("все ключевые слова KidLit", pos_kw)]:
        vc = part.det_label_v1.value_counts()
        L.append(f"Морфологические метки детектора v1 ({name}, {len(part)} лемм): " +
                 ", ".join(f"{lab} — {vc.get(lab, 0)} ({vc.get(lab, 0) / max(len(part), 1):.0%})" for lab in ["DIM", "YOUNG", "HYP", "LEX", "NO", "—"]) +
                 f"; имена собственные — {int(part.is_proper.sum())}.")
    L.append("\nЭто черновые доли: метки детектора v1 ещё не проверены на золотой выборке; итоговая доля — после шага 2.4 "
             "и ручной проверки (колонка morph_check).\n")
    L.append(f"Топ-30: " + ", ".join(f"{x.lemma} ({x.det_label_v1})" if x.det_label_v1 != "—" else x.lemma for x in top.head(30).itertuples()) + "\n")
    L.append("Сопоставление со словарём для топ-ключевых слов: " +
             ", ".join(f"{kk} — {vv}" for kk, vv in top.ref_match.value_counts().items()) +
             ". Слова без пары в словаре (none) получают LL только за счёт частоты в KidLit — их стоит просмотреть отдельно.\n")
    L.append("## 2. RUS-O против RUS-T\n")
    for grp in ["RUS-O", "RUS-T"]:
        sub = r[r.is_keyword_ll & (r.owner == grp)]
        L.append(f"**{grp}:** ключевых по LL — {len(sub)}, из них устойчивы на уровне книг — {int(sub.robust.sum())}. "
                 "Топ-20: " + ", ".join(f"{x.lemma}{'*' if x.robust else ''}" for x in sub.head(20).itertuples()) + "\n")
    L.append("\\* — подтверждено на уровне книг (Mann–Whitney по частотам на 1000 слов, поправка Бенджамини–Хохберга, q < 0,05).\n")
    dim_o = r[r.det_label_v1 == "DIM"].freq_rus_o.sum() / c * 1000
    dim_t = r[r.det_label_v1 == "DIM"].freq_rus_t.sum() / d * 1000
    L.append(f"Справочно: употреблений лемм с меткой DIM детектора v1 — RUS-O {dim_o:.1f}, RUS-T {dim_t:.1f} на 1000 слов "
             "(не результат: детектор не оценён, а сравнение групп по диминутивам делается на шаге 3.2 на уровне книг).\n")
    L.append("## Ограничения\n")
    L.append("- Словарь построен на текстах 1950–2007 годов почти без детской литературы: ключевыми окажутся и просто "
             "современные слова (класс «современная лексика»).")
    L.append("- Лемматизация KidLit (pymorphy3) и словаря различается; сопоставление по лемме и части речи с откатом к лемме.")
    L.append("- Части речи у слов вне словаря pymorphy3 угаданы — у редких окказионализмов ключ «лемма + часть речи» может расщепиться.")
    (out / "keyness_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[1.4] Готово → {out}")


if __name__ == "__main__":
    main()
