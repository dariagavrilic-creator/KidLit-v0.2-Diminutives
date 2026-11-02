#!/usr/bin/env python3
"""
KidLit by Numbers — статья о диминутивах (ISP RAS Open 2026).
Шаги 0.2–0.4 дорожной карты: заморозка корпуса, предобработка, референсный ресурс.

Запуск (все шаги):
    python kidlit_dim_step0.py --corpus corpus.csv --extra-texts kid_lit_all_ru.txt \
        --freq-zip Freq2011_original.zip --out kidlit_dim

Отдельные шаги:
    python kidlit_dim_step0.py ... --steps freeze
    python kidlit_dim_step0.py ... --steps preprocess,reference,report

Зависимости:
    pip install pandas pyarrow razdel pymorphy3 pymorphy3-dicts-ru spacy
    pip install https://github.com/explosion/spacy-models/releases/download/ru_core_news_sm-3.8.0/ru_core_news_sm-3.8.0-py3-none-any.whl

Структура результата (--out):
    data/raw_texts/<book_id>.txt     тексты как в источнике (CSV — байт в байт; из txt — с \r\n → \n)
    data/texts/<book_id>.txt         нормализованные тексты — с ними работают все следующие шаги
    data/books.csv                   метаданные, одна строка на книгу
    data/normalization_log.csv       каждое изменение текста: что, где, на что
    data/freeze_manifest.json        версия, дата, sha256 каждого файла
    data/tokens.parquet              одна строка на токен
    data/sentences.parquet           одна строка на предложение
    data/book_stats.csv              знаменатели для нормировок (токены, слова, существительные)
    reference/freqrnc2011_clean.csv  частотный словарь Ляшевской–Шарова в рабочем виде
    reference/kidlit_lemma_freq.csv  частоты лемм KidLit + сопоставление со словарём
    reference/check_top50.csv        проверка 50 самых частотных лемм KidLit
    reference/unmatched_top200.csv   самые частотные леммы KidLit, которых нет в словаре
    results/step0_report.md          отчёт по всем проверкам
"""

import argparse
import bisect
import datetime as dt
import hashlib
import io
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

CORPUS_VERSION = "KidLit v0.2"

# ---------------------------------------------------------------------------
# 0.2. Описание колонок исходного CSV (файл без заголовка; группы разделены
# строками-маркерами «РУ» и «ИНО» в первой колонке).
# Колонки с суффиксом _raw — назначение не подтверждено, имена временные.
# ---------------------------------------------------------------------------
COLUMNS = {
    0: "id",
    1: "title",
    2: "original_title",
    3: "author",
    4: "author_gender",
    5: "year",
    6: "language",
    7: "country_of_origin",
    8: "publisher",
    9: "age_marker",
    10: "pages",
    11: "illustrator",
    12: "УДК",
    13: "genre",
    14: "NER_annotation",
    15: "Sentiment__annotation",
    16: "NER_annotation_author",
    17: "Sentiment_annotation_author",
    18: "n_words",
    19: "awards",
    20: "is_translation",
    21: "translator",
    22: "description",
    23: "text",
}
GROUP_MARKERS = {"РУ": "RUS-O", "ИНО": "RUS-T"}
REQUIRED = ["title", "author", "year", "publisher", "age_marker", "genre"]

# ID KidLit: <группа>_<длина>_<NN>. 30 ID заданы в таблице проекта и не меняются
# (проверяются по началу названия). Остальным книгам ID присваиваются по правилу ниже.
FIXED_IDS = [
    ("RUS-O", 1, "RUS-O_medium_01", "Автомобильчик"),
    ("RUS-O", 2, "RUS-O_medium_02", "Кораблик, который хотел летать"),
    ("RUS-O", 4, "RUS-O_short_01", "Добрая акула"),
    ("RUS-O", 5, "RUS-O_short_02", "Как лисичка у зайчика"),
    ("RUS-O", 7, "RUS-O_long_01", "Обними меня!"),
    ("RUS-O", 8, "RUS-O_medium_03", "Принц Филипп"),
    ("RUS-O", 16, "RUS-O_short_03", "Жёлтый, Красный, Синий"),
    ("RUS-O", 17, "RUS-O_long_02", "Детский сад на колёсиках"),
    ("RUS-O", 21, "RUS-O_long_03", "Грузовик, прицеп"),
    ("RUS-O", 22, "RUS-O_medium_04", "Едет мамин человечек"),
    ("RUS-O", 19, "RUS-O_long_04", "Как же я люблю маму"),
    ("RUS-O", 27, "RUS-O_medium_05", "Подарок для мышки"),
    ("RUS-O", 28, "RUS-O_long_05", "Лисья сказка"),
    ("RUS-O", 31, "RUS-O_short_04", "Весёлый пожарный"),
    ("RUS-O", 32, "RUS-O_short_05", "Оркестр маленьких жучков"),
    ("RUS-T", 1, "RUS-T_medium_01", "Когда я злюсь"),
    ("RUS-T", 16, "RUS-T_long_01", "Как Лисенок встречал"),
    ("RUS-T", 6, "RUS-T_medium_02", "Маленький вертолёт"),
    ("RUS-T", 39, "RUS-T_medium_03", "Музей"),
    ("RUS-T", 12, "RUS-T_long_02", "Счастливый день с папой"),
    ("RUS-T", 17, "RUS-T_short_01", "Дом"),
    ("RUS-T", 4, "RUS-T_short_02", "Самая любимая мама"),
    ("RUS-T", 43, "RUS-T_medium_04", "Принц и Свинтус"),
    ("RUS-T", 21, "RUS-T_long_03", "Маленький красный трактор"),
    ("RUS-T", 22, "RUS-T_short_03", "Мышонок Тим"),
    ("RUS-T", 24, "RUS-T_medium_05", "Когда наступит Новый год"),
    ("RUS-T", 32, "RUS-T_long_04", "Счастье кота Феличе"),
    ("RUS-T", 33, "RUS-T_long_05", "Мамино сокровище"),
    ("RUS-T", 34, "RUS-T_short_04", "Бабушка и птица"),
    ("RUS-T", 45, "RUS-T_short_05", "Большое путешествие Горошка"),
]
# Правило для остальных книг — по числу слов в нормализованном тексте (как в tokens.parquet):
# short < 500, medium 500–999, long ≥ 1000. Оно воспроизводит 29 из 30 заданных ID
# (исключение — RUS-O_short_04, 565 слов; его ID не меняем). Номера продолжают ряд с 06
# внутри «группа × длина», по возрастанию id из CSV.
SHORT_MAX, MEDIUM_MAX = 499, 999

# Ручные исправления OCR-артефактов: латиница вместо кириллицы, служебная строка,
# разорванное слово. Каждое найдено при осмотре корпуса 12.10.2026.
# Формат: (book_id, было, стало, причина). Применяется str.replace, число замен логируется.
MANUAL_FIXES = [
    ("RUS-O_18", "KPAK", "КРАК", "латиница вместо кириллицы"),
    ("RUS-O_37", "Tcc!", "Тсс!", "латиница вместо кириллицы"),
    ("RUS-O_37", "Koro?!", "Кого?!", "латиница вместо кириллицы"),
    ("RUS-O_38", "Иссо-xшая", "Иссохшая", "латинская x и разрыв слова"),
    ("RUS-O_44", "Ara!", "Ага!", "латиница вместо кириллицы"),
    ("RUS-T_16", "\ntext\n", "\n", "служебная строка 'text' (артефакт выгрузки)"),
    ("RUS-T_38", "OTCTAHBTE OT MEHЯ", "ОТСТАНЬТЕ ОТ МЕНЯ", "латиница вместо кириллицы"),
    ("RUS-T_41", "A сегодня", "А сегодня", "латиница вместо кириллицы"),
    ("RUS-T_42", "P-P-P!", "Р-Р-Р!", "латиница вместо кириллицы"),
    ("RUS-T_43", "XPЮШКА", "ХРЮШКА", "латиница вместо кириллицы"),
    ("RUS-T_47", "БAM!", "БАМ!", "латиница вместо кириллицы"),
    ("RUS-T_49", "A B BOCKPECEHLE?", "А В ВОСКРЕСЕНЬЕ?", "латиница вместо кириллицы"),
    ("RUS-T_50", "Ppppppp!", "Ррррррр!", "латиница вместо кириллицы"),
]
# Латиница, которую оставляем намеренно (QC её не считает ошибкой).
ALLOWED_LATIN = {
    ("RUS-O_55", "h"),      # «перевернулись вот так h» — графическая игра в тексте
    ("RUS-O_06", "C"),      # «200 °C» в рецепте
    ("RUS-O_46", "Look"),   # английское слово в тексте («Look у Баблёли…»)
    ("RUS-O_46", "look"),   # английское слово в тексте (строчное)
}

WORD_RE = re.compile(r"[^\W\d_]", re.UNICODE)       # есть хотя бы одна буква
LATIN_RE = re.compile(r"[A-Za-z]")
CYR_RE = re.compile(r"[А-Яа-яЁё]")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def e_norm(s):
    return s.replace("ё", "е").replace("Ё", "Е") if isinstance(s, str) else s


def get_morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()


# ---------------------------------------------------------------------------
# Шаг 0.2. Заморозка корпуса и метаданных
# ---------------------------------------------------------------------------
def read_corpus_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False, na_values=[""])
    if raw.shape[1] != len(COLUMNS):
        sys.exit(f"Ожидалось {len(COLUMNS)} колонок, в файле {raw.shape[1]}. Проверьте формат CSV.")
    rows, group = [], None
    read_corpus_csv.marker_notes = []
    for idx, r in raw.iterrows():
        first = r[0].strip() if isinstance(r[0], str) else ""
        if first in GROUP_MARKERS and pd.isna(r[1]):
            group = GROUP_MARKERS[first]
            stray = {c: r[c] for c in range(1, len(COLUMNS)) if pd.notna(r[c])}
            if stray:
                read_corpus_csv.marker_notes.append(
                    f"В строке-маркере «{first}» (строка {idx + 1} файла) есть лишние значения: "
                    + ", ".join(f"колонка {c} = «{v}»" for c, v in stray.items()) + ". Они проигнорированы.")
            continue
        if group is None:
            sys.exit(f"Строка {idx}: книга до первого маркера группы (РУ/ИНО).")
        rec = {COLUMNS[c]: r[c] for c in COLUMNS}
        rec["group"] = group
        rec["source_row"] = idx + 1  # номер строки в исходном файле (с 1)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["id"] = pd.to_numeric(df["id"], errors="raise").astype(int)
    df["book_id"] = df["group"] + "_" + df["id"].map(lambda n: f"{n:02d}")
    for c in ["year", "pages", "n_words"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["is_translation", "NER_annotation", "Sentiment__annotation"]:
        df[c] = df[c].map(lambda v: str(v).strip().lower() == "true" if pd.notna(v) else False)
    df["text_source"] = df["text"].map(lambda t: "csv" if isinstance(t, str) and t.strip() else pd.NA)
    return df


def title_key(s):
    s = e_norm(str(s)).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def read_extra_texts(path: Path) -> list:
    """Файл вида: title\\n<название>\\n\\ndescription\\n<...>\\n\\ntext\\n<текст> — повторяется для каждой книги."""
    raw = path.read_text(encoding="utf-8")
    raw = raw.replace("\r\n", "\n")
    starts = [m.start() for m in re.finditer(r"(?m)^title\n", raw)]
    recs = []
    for a, b in zip(starts, starts[1:] + [len(raw)]):
        parts = re.split(r"(?m)^(title|description|text)\n", raw[a:b])
        rec = dict(zip(parts[1::2], parts[2::2]))
        rec = {k: v.strip("\n") for k, v in rec.items()}
        recs.append(rec)
    return recs


def fill_missing_texts(df: pd.DataFrame, extra_path: Path, issues: list, info: list):
    """Добавляет тексты только тем книгам, у которых их нет в CSV. Тексты из CSV не заменяются."""
    recs = read_extra_texts(extra_path)
    by_title = {}
    for r in recs:
        by_title.setdefault(title_key(r.get("title", "")), []).append(r)
    norm_ws = lambda s: re.sub(r"\s+", " ", s or "").strip()
    added, differ = [], []
    for i, b in df.iterrows():
        cands = by_title.get(title_key(b.title), [])
        if b.text_source == "csv":
            if cands and all(norm_ws(c.get("text")) != norm_ws(b.text) for c in cands):
                differ.append(b.book_id)
            continue
        if len(cands) == 1 and cands[0].get("text", "").strip():
            df.at[i, "text"] = cands[0]["text"]
            df.at[i, "text_source"] = extra_path.name
            if pd.isna(b.description) and cands[0].get("description"):
                df.at[i, "description"] = cands[0]["description"]
            added.append(b.book_id)
        elif len(cands) > 1:
            issues.append(f"{b.book_id}: в {extra_path.name} несколько книг с названием «{b.title}» — текст не добавлен.")
    csv_titles = {title_key(t) for t in df.title}
    unknown = [r.get("title", "") for r in recs if title_key(r.get("title", "")) not in csv_titles]
    info.append(f"Из {extra_path.name} ({len(recs)} книг) добавлены тексты: {', '.join(added) or 'нет'}.")
    if differ:
        info.append(f"У {len(differ)} книг текст в {extra_path.name} отличается от CSV; используется версия из CSV: {', '.join(differ)}.")
    if unknown:
        info.append(f"Названия из {extra_path.name}, которых нет в CSV (не использованы): " + "; ".join(f"«{u}»" for u in unknown) + ".")


def count_words(text: str) -> int:
    """То же определение слова, что в предобработке: токен razdel внутри предложения, содержащий букву."""
    import razdel
    return sum(1 for s in razdel.sentenize(text) for t in razdel.tokenize(s.text) if WORD_RE.search(t.text))


def assign_kidlit_ids(meta: pd.DataFrame, issues: list, info: list) -> pd.DataFrame:
    meta = meta.copy()
    meta["kidlit_id"] = pd.NA
    meta["id_source"] = pd.NA
    for g, n, kid, prefix in FIXED_IDS:
        m = (meta.group == g) & (meta.id == n)
        if m.sum() != 1:
            sys.exit(f"Заданный ID {kid}: книга {g} id={n} не найдена.")
        title = meta.loc[m, "title"].iloc[0]
        if not e_norm(title).lower().startswith(e_norm(prefix).lower()):
            sys.exit(f"Заданный ID {kid}: ожидалось название «{prefix}…», в CSV «{title}».")
        meta.loc[m, ["kidlit_id", "id_source"]] = [kid, "задан в таблице проекта"]

    def stratum(w):
        if pd.isna(w):
            return pd.NA
        return "short" if w <= SHORT_MAX else ("medium" if w <= MEDIUM_MAX else "long")
    meta["length_by_rule"] = meta.n_words_counted.map(stratum)
    fixed = meta[meta.kidlit_id.notna()]
    mism = fixed[fixed.kidlit_id.str.split("_").str[1] != fixed.length_by_rule]
    for _, r in mism.iterrows():
        info.append(f"{r.kidlit_id} ({r.n_words_counted} слов) по правилу попал бы в {r.length_by_rule}; ID оставлен как в таблице проекта.")

    for (g, s), part in meta[meta.kidlit_id.isna() & meta.length_by_rule.notna()].groupby(["group", "length_by_rule"]):
        used = fixed[fixed.kidlit_id.str.startswith(f"{g}_{s}_")].kidlit_id.str[-2:].astype(int)
        nxt = (used.max() if len(used) else 0) + 1
        for i in part.sort_values("id").index:
            meta.at[i, "kidlit_id"] = f"{g}_{s}_{nxt:02d}"
            meta.at[i, "id_source"] = "присвоен по правилу"
            nxt += 1
    no_id = meta.loc[meta.kidlit_id.isna(), "book_id"].tolist()
    if no_id:
        issues.append(f"Без kidlit_id (нет текста): {', '.join(no_id)}.")
    border = meta[(meta.id_source == "присвоен по правилу") &
                  (meta.n_words_counted.between(SHORT_MAX + 1 - 15, SHORT_MAX + 15) |
                   meta.n_words_counted.between(MEDIUM_MAX + 1 - 30, MEDIUM_MAX + 30))]
    if len(border):
        info.append("Близко к границе длины (стоит проверить вручную): " +
                    ", ".join(f"{r.kidlit_id} ({r.n_words_counted} слов)" for _, r in border.iterrows()) + ".")
    meta["corpus_type"] = meta.kidlit_id.str.rsplit("_", n=1).str[0]
    if meta.kidlit_id.dropna().duplicated().any():
        sys.exit("Повторяющиеся kidlit_id — проверьте FIXED_IDS.")
    return meta


def normalize_text(book_id: str, text: str, morph, log: list) -> str:
    """Нормализация с полным логом. Порядок шагов важен."""
    def note(kind, before, after, count, pos=None):
        log.append({"book_id": book_id, "kind": kind, "before": before, "after": after,
                    "count": count, "pos_in_raw_or_step": pos})

    # 1. Unicode NFC (например, «й» из двух кодовых точек)
    nfc = unicodedata.normalize("NFC", text)
    if nfc != text:
        note("unicode_nfc", "", "", sum(1 for a, b in zip(text, nfc) if a != b) or 1)
    text = nfc
    # 2. Переводы строк
    n_crlf = text.count("\r\n")
    text = text.replace("\r\n", "\n")
    n_cr = text.count("\r")
    text = text.replace("\r", "\n")
    if n_crlf or n_cr:
        note("newline", "\\r\\n | \\r", "\\n", n_crlf + n_cr)
    # 3. Мягкие переносы и неразрывные пробелы
    for ch, rep, kind in [("\xad", "", "soft_hyphen"), ("\xa0", " ", "nbsp"), ("​", "", "zero_width_space")]:
        n = text.count(ch)
        if n:
            text = text.replace(ch, rep)
            note(kind, repr(ch), repr(rep), n)
    # 4. Ручные исправления
    for bid, old, new, why in MANUAL_FIXES:
        if bid != book_id:
            continue
        n = text.count(old)
        if n == 0:
            note("manual_fix_NOT_FOUND", old, new, 0, why)
        else:
            text = text.replace(old, new)
            note("manual_fix", old, new, n, why)
    # 5. Перенос слова на новую строку: «сестрён-\nка» → «сестрёнка»,
    #    только если продолжение со строчной буквы и склеенное слово есть в словаре.
    def join(m):
        a, b = m.group(1), m.group(2)
        joined = a + b
        if b[:1].islower() and morph.word_is_known(joined.lower()):
            note("hyphen_linebreak_join", m.group(0).replace("\n", "\\n"), joined, 1)
            return joined
        return m.group(0)
    text = re.sub(r"(\w+)-\n(\w+)", join, text)
    # 6. Пробелы в конце строк (прецедент проекта: пробел перед \n сдвигал смещения)
    new = re.sub(r"[ \t]+\n", "\n", text)
    if new != text:
        note("trailing_space", "' +\\n'", "\\n", len(re.findall(r"[ \t]+\n", text)))
    text = new
    # 7. Пробелы и пустые строки в начале и конце текста
    stripped = text.strip()
    if stripped != text:
        note("strip", "", "", 1)
    return stripped


def step_freeze(args, out: Path):
    data = out / "data"
    manifest_path = data / "freeze_manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"[0.2] Корпус уже заморожен ({manifest_path}). Чтобы пересобрать, добавьте --force.")
        return
    (data / "raw_texts").mkdir(parents=True, exist_ok=True)
    (data / "texts").mkdir(parents=True, exist_ok=True)
    morph = get_morph()

    for sub in ("raw_texts", "texts"):  # при перезаморозке убираем старые файлы
        for f in (data / sub).glob("*.txt"):
            f.unlink()
    df = read_corpus_csv(Path(args.corpus))
    log, issues, info = [], list(read_corpus_csv.marker_notes), []
    if args.extra_texts:
        fill_missing_texts(df, Path(args.extra_texts), issues, info)

    # Проверки метаданных
    if df["book_id"].duplicated().any():
        sys.exit(f"Повторяющиеся book_id: {df.loc[df.book_id.duplicated(), 'book_id'].tolist()}")
    for g, n in df.groupby("group").size().items():
        if n != 60:
            issues.append(f"В группе {g} {n} книг вместо 60.")
    for c in REQUIRED:
        miss = df.loc[df[c].isna(), "book_id"].tolist()
        if miss:
            issues.append(f"Пустое обязательное поле «{c}»: {', '.join(miss)}.")
    bad_tr = df[(df.group == "RUS-T") != df.is_translation]["book_id"].tolist()
    if bad_tr:
        issues.append(f"Признак перевода не совпадает с группой: {', '.join(bad_tr)}.")
    bad_lang = df[((df.group == "RUS-O") & (df.language != "русский")) |
                  ((df.group == "RUS-T") & ((df.language == "русский") | df.language.isna()))]["book_id"].tolist()
    if bad_lang:
        issues.append(f"Язык оригинала не соответствует группе: {', '.join(bad_lang)}.")
    out_years = df[(df.year < 2015) | (df.year > 2026)][["book_id", "year"]].values.tolist()
    if out_years:
        issues.append("Год вне декларированного диапазона 2015–2026: " + ", ".join(f"{b} ({y})" for b, y in out_years) + ".")
    for c in ["NER_annotation_author", "Sentiment_annotation_author"]:
        odd = df[df[c].notna() & ~df[c].isin(["A", "B"])][["book_id", c]].values.tolist()
        if odd:
            issues.append(f"Нестандартное значение в {c} (ожидается A/B): " + ", ".join(f"{b} = «{v}»" for b, v in odd) + ".")

    df["has_text"] = df["text"].notna() & (df["text"].fillna("").str.strip() != "")
    missing = df.loc[~df.has_text, "book_id"].tolist()
    if missing:
        issues.append(f"Нет текста у {len(missing)} книг: {', '.join(missing)}. В предобработку они не попадут.")

    files = {}
    clean_len, raw_len, n_words_counted = {}, {}, {}
    for _, r in df[df.has_text].iterrows():
        bid, raw = r.book_id, r.text
        (data / "raw_texts" / f"{bid}.txt").write_text(raw, encoding="utf-8", newline="")
        clean = normalize_text(bid, raw, morph, log)
        (data / "texts" / f"{bid}.txt").write_text(clean, encoding="utf-8", newline="\n")
        files[bid] = {"raw_sha256": sha256_text(raw), "text_sha256": sha256_text(clean),
                      "raw_chars": len(raw), "text_chars": len(clean), "text_source": r.text_source}
        raw_len[bid], clean_len[bid] = len(raw), len(clean)
        n_words_counted[bid] = count_words(clean)

    # Ручные исправления, которые не нашлись
    for e in log:
        if e["kind"] == "manual_fix_NOT_FOUND":
            issues.append(f"Ручное исправление не применилось: {e['book_id']} «{e['before']}».")

    # Остаточная латиница после исправлений — на ручной просмотр
    latin_left = []
    for bid in files:
        t = (data / "texts" / f"{bid}.txt").read_text(encoding="utf-8")
        for m in re.finditer(r"\w*[A-Za-z]\w*", t):
            if (bid, m.group()) not in ALLOWED_LATIN:
                latin_left.append((bid, m.group(), t[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")))
    pd.DataFrame(latin_left, columns=["book_id", "token", "context"]).to_csv(data / "qc_latin_left.csv", index=False)
    if latin_left:
        issues.append(f"После исправлений осталось {len(latin_left)} латинских токенов — см. data/qc_latin_left.csv.")

    meta = df.drop(columns=["text"]).copy()
    meta["text_chars"] = meta.book_id.map(clean_len).astype("Int64")
    meta["raw_chars"] = meta.book_id.map(raw_len).astype("Int64")
    meta["n_words_counted"] = meta.book_id.map(n_words_counted).astype("Int64")
    ratio = (meta.n_words_counted / meta.n_words).round(3)
    far = meta[(ratio < 0.9) | (ratio > 1.1)]
    for _, r in far.iterrows():
        issues.append(f"{r.book_id}: в тексте {r.n_words_counted} слов, в метаданных n_words = {r.n_words} "
                      f"(отношение {r.n_words_counted / r.n_words:.2f}). Проверить, тот ли это текст и полный ли он.")
    meta = assign_kidlit_ids(meta, issues, info)
    first = ["book_id", "kidlit_id", "corpus_type", "id_source", "group", "id", "has_text", "text_source", "title",
             "original_title", "author", "author_gender", "year", "language", "country_of_origin", "publisher",
             "age_marker", "pages", "illustrator", "УДК", "genre", "NER_annotation", "Sentiment__annotation",
             "NER_annotation_author", "Sentiment_annotation_author", "n_words", "n_words_counted", "length_by_rule",
             "awards", "is_translation", "translator", "text_chars", "raw_chars"]
    meta = meta[first + [c for c in meta.columns if c not in first]]
    meta.to_csv(data / "books.csv", index=False)
    pd.DataFrame(log).to_csv(data / "normalization_log.csv", index=False)

    manifest = {
        "corpus_version": CORPUS_VERSION,
        "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_csv": Path(args.corpus).name,
        "source_csv_sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
        "extra_texts": Path(args.extra_texts).name if args.extra_texts else None,
        "extra_texts_sha256": hashlib.sha256(Path(args.extra_texts).read_bytes()).hexdigest() if args.extra_texts else None,
        "books_total": int(len(df)),
        "books_with_text": int(df.has_text.sum()),
        "books_by_group": df.groupby("group").size().to_dict(),
        "books_with_text_by_group": df[df.has_text].groupby("group").size().to_dict(),
        "books_by_corpus_type": meta.corpus_type.value_counts().sort_index().to_dict(),
        "books_csv_sha256": hashlib.sha256((data / "books.csv").read_bytes()).hexdigest(),
        "files": files,
        "issues": issues,
        "info": info,
        "rule": "После заморозки тексты в data/texts не редактируются. Любая правка — новая версия корпуса и новый манифест.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[0.2] Заморожено: {manifest['books_with_text']} книг с текстом из {manifest['books_total']}. "
          f"Замечаний: {len(issues)}.")


def verify_freeze(out: Path):
    """Проверяет, что тексты не менялись после заморозки."""
    data = out / "data"
    man = json.loads((data / "freeze_manifest.json").read_text(encoding="utf-8"))
    changed = [b for b, f in man["files"].items()
               if sha256_text((data / "texts" / f"{b}.txt").read_text(encoding="utf-8")) != f["text_sha256"]]
    if changed:
        sys.exit(f"Тексты изменены после заморозки: {changed}. Перезаморозьте корпус (--steps freeze --force).")
    return man


# ---------------------------------------------------------------------------
# Шаг 0.3. Предобработка
# ---------------------------------------------------------------------------
# Какие части речи pymorphy3 совместимы с UD-тегом spaCy (для выбора разбора в контексте)
UD2OC = {
    "NOUN": {"NOUN"}, "PROPN": {"NOUN"},
    "ADJ": {"ADJF", "ADJS", "COMP", "PRTF", "PRTS"},
    "DET": {"ADJF", "NPRO"},
    "VERB": {"VERB", "INFN", "PRTF", "PRTS", "GRND"}, "AUX": {"VERB", "INFN"},
    "ADV": {"ADVB", "COMP", "PRED"},
    "PRON": {"NPRO"}, "NUM": {"NUMR", "ADJF"},
    "ADP": {"PREP"}, "CCONJ": {"CONJ"}, "SCONJ": {"CONJ"},
    "PART": {"PRCL"}, "INTJ": {"INTJ"},
}


def choose_parse(parses, ud_pos):
    allowed = UD2OC.get(ud_pos)
    if allowed:
        for rank, p in enumerate(parses):
            if p.tag.POS in allowed:
                return rank, p
    return 0, parses[0]


def step_preprocess(args, out: Path):
    import razdel
    import spacy
    from spacy.tokens import Doc

    man = verify_freeze(out)
    data = out / "data"
    books = pd.read_csv(data / "books.csv")
    morph = get_morph()
    nlp = None
    if not args.no_syntax:
        nlp = spacy.load(args.spacy_model, exclude=["ner"])

    tok_rows, sent_rows, stats = [], [], []
    for _, b in books[books.has_text].iterrows():
        bid, grp, kid = b.book_id, b.group, b.kidlit_id
        text = (data / "texts" / f"{bid}.txt").read_text(encoding="utf-8")
        book_start = len(tok_rows)
        newlines = [i for i, ch in enumerate(text) if ch == "\n"]
        sents = list(razdel.sentenize(text))
        sent_tokens = []
        for s in sents:
            toks = [(s.start + t.start, s.start + t.stop, t.text) for t in razdel.tokenize(s.text)]
            if toks:
                sent_tokens.append((s, toks))

        ud = []
        if nlp is not None:
            docs = []
            for s, toks in sent_tokens:
                words = [t[2] for t in toks]
                spaces = [toks[i][1] < toks[i + 1][0] for i in range(len(toks) - 1)] + [False]
                docs.append(Doc(nlp.vocab, words=words, spaces=spaces))
            ud = list(nlp.pipe(docs, batch_size=64))

        tok_idx = 0
        for sid, (s, toks) in enumerate(sent_tokens):
            sent_rows.append({"book_id": bid, "kidlit_id": kid, "group": grp, "sent_id": sid, "start": s.start, "end": s.stop,
                              "n_tokens": len(toks), "text": s.text})
            sdoc = ud[sid] if ud else None
            for i, (st, en, form) in enumerate(toks):
                is_word = bool(WORD_RE.search(form))
                row = {"book_id": bid, "kidlit_id": kid, "group": grp, "sent_id": sid, "tok_id": i, "tok_idx_book": tok_idx,
                       "start": st, "end": en, "line_no": bisect.bisect_right(newlines, st),
                       "token": form, "token_lower": form.lower(), "token_e": e_norm(form.lower()),
                       "is_word": is_word, "is_cap": form[:1].isupper(),
                       "mixed_script": bool(LATIN_RE.search(form) and CYR_RE.search(form))}
                if sdoc is not None:
                    t = sdoc[i]
                    row.update({"pos_ud": t.pos_, "morph_ud": str(t.morph), "head": t.head.i,
                                "deprel": t.dep_, "lemma_sp": t.lemma_})
                if is_word:
                    parses = morph.parse(form)
                    rank, p = choose_parse(parses, row.get("pos_ud"))
                    row.update({"lemma": p.normal_form, "lemma_e": e_norm(p.normal_form), "pos": p.tag.POS,
                                "gram": str(p.tag), "pm_score": round(float(p.score), 4), "pm_rank": rank,
                                "is_oov": not morph.word_is_known(form.lower())})
                tok_rows.append(row)
                tok_idx += 1

        words = [r for r in tok_rows[book_start:] if r["is_word"]]
        stats.append({
            "book_id": bid, "kidlit_id": kid, "corpus_type": b.corpus_type, "group": grp,
            "n_chars": len(text), "n_sents": len(sent_tokens), "n_tokens": tok_idx, "n_words": len(words),
            "n_lemmas": len({w["lemma"] for w in words}),
            "n_nouns_pm": sum(1 for w in words if w.get("pos") == "NOUN"),
            "n_nouns_ud": sum(1 for w in words if w.get("pos_ud") == "NOUN"),
            "n_propn_ud": sum(1 for w in words if w.get("pos_ud") == "PROPN"),
            "n_adj_pm": sum(1 for w in words if w.get("pos") in ("ADJF", "ADJS")),
            "n_adv_pm": sum(1 for w in words if w.get("pos") == "ADVB"),
            "n_oov": sum(1 for w in words if w.get("is_oov")),
            "mean_sent_len_words": round(len(words) / max(len(sent_tokens), 1), 2),
            "n_words_meta": b.n_words,
        })
        print(f"[0.3] {bid}: {len(sent_tokens)} предл., {len(words)} слов")

    tok = pd.DataFrame(tok_rows)
    for c in ["head"]:
        if c in tok:
            tok[c] = tok[c].astype("Int32")
    tok.to_parquet(data / "tokens.parquet", index=False)
    pd.DataFrame(sent_rows).to_parquet(data / "sentences.parquet", index=False)
    st = pd.DataFrame(stats)
    st["words_vs_meta"] = (st.n_words / st.n_words_meta).round(3)
    mism = st[st.n_words != st.book_id.map(books.set_index("book_id").n_words_counted)]
    if len(mism):
        sys.exit(f"[0.3] Число слов расходится с подсчётом при заморозке: {mism.book_id.tolist()}")
    st.to_csv(data / "book_stats.csv", index=False)

    # Самопроверка смещений: каждый токен совпадает с текстом по своим start/end
    bad = 0
    for bid, g in tok.groupby("book_id"):
        text = (data / "texts" / f"{bid}.txt").read_text(encoding="utf-8")
        bad += sum(text[s:e] != f for s, e, f in zip(g.start, g.end, g.token))
    if bad:
        sys.exit(f"[0.3] Ошибка: у {bad} токенов смещения не совпадают с текстом.")
    print(f"[0.3] Готово: {len(tok)} токенов, смещения проверены.")


# ---------------------------------------------------------------------------
# Шаг 0.4. Референсный ресурс: частотный словарь Ляшевской–Шарова (НКРЯ)
# ---------------------------------------------------------------------------
RNC_CORPUS_SIZE = 92_000_000  # объём корпуса словаря, словоупотреблений (freqrnc_readme.txt)
RNC_POS_LABEL = {"s": "сущ.", "s.PROP": "имя собств.", "v": "глагол", "a": "прил.", "adv": "нареч.",
                 "advpro": "местоим. нареч.", "apro": "местоим. прил.", "spro": "местоим. сущ.",
                 "num": "числ.", "anum": "порядк. числ.", "pr": "предлог", "conj": "союз",
                 "part": "частица", "intj": "междом."}
PROPER_GRAMMEMES = {"Name", "Surn", "Patr", "Geox", "Orgn", "Trad"}


def pm_to_rnc(pos, gram, pos_ud=None):
    """Часть речи pymorphy3 → тег словаря Ляшевской–Шарова."""
    g = set(re.split(r"[ ,]", gram or ""))
    if pos == "NOUN":
        return "s.PROP" if (g & PROPER_GRAMMEMES or pos_ud == "PROPN") else "s"
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
    return {"NPRO": "spro", "NUMR": "num", "PREP": "pr", "CONJ": "conj", "PRCL": "part", "INTJ": "intj"}.get(pos)


def load_freq(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv") and "freqrnc" in n.lower())
        raw = z.read(name).decode("utf-8")
    f = pd.read_csv(io.StringIO(raw), sep="\t", dtype={"Lemma": str, "PoS": str}, keep_default_na=False)
    f.columns = ["lemma", "pos_rnc", "ipm", "R", "D", "doc"]
    return f


def step_reference(args, out: Path):
    ref = out / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    f = load_freq(Path(args.freq_zip))
    f["lemma_e"] = f.lemma.str.lower().map(e_norm)
    f["approx_count"] = (f.ipm * RNC_CORPUS_SIZE / 1e6).round().astype(int)
    f.to_csv(ref / "freqrnc2011_clean.csv", index=False)
    by_lp = f.groupby(["lemma_e", "pos_rnc"], as_index=False).agg(ipm=("ipm", "sum"), doc=("doc", "max"))
    by_l = f.groupby("lemma_e", as_index=False).agg(ipm_any_pos=("ipm", "sum"), pos_list=("pos_rnc", lambda s: "|".join(sorted(set(s)))))

    tok = pd.read_parquet(out / "data" / "tokens.parquet")
    w = tok[tok.is_word].copy()
    w["pos_rnc"] = [pm_to_rnc(p, g, u) for p, g, u in zip(w.pos, w.gram, w.get("pos_ud", [None] * len(w)))]
    kl = (w.groupby(["lemma_e", "pos_rnc"], dropna=False)
            .agg(freq=("token", "size"), n_books=("book_id", "nunique"),
                 freq_rus_o=("group", lambda s: (s == "RUS-O").sum()),
                 freq_rus_t=("group", lambda s: (s == "RUS-T").sum()),
                 lemma_example=("lemma", "first"), is_oov_share=("is_oov", "mean"))
            .reset_index().sort_values("freq", ascending=False))
    kl = kl.merge(by_lp.rename(columns={"ipm": "ref_ipm", "doc": "ref_doc"}), on=["lemma_e", "pos_rnc"], how="left")
    kl = kl.merge(by_l, on="lemma_e", how="left")
    kl["match"] = "lemma+pos"
    kl.loc[kl.ref_ipm.isna() & kl.ipm_any_pos.notna(), "match"] = "lemma_only"
    kl.loc[kl.ref_ipm.isna() & kl.ipm_any_pos.isna(), "match"] = "none"
    kl["ref_pos_list"] = kl.pos_list
    kl = kl.drop(columns=["pos_list"])
    kl.to_csv(ref / "kidlit_lemma_freq.csv", index=False)

    top50 = kl.head(50)[["lemma_example", "lemma_e", "pos_rnc", "freq", "n_books", "match", "ref_ipm", "ipm_any_pos", "ref_pos_list"]]
    top50.to_csv(ref / "check_top50.csv", index=False)
    unmatched = kl[kl.match == "none"].head(200)[["lemma_example", "pos_rnc", "freq", "n_books", "freq_rus_o", "freq_rus_t", "is_oov_share"]]
    unmatched.to_csv(ref / "unmatched_top200.csv", index=False)

    tok_cov = {m: int(kl.loc[kl.match == m, "freq"].sum()) for m in ["lemma+pos", "lemma_only", "none"]}
    meta = {"rnc_rows": int(len(f)), "rnc_lemma_pos": int(len(by_lp)), "rnc_has_yo": bool(f.lemma.str.contains("ё").any()),
            "rnc_ipm_sum": round(float(f.ipm.sum()), 1), "kidlit_lemma_pos": int(len(kl)),
            "kidlit_word_tokens": int(len(w)), "token_coverage": tok_cov,
            "type_coverage": kl.match.value_counts().to_dict(),
            "top50_mismatch": top50[top50.match != "lemma+pos"][["lemma_example", "pos_rnc", "match", "ref_pos_list"]].values.tolist(),
            "kidlit_lemmas_with_yo": int(w.lemma.str.contains("ё").sum())}
    (ref / "reference_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[0.4] Словарь: {len(f)} строк. Покрытие словоупотреблений KidLit: {tok_cov}")


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------
def step_report(args, out: Path):
    data, ref = out / "data", out / "reference"
    man = json.loads((data / "freeze_manifest.json").read_text(encoding="utf-8"))
    books = pd.read_csv(data / "books.csv")
    log = pd.read_csv(data / "normalization_log.csv")
    st = pd.read_csv(data / "book_stats.csv") if (data / "book_stats.csv").exists() else None
    rm = json.loads((ref / "reference_meta.json").read_text(encoding="utf-8")) if (ref / "reference_meta.json").exists() else None
    L = []
    a = L.append
    a(f"# Шаги 0.2–0.4: отчёт\n\nСформирован {dt.datetime.now():%d.%m.%Y %H:%M}. Все числа взяты из файлов в папке результата.\n")
    a("## 0.2. Заморозка корпуса\n")
    a(f"- Версия: {man['corpus_version']}, заморожено {man['frozen_at']}.")
    a(f"- Книг в метаданных: {man['books_total']} ({', '.join(f'{k}: {v}' for k, v in man['books_by_group'].items())}).")
    a(f"- Книг с текстом: {man['books_with_text']} ({', '.join(f'{k}: {v}' for k, v in man['books_with_text_by_group'].items())}).")
    a(f"- SHA-256 исходного CSV: `{man['source_csv_sha256'][:16]}…`" +
      (f"; дополнительный файл текстов: {man['extra_texts']} (`{man['extra_texts_sha256'][:16]}…`)." if man.get("extra_texts") else ""))
    src = books[books.has_text].text_source.value_counts()
    a("- Источник текстов: " + ", ".join(f"{k} — {v}" for k, v in src.items()) + ".\n")
    a("### Замечания к данным\n")
    a("\n".join(f"- {x}" for x in man["issues"]) if man["issues"] else "- Нет.")
    a("\n### Информация\n")
    a("\n".join(f"- {x}" for x in man.get("info", [])) if man.get("info") else "- Нет.")
    a("\n### ID KidLit\n")
    a("Формат `<группа>_<длина>_<NN>`. 30 ID взяты из таблицы проекта; остальные присвоены по правилу: "
      f"short — до {SHORT_MAX} слов, medium — {SHORT_MAX + 1}–{MEDIUM_MAX}, long — от {MEDIUM_MAX + 1} "
      "(слова считаются по нормализованному тексту). Нумерация продолжается с 06 в порядке id из CSV.\n")
    ct = books.dropna(subset=["corpus_type"]).groupby(["corpus_type", "id_source"]).size().unstack(fill_value=0)
    a("| Тип | " + " | ".join(ct.columns) + " | Всего |\n|---|" + "---|" * (len(ct.columns) + 1))
    for k, r in ct.iterrows():
        a(f"| {k} | " + " | ".join(str(int(v)) for v in r.values) + f" | {int(r.sum())} |")
    a("\n### Нормализация текстов\n")
    summ = log.groupby("kind").agg(books=("book_id", "nunique"), changes=("count", "sum")).reset_index()
    a("| Тип изменения | Книг | Изменений |\n|---|---|---|")
    for _, r in summ.iterrows():
        a(f"| {r.kind} | {r.books} | {r['changes']} |")
    a("\nПолный список — `data/normalization_log.csv`. Исходные тексты сохранены без изменений в `data/raw_texts/`.\n")
    a("### Языки оригинала (RUS-T)\n")
    a("| Язык | Книг |\n|---|---|")
    for k, v in books[books.group == "RUS-T"].language.value_counts().items():
        a(f"| {k} | {v} |")
    if st is not None:
        a("\n## 0.3. Предобработка\n")
        g = st.groupby("group").agg(books=("book_id", "size"), sents=("n_sents", "sum"), words=("n_words", "sum"),
                                    words_median=("n_words", "median"), nouns=("n_nouns_pm", "sum"), oov=("n_oov", "sum"))
        a("| Группа | Книг | Предложений | Слов | Медиана слов на книгу | Сущ. (pymorphy3) | Вне словаря |\n|---|---|---|---|---|---|---|")
        for k, r in g.iterrows():
            a(f"| {k} | {int(r.books)} | {int(r.sents)} | {int(r.words)} | {r.words_median:.0f} | {int(r.nouns)} | {int(r.oov)} ({r.oov / r.words:.1%}) |")
        tot = st.n_words.sum()
        a(f"| Всего | {len(st)} | {st.n_sents.sum()} | {tot} | {st.n_words.median():.0f} | {st.n_nouns_pm.sum()} | {st.n_oov.sum()} ({st.n_oov.sum() / tot:.1%}) |")
        rat = st.words_vs_meta.dropna()
        a(f"\nСверка с n_words из метаданных: есть у {len(rat)} книг; "
          f"медиана отношения «наш подсчёт / n_words» = {rat.median():.3f}, диапазон {rat.min():.3f}–{rat.max():.3f}.")
        far = st[(st.words_vs_meta < 0.9) | (st.words_vs_meta > 1.1)][["book_id", "kidlit_id", "n_words", "n_words_meta", "words_vs_meta"]]
        if len(far):
            a(f"Расхождение больше 10% у {len(far)} книг:\n")
            a("| Книга | ID KidLit | Наш подсчёт | n_words | Отношение |\n|---|---|---|---|---|")
            for _, r in far.iterrows():
                a(f"| {r.book_id} | {r.kidlit_id} | {r.n_words} | {r.n_words_meta} | {r.words_vs_meta} |")
        tok = pd.read_parquet(data / "tokens.parquet", columns=["is_word", "is_oov", "lemma", "token_lower", "pm_rank", "mixed_script"])
        oov = tok[tok.is_word & tok.is_oov].token_lower.value_counts().head(40)
        a("\nСамые частотные слова вне словаря pymorphy3 (среди них ищем окказиональные диминутивы):\n")
        a(", ".join(f"{k} ({v})" for k, v in oov.items()))
        a(f"\nРазбор pymorphy3 выбран не первым по вероятности, а по согласованию с частью речи spaCy: "
          f"{(tok[tok.is_word].pm_rank > 0).mean():.1%} словоупотреблений. Смешанное написание (латиница + кириллица) после нормализации: {int(tok.mixed_script.sum())}.")
    if rm is not None:
        a("\n## 0.4. Референсный ресурс\n")
        a(f"- Частотный словарь Ляшевской–Шарова: {rm['rnc_rows']} строк (лемма + часть речи). Буква ё в словаре "
          f"{'встречается' if rm['rnc_has_yo'] else 'не используется'} — сопоставление ведётся по леммам с заменой ё → е.")
        a(f"- Абсолютные частоты восстановлены как ipm × 92 (объём корпуса словаря — 92 млн словоупотреблений).")
        tc = rm["token_coverage"]; tw = sum(tc.values())
        a(f"- Покрытие словоупотреблений KidLit: лемма и часть речи совпали — {tc['lemma+pos'] / tw:.1%}; "
          f"только лемма — {tc['lemma_only'] / tw:.1%}; нет в словаре — {tc['none'] / tw:.1%}.")
        a(f"- Среди 50 самых частотных лемм KidLit без полного совпадения: {len(rm['top50_mismatch'])}.")
        if rm["top50_mismatch"]:
            a("\n| Лемма | Часть речи (наша) | Совпадение | Части речи в словаре |\n|---|---|---|---|")
            for lem, pos, m, pl in rm["top50_mismatch"]:
                a(f"| {lem} | {pos} | {m} | {pl} |")
        a("\nДетали: `reference/check_top50.csv`, `reference/unmatched_top200.csv`, `reference/kidlit_lemma_freq.csv`.")
    a("\n## Связь ID\n")
    a("`book_id` (например, `RUS-O_07`) — ключ файлов, построен из id в CSV. `kidlit_id` (например, `RUS-O_long_01`) — "
      "ID проекта, совпадает с логом адъюдикации. Оба есть в `books.csv`, `book_stats.csv`, `tokens.parquet` и `sentences.parquet`.")
    (out / "results").mkdir(exist_ok=True)
    (out / "results" / "step0_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[отчёт] {out / 'results' / 'step0_report.md'}")


def main():
    ap = argparse.ArgumentParser(description="KidLit: шаги 0.2–0.4 (заморозка, предобработка, референсный словарь)")
    ap.add_argument("--corpus", required=True, help="CSV с текстами и метаданными")
    ap.add_argument("--freq-zip", required=True, help="архив Freq2011 с freqrnc2011.csv")
    ap.add_argument("--extra-texts", help="txt с текстами (title/description/text) — для книг без текста в CSV")
    ap.add_argument("--out", default="kidlit_dim", help="папка результата")
    ap.add_argument("--steps", default="freeze,preprocess,reference,report")
    ap.add_argument("--force", action="store_true", help="перезаморозить корпус, если он уже заморожен")
    ap.add_argument("--no-syntax", action="store_true", help="без spaCy (нет UD-тегов и синтаксиса)")
    ap.add_argument("--spacy-model", default="ru_core_news_sm")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = [s.strip() for s in args.steps.split(",")]
    if "freeze" in steps:
        step_freeze(args, out)
    if "preprocess" in steps:
        step_preprocess(args, out)
    if "reference" in steps:
        step_reference(args, out)
    if "report" in steps:
        step_report(args, out)


if __name__ == "__main__":
    main()
