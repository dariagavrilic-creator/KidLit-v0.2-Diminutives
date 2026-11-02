#!/usr/bin/env python3
"""
KidLit by Numbers — шаги 1.2 и 2.4: детектор диминутивов v3 (замороженная версия, по которой посчитана оценка на test).
С полным словарём прецедентов (dim_lexicon_v3.1.csv) это рабочая версия v3.1, которой размечен корпус.

Реализует раздел 6 «Инструкции по разметке диминутивов KidLit» (v0.4–v0.5):
  A. кандидаты по суффиксу → B. восстановление производящего (pymorphy3 + DerivBase.Ru)
  → C. списки прецедентов → D. метка по первому сработавшему правилу (R1–R5).

Запуск:
    python kidlit_dim_detect.py --data kidlit_dim --derivbase v001.00-connections.zip

    --data        папка результата шагов 0.2–0.4 (нужны data/tokens.parquet и reference/freqrnc2011_clean.csv)
    --derivbase   архив связей DerivBase.Ru (data/wiktionary/v001.00-connections.zip из репозитория
                  github.com/s231644/DerivBaseRu); без него работает только восстановление по pymorphy3
    --lexicon     файл списков прецедентов (по умолчанию <data>/detector/dim_lexicon.csv).
                  При первом запуске создаётся из стартового набора; дальше редактируется вручную
                  и при следующих запусках берётся как есть.
    --no-lexicon  прогон без списков прецедентов (абляция для статьи)

Результат (<data>/detector/):
    candidates.parquet / candidates.csv  одна строка на словоупотребление-кандидат
    lemma_summary.csv                    одна строка на лемму: метка, производящее, частота, примеры —
                                         для проверки по типам (план Б дорожной карты)
    detector_report.md                   сводка
"""

import argparse
import io
import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pymorphy3

DETECTOR_VERSION = "v3"

# ---------------------------------------------------------------------------
# Этап A. Суффиксы (конец леммы). Порядок важен: длинные раньше коротких.
# min_stem — минимальная длина остатка основы (защита от «река → рея»).
# ---------------------------------------------------------------------------
M = ["", "ь", "й", "а", "я", "о"]        # производящее для диминутивов м. р. (дом, огонь, папа)
F = ["а", "я", "ь", ""]                  # ж. р. (лапа, капля, мышь)
N = ["о", "е", "ё"]                      # ср. р. (окно, солнце)
# (суффикс, минимальная длина основы, окончания производящего)
NOUN_SUFFIXES = [
    # диминутив от названия детёныша (котёночек → котёнок) — раньше, чем YOUNG
    ("ёночек", 2, ["ёнок"]), ("оночек", 2, ["онок"]),
    # уничижительные
    ("ишка", 2, M + F), ("ишко", 2, M + N), ("ёнка", 3, F + M), ("онка", 3, F + M),
    # двухступенчатые: сначала ближайшее производящее (кусочек → кусок, ложечка → ложка)
    ("очек", 2, ["ок"] + M), ("ечек", 2, ["ек", "ок"] + M), ("ичек", 2, ["ик"] + M),
    ("ышек", 2, M), ("ушек", 2, M),
    ("очка", 2, ["ка"] + F), ("ечка", 2, ["ка"] + F + N), ("ичка", 2, F + N),
    ("ушка", 2, F + M), ("юшка", 2, F + M), ("ышка", 2, F),
    ("онька", 2, F + M), ("енька", 2, F + M),
    ("очко", 2, ["ко"] + N), ("ечко", 2, ["ко"] + N + ["ье"]), ("ышко", 2, N), ("ушко", 2, N),
    ("унчик", 2, M), ("уля", 2, F + M), ("уся", 2, F + M),
    ("ико", 2, N), ("ице", 2, N + ["ье"]), ("ица", 3, F),
    ("чик", 2, M), ("ик", 2, M), ("ёк", 2, M), ("ек", 3, M), ("ок", 2, M), ("ец", 3, M),
    ("ка", 3, F + M), ("ко", 2, N), ("цо", 3, N), ("це", 3, N),
]
YOUNG_SUFFIXES = ("ёнок", "онок")
TWO_STEP = ("очек", "ечек", "ичек")
ADJ_SUFFIXES = [(suf, 2, ["ый", "ий", "ой", "кий"]) for suf in ("ёхонький", "охонький", "ешенький", "ошенький", "енький", "онький")]
ADV_SUFFIXES = [("онько", 2, ["о", "ко"]), ("енько", 2, ["о", "ко"]), ("оньку", 2, ["у", "ку"]), ("еньку", 2, ["у", "ку"]),
                ("ечко", 3, ["ко", "о"]), ("очко", 3, ["ко", "о"]), ("очку", 2, ["ку", "у"]), ("ечку", 2, ["ку", "у"]),
                ("шко", 3, ["го", "ко"])]
PEJ_SUFFIXES = ("ишка", "ишко", "ёнка", "онка")
HYP_SUFFIXES = ("енька", "онька", "очка", "ечка", "ушка", "юшка", "ишка", "уля", "уся", "ка", "ик", "ок")
YOUNG_BASE_ENDINGS = ["", "а", "я", "ь", "ка"]
# Обратные чередования последней согласной основы (ручка → рука, ножка → нога, ушко → ухо)
ALTERNATIONS = {"ч": ["к", "ц", "т"], "ж": ["г", "з", "д"], "ш": ["х", "с"], "щ": ["ст", "ск"]}
# Производящее отвергается, если в словаре Ляшевской–Шарова оно больше чем в BASE_RATIO раз
# реже производного (солнце ↛ «солонка», сказка ↛ «сказ», коробка ↛ «коробок»)
BASE_RATIO = 2.0

# ---------------------------------------------------------------------------
# Этап C. Стартовые списки прецедентов (журнал прецедентов инструкции v0.2).
# Записываются в dim_lexicon.csv при первом запуске; дальше правится файл.
# label: DIM / LEX / HYP / YOUNG / NO / CONTEXT (CONTEXT → DIM + needs_review)
# ---------------------------------------------------------------------------
SEED = [
    # lemma, label, base, pej, name, precedent, comment
    # P-17: у LEX base — само слово; у YOUNG — взрослое животное; у DIM — ближайшее производящее
    ("бабушка", "LEX", "бабушка", 0, 0, "P-01", ""),
    ("дедушка", "LEX", "дедушка", 0, 0, "P-01", ""),
    ("девочка", "LEX", "девочка", 0, 0, "P-01", ""),
    ("мальчик", "LEX", "мальчик", 0, 0, "P-01", ""),
    ("книжка", "DIM", "книга", 0, 0, "P-02", ""),
    ("морковка", "DIM", "морковь", 0, 0, "P-02", ""),
    ("речка", "DIM", "река", 0, 0, "P-02", ""),
    ("машинка", "DIM", "машина", 0, 0, "P-02", ""),
    ("ветка", "LEX", "ветка", 0, 0, "P-03", ""),
    ("маленький", "LEX", "маленький", 0, 0, "P-03", ""),
    ("ручка", "CONTEXT", "рука", 0, 0, "P-04", ""),
    ("ножка", "CONTEXT", "нога", 0, 0, "P-04", ""),
    ("глазок", "CONTEXT", "глаз", 0, 0, "P-04", ""),
    ("конёк", "CONTEXT", "конь", 0, 0, "P-04", ""),
    ("горка", "CONTEXT", "гора", 0, 0, "P-04", ""),
    ("лисичка", "CONTEXT", "лиса", 0, 0, "P-04", ""),
    ("звёздочка", "CONTEXT", "звезда", 0, 0, "P-04", ""),
    ("грудка", "CONTEXT", "грудь", 0, 0, "P-04", "часть тела птицы — LEX"),
    ("шарик", "CONTEXT", "шар", 0, 0, "P-05", "воздушный шарик — LEX"),
    ("кроватка", "CONTEXT", "кровать", 0, 0, "P-05", "детская кроватка — LEX; кукольная — DIM"),
    ("щенок", "NO", "", 0, 0, "P-07", ""),
    ("ребёнок", "NO", "", 0, 0, "P-07", ""),
    ("зайчишка", "DIM", "заяц", 1, 0, "P-08", ""),
    ("муравьишка", "DIM", "муравей", 1, 0, "P-08", ""),
    ("мальчишка", "DIM", "мальчик", 1, 0, "P-08", ""),
    ("девчонка", "DIM", "девочка", 1, 0, "P-08", ""),
    ("ежунчик", "DIM", "ёж", 0, 1, "P-09", ""),
    ("мишутка", "DIM", "Миша", 0, 1, "P-09", ""),
    ("шпунчик", "NO", "", 0, 1, "P-09", ""),
    ("квачик", "NO", "", 0, 1, "P-09", ""),
    ("снежинка", "NO", "", 0, 0, "P-11", ""),
    ("тропинка", "DIM", "тропа", 0, 0, "P-11", ""),
    ("снегурочка", "LEX", "снегурочка", 0, 1, "P-13", ""),
    ("игрушка", "LEX", "игрушка", 0, 0, "P-15", ""),
    ("мишка", "LEX", "мишка", 0, 0, "P-16", ""),
    ("бедняжка", "DIM", "бедняга", 0, 0, "P-18", ""),
    ("кошка", "LEX", "кошка", 0, 0, "разд. 3", ""),
    ("коробка", "LEX", "коробка", 0, 0, "пилот", "адъюдикация пилота"),
    ("крыльцо", "NO", "", 0, 0, "пилот", "отсечение даёт «крыло»"),
    ("кабачок", "NO", "", 0, 0, "разд. 6", "отсечение даёт «кабак»"),
    ("котофеечка", "DIM", "котофей", 0, 0, "контр. пример 13", "nonce"),
    # dev v1 по частотному списку — проверить
    ("цветок", "LEX", "цветок", 0, 0, "", "dev v1: другой денотат"),
    ("площадка", "LEX", "площадка", 0, 0, "", "dev v1: другой денотат"),
    # производящее не восстанавливается отсечением
    ("зайчик", "DIM", "заяц", 0, 0, "", "база задана вручную"),
    ("зайка", "DIM", "заяц", 0, 0, "", "база задана вручную"),
    ("немножко", "DIM", "немного", 0, 0, "", "база задана вручную"),
    ("чуточку", "DIM", "чуть", 0, 0, "", "база задана вручную"),
    # YOUNG: P-17 — base = взрослое животное (где отсечение даёт не то)
    ("зайчонок", "YOUNG", "заяц", 0, 0, "P-17", ""),
    ("поросёнок", "YOUNG", "свинья", 0, 0, "P-17", ""),
    ("ягнёнок", "YOUNG", "овца", 0, 0, "P-17", ""),
    ("жеребёнок", "YOUNG", "лошадь", 0, 0, "P-17", ""),
    ("цыплёнок", "YOUNG", "курица", 0, 0, "P-17", ""),
    ("телёнок", "YOUNG", "корова", 0, 0, "P-17", ""),
    ("крольчонок", "YOUNG", "кролик", 0, 0, "P-17", ""),
    ("бурундучонок", "YOUNG", "бурундук", 0, 0, "P-17", ""),
]
# 20 контрольных примеров из раздела 8 инструкции: (№, kidlit_id, словоформа, метка по ключу)
CONTROL = [(1, "RUS-O_long_09", "ручки", "DIM"), (2, "RUS-O_long_07", "ручку", "LEX"), (3, "RUS-O_long_06", "мышатам", "YOUNG"),
           (4, "RUS-O_long_25", "Анька", "HYP"), (5, "RUS-O_long_08", "грузовик", "NO"), (6, "RUS-O_medium_01", "тихонько", "DIM"),
           (7, "RUS-O_long_07", "новеньким", "DIM"), (8, "RUS-O_long_07", "снежинкам", "NO"), (9, "RUS-O_long_07", "тропинке", "DIM"),
           (10, "RUS-O_long_07", "горки", "LEX"), (11, "RUS-O_long_07", "Мамочка", "DIM"), (12, "RUS-O_medium_01", "Автомобильчик", "DIM"),
           (13, "RUS-O_long_26", "котофеечки", "DIM"), (14, "RUS-O_long_06", "деток", "DIM"), (15, "RUS-O_short_02", "Зайчишка", "DIM"),
           (16, "RUS-O_short_02", "морковку", "DIM"), (17, "RUS-O_long_15", "Мишутка", "DIM"), (18, "RUS-O_long_07", "глазок", "LEX"),
           (19, "RUS-O_long_06", "глазки", "DIM"), (20, "RUS-O_long_06", "щенята", "NO")]
LEXICON_COLUMNS = ["lemma", "label", "base", "pej", "name", "precedent", "source", "comment"]
# Ложные «имена» pymorphy3 (раздел 6: «сколька», «столька»)
NAME_STOP = {"сколька", "столька", "доминик", "ник", "мик"}
DIM_RULE_RE = re.compile(r"к\(а\)|к\(о\)|очк|ечк|ичк|ушк|ышк|ок|ик|иц|ц\(|оньк|еньк|ишк|ец")
NON_DIM_RULE_RE = re.compile(r"ник|щик|ист|изм|ациj|ор\b|ит\b|^\w+\(не|без|анти|под \+|пере \+|раз \+|за \+")

morph = pymorphy3.MorphAnalyzer()


def e_norm(s):
    return s.replace("ё", "е") if isinstance(s, str) else s


@lru_cache(maxsize=None)
def lemma_parses(word: str, pos_set: frozenset):
    """Разборы, в которых word — начальная форма известной лексемы нужной части речи."""
    out = []
    for p in morph.parse(word):
        if p.is_known and p.tag.POS in pos_set and e_norm(p.normal_form) == e_norm(word):
            out.append(p)
    return out


def stem_variants(stem: str):
    """Основа с обращёнными чередованиями (ручк- → рук-), без ь перед последней согласной
    (бельчонок → белк-) и с ё → е (колёсико → колесо)."""
    vs = {stem}
    for alt in ALTERNATIONS.get(stem[-1:], []):
        vs.add(stem[:-1] + alt)
    for v in list(vs):
        if len(v) >= 3 and v[-2] == "ь":
            vs.add(v[:-2] + v[-1])
    vs |= {e_norm(v) for v in vs}
    return vs


class Reference:
    def __init__(self, path: Path):
        f = pd.read_csv(path)
        self.ipm = f.groupby("lemma_e").ipm.sum().to_dict()

    def __call__(self, w):
        return self.ipm.get(e_norm(w), 0.0)


class DerivBase:
    def __init__(self, zip_path):
        self.pairs = {}
        if not zip_path:
            return
        with zipfile.ZipFile(zip_path) as z:
            for name in ["connections_noun.csv", "connections_adj.csv", "connections_adv.csv", "connections_rare.csv"]:
                d = pd.read_csv(io.BytesIO(z.read(name)), sep="\t")
                d = d[(d.pos_s == d.pos_d) & (d.der_type == "SFX")]
                d = d[d.rule_id.str.contains(DIM_RULE_RE) & ~d.rule_id.str.contains(NON_DIM_RULE_RE)]
                for s, dd, r in zip(d.word_s, d.word_d, d.rule_id):
                    self.pairs.setdefault(e_norm(dd), []).append((s, r))

    def bases(self, derived):
        return self.pairs.get(e_norm(derived), [])


def gram_set(gram):
    return set(re.split(r"[ ,]", gram or ""))


@lru_cache(maxsize=None)
def lexeme_animacy(lemma: str):
    """Одушевлённость лексемы по всем словарным разборам её начальной формы (а не по одному выбранному):
    «мышку» pymorphy3 может разобрать как неодушевлённую (компьютерная мышь), но лексема «мышка» бывает и одушевлённой."""
    out = set()
    for p in morph.parse(lemma):
        if p.is_known and p.tag.POS == "NOUN" and e_norm(p.normal_form) == e_norm(lemma):
            out |= {g for g in ("anim", "inan") if g in p.tag}
    return frozenset(out)


@lru_cache(maxsize=None)
def lexeme_genders(lemma: str):
    out = set()
    for p in morph.parse(lemma):
        if p.is_known and p.tag.POS == "NOUN" and e_norm(p.normal_form) == e_norm(lemma):
            out |= {g for g in ("masc", "femn", "neut", "ms-f") if g in p.tag}
    return frozenset(out)


PROPER = {"Name", "Surn", "Patr", "Geox", "Orgn", "Trad", "Abbr", "Init"}


def compatible_noun(base_parse, d_gram, d_lemma, suf=""):
    """Защита от ложных пар (v3):
    — одушевлённость должна совпадать хотя бы в одном значении (песок ↛ пёс);
    — производящее не имя собственное (мартышка ↛ Марта);
    — муж. одуш. → жен. одуш. — феминитив (медведица), а не диминутив;
    — у неодушевлённых диминутив сохраняет род производящего (бочка ↛ бок, стройка ↛ строй, кувшинка ↛ кувшин);
    — у одушевлённых жен. → муж. не диминутив (птенец ↛ птица)."""
    bg, dg = set(base_parse.tag.grammemes), gram_set(d_gram)
    if bg & PROPER:
        return False, "proper_base"
    b_an = lexeme_animacy(base_parse.normal_form) or frozenset(g for g in ("anim", "inan") if g in bg)
    d_an = lexeme_animacy(d_lemma) or frozenset(g for g in ("anim", "inan") if g in dg)
    if b_an and d_an and not (b_an & d_an):
        return False, "animacy"
    if "anim" in bg and "masc" in bg and "femn" in dg and "ms-f" not in dg and ("anim" in d_an or "anim" in dg):
        return False, "feminitive"
    b_g, d_g = lexeme_genders(base_parse.normal_form), lexeme_genders(d_lemma)
    if b_g and d_g and "ms-f" not in b_g | d_g and not (b_g & d_g):
        if b_an == frozenset({"inan"}) and d_an == frozenset({"inan"}) and suf not in ("ишко", "ище"):
            return False, "gender"
        if "anim" in b_an and b_g == frozenset({"femn"}) and "masc" in d_g:
            return False, "gender"
    return True, ""


VOWELS_SOFT = set("аеёиоуыэюяйь")


def _drop_fleeting(x, k=2):
    """Все варианты x без до k беглых гласных / ь / й (капель- → капл-)."""
    out = {x}
    for _ in range(k):
        out |= {v[:i] + v[i + 1:] for v in out for i in range(len(v)) if v[i] in "еоёьй"}
    return out


def stem_compatible(d_stem, b_stem):
    """Основа производного совместима с основой производящего (нужно для пар из DerivBase, где есть случайные
    пары: крона → крошка, шина → шишка, птица → птенец). Допускается: совпадение; чередование последней согласной
    (зайч- ~ заяц-, с я/й); беглые гласные и ь (капель- ~ капл-); усечение основы производящего (пугов- ~ пуговиц-,
    солн- ~ солнц-); наращение на 1–2 буквы (малюс- ~ мал-); -ош- после усечения (окош- ~ окн-)."""
    d, b = e_norm(d_stem), e_norm(b_stem)
    if len(d) < 2 or len(b) < 2:
        return False
    alts = dict(ALTERNATIONS)
    alts["ш"] = alts.get("ш", []) + ["н"]            # ладонь → ладошка, камень → камешек, окно → окошко
    variants = {d} | {d[:-1] + a for a in alts.get(d[-1:], [])}
    variants = {w for v in variants for w in _drop_fleeting(v)}
    b_vars = _drop_fleeting(b, 1)                      # ковёр → коврик
    vowels = "аеёиоуыэюяйь"
    for v in variants:
        if v in b_vars:
            return True
        for bb in b_vars:
            if len(v) == len(bb):
                diff = [i for i in range(len(v)) if v[i] != bb[i]]
                if len(diff) == 1 and v[diff[0]] in vowels and bb[diff[0]] in vowels:   # улоч- ~ улиц-
                    return True
    if d.endswith(("юс", "ус")) and len(d) >= 4 and b.startswith(d[:-2]) and len(b) - len(d[:-2]) <= 1:
        return True                                    # тонкий → тонюсенький
    if len(d) >= 3 and b.startswith(d) and len(b) - len(d) <= 3:      # усечение: пуговица → пуговка
        return True
    if d.startswith(b) and 0 < len(d) - len(b) <= 2 and len(b) >= 3:     # наращение: малый → малюсенький
        return True
    if len(b) >= 3 and d in (b[:-1] + "ош", b[:-1] + "еш"):               # окно → окошко
        return True
    return False


def base_stem(b):
    b = e_norm(b)
    for end in ("ый", "ий", "ой"):          # прилагательные: малый → мал-
        if b.endswith(end) and len(b) > 4:
            return b[:-2]
    return b[:-1] if b[-1:] in "аяоеьйыи" else b


def find_base(lemma, pos_group, d_gram, is_oov, ref, db):
    """Возвращает (base, suffix, source) или (None, suffix_or_None, причина отказа)."""
    if pos_group == "NOUN":
        suffixes, pos_set = NOUN_SUFFIXES, frozenset({"NOUN"})
    elif pos_group == "ADJ":
        suffixes, pos_set = ADJ_SUFFIXES, frozenset({"ADJF"})
    else:
        suffixes, pos_set = ADV_SUFFIXES, frozenset({"ADVB", "PRED"})
    d_ipm = ref(lemma)

    def ok(p):
        if is_oov:                       # у слов вне словаря теги угаданы — проверки по ним не делаем,
            if set(p.tag.grammemes) & PROPER:      # кроме запрета на имя собственное как производящее (Филик ↛ Фил)
                return False, "proper_base"
            return True, ""
        if pos_group == "NOUN":
            good, why = compatible_noun(p, d_gram, lemma, cur_suf[0])
            if not good:
                return False, why
        if d_ipm > 0 and ref(p.normal_form) * BASE_RATIO < d_ipm:
            return False, "base_rarer"
        return True, ""

    matched, rejected = None, ""
    cur_suf = [""]
    for suf, min_stem, endings in suffixes:
        cur_suf[0] = suf
        if not lemma.endswith(suf) or len(lemma) - len(suf) < min_stem:
            continue
        matched = matched or suf
        stem = lemma[: -len(suf)]
        cands = []
        for sv in stem_variants(stem):
            for k, end in enumerate(endings):
                b = sv + end
                if e_norm(b) == e_norm(lemma):
                    continue
                # P-17: у двухступенчатых -очек/-ечек/-ичек — ближайшее производящее (листочек → листок, не лист)
                prio = 0 if (suf in TWO_STEP and k == 0) else 1
                for p in lemma_parses(b, pos_set):
                    good, why = ok(p)
                    if good:
                        cands.append((prio, -ref(p.normal_form), p.normal_form))
                    else:
                        rejected = why
        if cands:
            return min(cands)[2], suf, "pymorphy"
    if matched:
        best = []
        d_stems = [lemma[: -len(sf)] for sf, ms, _ in suffixes if lemma.endswith(sf) and len(lemma) - len(sf) >= ms]
        for b, rule in db.bases(lemma):
            # v3: пара из DerivBase принимается, только если основы совместимы (крона ↛ крошка, шина ↛ шишка)
            if not any(stem_compatible(ds, base_stem(b)) for ds in d_stems):
                rejected = "derivbase_stem"
                continue
            for p in lemma_parses(b, pos_set):
                if ok(p)[0]:
                    best.append((ref(p.normal_form), p.normal_form))
        if best:
            return max(best)[1], matched, "derivbase"
    return None, matched, rejected


@lru_cache(maxsize=None)
def lexeme_has_name(lemma):
    return any("Name" in p.tag for p in morph.parse(lemma)) or not morph.word_is_known(lemma)


def suffix_in_form(token, lemma):
    """Есть ли в словоформе уменьшительный суффикс леммы. Суффиксальная часть — от последней к/ч/ц леммы;
    если словоформа расходится с леммой раньше неё и дальше в словоформе нет к/ч/ц — суффикса нет
    (цветы ← цветок, меньше ← маленький)."""
    t, l = e_norm(token.lower()), e_norm(lemma.lower())
    j = max(l.rfind(ch) for ch in "кчц")
    if j < 0:
        return True
    k = 0
    while k < min(len(t), len(l)) and t[k] == l[k]:
        k += 1
    if k > j:
        return True
    return any(ch in t[k:] for ch in "кчц")


def hyp_base(lemma):
    """Уменьшительная форма имени: отсечь суффикс и найти полное/краткое имя (Аннушка → Анна, Петька → Петя)."""
    for suf in HYP_SUFFIXES:
        if lemma.endswith(suf) and len(lemma) - len(suf) >= 2:
            stem = lemma[: -len(suf)]
            stems = [stem.rstrip("ь"), stem] + sorted(stem_variants(stem) - {stem})   # без чередования — первым
            for s in stems:
                for end in ["", "а", "я", "ь"]:
                    b = s + end
                    if e_norm(b) == e_norm(lemma):
                        continue
                    for p in morph.parse(b):
                        if p.is_known and "Name" in p.tag and e_norm(p.normal_form) == e_norm(b):
                            return p.normal_form.capitalize(), suf
    return None, None


def classify(lemma, pos, gram, is_oov, lex, ref, db):
    """Этап D: R1 списки → R2 YOUNG → R3 HYP → R4 DIM → R5 NO. Возвращает dict или None (не кандидат)."""
    lem = lemma.lower()
    g = gram_set(gram)
    pos_group = {"NOUN": "NOUN", "ADJF": "ADJ", "ADJS": "ADJ", "ADVB": "ADV"}.get(pos)
    res = {"cand_suffix": None, "base_lemma": None, "base_source": None, "det_label": None, "rule_id": None,
           "pej": 0, "nonce": int(bool(is_oov)), "needs_review": 0, "precedent": None, "reject_reason": None}
    if lex is not None and e_norm(lem) in lex:
        e = lex[e_norm(lem)]
        lab = e["label"]
        res.update(det_label="DIM" if lab == "CONTEXT" else lab, base_lemma=e["base"] or None, base_source="list",
                   rule_id="R1", pej=int(e["pej"]), needs_review=int(lab == "CONTEXT"), precedent=e["precedent"])
        return res
    if pos_group is None:
        return None
    if pos_group == "NOUN" and lem.endswith(YOUNG_SUFFIXES) and "anim" in g and not lem.endswith(("ёночек", "оночек")):
        res.update(det_label="YOUNG", cand_suffix="ёнок" if lem.endswith("ёнок") else "онок", rule_id="R2")
        stem = lem[:-4]   # лисёнок → лиса, медвежонок → медведь, бельчонок → белка
        cands = []
        for sv in stem_variants(stem):
            for end in YOUNG_BASE_ENDINGS:
                # любая форма известного одушевлённого существительного: бобр- → бобёр, орл- → орёл
                for p in morph.parse(sv + end):
                    if p.is_known and p.tag.POS == "NOUN" and "anim" in p.tag and "nomn" in p.tag:
                        cands.append((ref(p.normal_form), p.normal_form))
        if cands:
            res.update(base_lemma=max(cands)[1], base_source="pymorphy")
        return res
    if pos_group == "NOUN" and "Name" in g and lem not in NAME_STOP:
        hb, hsuf = hyp_base(lem)
        if hb:
            res.update(det_label="HYP", cand_suffix=hsuf, base_lemma=hb, base_source="pymorphy", rule_id="R3")
            return res
    base, suf, src = find_base(lem, pos_group, gram, is_oov, ref, db)
    if suf is None:
        return None
    if base is None and src == "proper_base" and pos_group == "NOUN" and lem not in NAME_STOP and lexeme_has_name(lem):
        hb, hsuf = hyp_base(lem)     # v3: производящее — имя (Анютка ← Аня, Тимошка ← Тима) → HYP, а не NO
        if hb:
            res.update(det_label="HYP", cand_suffix=hsuf, base_lemma=hb, base_source="pymorphy", rule_id="R3")
            return res
    res["cand_suffix"] = suf
    if base:
        res.update(det_label="DIM", base_lemma=base, base_source=src, rule_id="R4", pej=int(suf in PEJ_SUFFIXES))
    else:
        res.update(det_label="NO", rule_id="R5", reject_reason=src or None)
    return res


def load_lexicon(path: Path):
    if not path.exists():
        rows = [dict(zip(["lemma", "label", "base", "pej", "name", "precedent", "comment"], r), source="seed v2 (инструкция v0.3)") for r in SEED]
        pd.DataFrame(rows)[LEXICON_COLUMNS].to_csv(path, index=False)
        print(f"[1.2] Создан стартовый список прецедентов: {path}")
    lx = pd.read_csv(path, dtype=str, keep_default_na=False)
    bad = lx[~lx.label.isin(["DIM", "LEX", "HYP", "YOUNG", "NO", "CONTEXT"])]
    if len(bad):
        raise SystemExit(f"Неизвестные метки в {path}: {bad.label.unique().tolist()}")
    lx["pej"] = lx.pej.replace("", "0")
    lx["name"] = lx.name.replace("", "0")
    return {e_norm(r.lemma.lower()): r._asdict() for r in lx.itertuples(index=False)}, lx


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return p, m - h, m + h


def dev_eval(c, gold_path, out):
    """Оценка на dev по сплошному чтению (часть 1): итоговая разметка содержит ВСЕ DIM/HYP/YOUNG книги,
    поэтому и точность, и полнота считаются по ней, для любого набора кандидатов."""
    g = pd.read_csv(gold_path)
    g = g[g.part == 1].copy()
    g["gold"] = g.gold.fillna("")
    key = ["kidlit_id", "sent_id", "tok_id"]
    books = sorted(g.kidlit_id.unique())
    cc = c[c.kidlit_id.isin(books)]
    L = [f"## Оценка на dev ({', '.join(books)})\n",
         "По итоговой разметке сплошного чтения. Это dev: по этим ошибкам детектор дорабатывается; итоговые цифры — только на test.\n",
         "| Метка | Точность | 95% ДИ | Полнота | 95% ДИ | F1 |\n|---|---|---|---|---|---|"]
    rows = []
    for lab in ["DIM", "YOUNG", "HYP"]:
        gs = set(map(tuple, g[g.gold == lab][key].values))
        ds = set(map(tuple, cc[cc.det_label == lab][key].values))
        tp = len(gs & ds)
        P, R = wilson(tp, len(ds)), wilson(tp, len(gs))
        f1 = 2 * P[0] * R[0] / (P[0] + R[0]) if (P[0] == P[0] and R[0] == R[0] and P[0] + R[0]) else float("nan")
        f = lambda x: "—" if x != x else f"{x:.3f}"
        L.append(f"| {lab} | {f(P[0])} ({tp}/{len(ds)}) | {f(P[1])}–{f(P[2])} | {f(R[0])} ({tp}/{len(gs)}) | {f(R[1])}–{f(R[2])} | {f(f1)} |")
        tok = g.set_index(key).token.to_dict()
        for k in ds - gs:
            rows.append({"label": lab, "error": "FP", "kidlit_id": k[0], "sent_id": k[1], "tok_id": k[2], "token": tok.get(k, ""),
                         "gold": g.set_index(key).gold.get(k, "")})
        for k in gs - ds:
            dl = cc.set_index(key).det_label.get(k, "не кандидат")
            rows.append({"label": lab, "error": "FN", "kidlit_id": k[0], "sent_id": k[1], "tok_id": k[2], "token": tok.get(k, ""),
                         "gold": lab, "det_label": dl})
    err = pd.DataFrame(rows)
    err.to_csv(out / "dev_errors.csv", index=False)
    if len(err):
        L.append("\nОшибки на dev (`dev_errors.csv`): " + "; ".join(
            f"{r.error} {r.label}: {r.token}" + (f" (детектор: {r.det_label})" if r.error == "FN" else f" (эталон: {r.gold or '—'})")
            for r in err.itertuples()))
    L.append("")
    return L


def main():
    ap = argparse.ArgumentParser(description="KidLit: детектор диминутивов v3")
    ap.add_argument("--data", default="kidlit_dim")
    ap.add_argument("--derivbase", help="v001.00-connections.zip из DerivBase.Ru")
    ap.add_argument("--lexicon")
    ap.add_argument("--no-lexicon", action="store_true")
    ap.add_argument("--out-name", default="detector")
    ap.add_argument("--dev-gold", help="pilot_gold.csv (и другие dev-файлы той же структуры) — оценка на dev")
    args = ap.parse_args()

    data = Path(args.data)
    out = data / args.out_name
    out.mkdir(exist_ok=True)
    lex_path = Path(args.lexicon) if args.lexicon else data / "detector" / "dim_lexicon.csv"
    lex_path.parent.mkdir(parents=True, exist_ok=True)
    lex, lex_df = load_lexicon(lex_path)
    if args.no_lexicon:
        lex = None
    ref = Reference(data / "reference" / "freqrnc2011_clean.csv")
    db = DerivBase(args.derivbase)
    print(f"[1.2] DerivBase.Ru: {len(db.pairs)} производных с диминутивными правилами" if args.derivbase
          else "[1.2] DerivBase.Ru не подключена")

    tok = pd.read_parquet(data / "data" / "tokens.parquet")
    sents = pd.read_parquet(data / "data" / "sentences.parquet").set_index(["book_id", "sent_id"]).text
    tok["prev_token"] = tok.groupby(["book_id", "sent_id"]).token.shift(1)
    w = tok[tok.is_word].copy()

    cache, rows = {}, []
    for r in w.itertuples(index=False):
        parts = [(r.lemma, r.pos, r.gram, r.is_oov, None)]
        if "-" in r.token_lower.strip("-"):   # P-12: составные через дефис — только по компонентам
            pieces = r.token_lower.split("-")
            parts = []
            if all(len(x) >= 3 for x in pieces):   # «р-р-ребёнок», «тигрё-ё-ёнок» — растяжки, не составные
                for part in pieces:
                    p = morph.parse(part)[0]
                    parts.append((p.normal_form, p.tag.POS, str(p.tag), not morph.word_is_known(part), part))
        best = None
        for lemma, pos, gram, oov, comp in parts:
            key = (lemma, pos, gram, bool(oov), lex is None)
            if key not in cache:
                cache[key] = classify(lemma, pos, gram, oov, lex, ref, db)
            res = cache[key]
            if res is None:
                continue
            res = dict(res, component=comp)
            if best is None or (best["det_label"] == "NO" and res["det_label"] != "NO"):
                best = res
        if best is None:
            continue
        if best["det_label"] in ("DIM", "LEX") and "-" not in r.token_lower and not suffix_in_form(r.token_lower, r.lemma):
            # v3, R0: словоформа без уменьшительного суффикса (цветы ← цветок, меньше ← маленький, P-21)
            best = dict(best, det_label="NO", rule_id="R0", needs_review=0, reject_reason="form_without_suffix")
        rows.append({"book_id": r.book_id, "kidlit_id": r.kidlit_id, "group": r.group, "sent_id": r.sent_id,
                     "tok_id": r.tok_id, "start": r.start, "end": r.end, "token": r.token, "lemma": r.lemma,
                     "pos": r.pos, "gram": r.gram, "is_cap": r.is_cap, "prev_token": r.prev_token, **best})
    c = pd.DataFrame(rows)

    # Атрибут name: лемма хотя бы раз в этой книге написана с заглавной не в начале предложения
    opening = {"—", "–", "-", "«", "\"", "„", "(", "…", "...", ":"}
    mid_cap = c[c.is_cap & (c.tok_id > 0) & ~c.prev_token.isin(opening)]
    names = set(zip(mid_cap.book_id, mid_cap.lemma))
    c["name"] = [int((b, l) in names and lab in ("DIM", "YOUNG")) for b, l, lab in zip(c.book_id, c.lemma, c.det_label)]   # P-25
    if lex is not None:
        forced = {k for k, v in lex.items() if str(v["name"]) == "1"}
        c.loc[c.lemma.map(e_norm).isin(forced) & (c.det_label == "DIM"), "name"] = 1
    c.loc[c.det_label != "DIM", ["pej", "nonce"]] = 0
    c["context"] = [sents.loc[(b, s)].replace("\n", " ") for b, s in zip(c.book_id, c.sent_id)]
    c["detector_version"] = DETECTOR_VERSION + ("-nolex" if lex is None else "")
    cols = ["book_id", "kidlit_id", "group", "sent_id", "tok_id", "start", "end", "token", "lemma", "pos",
            "cand_suffix", "base_lemma", "base_source", "det_label", "rule_id", "pej", "name", "nonce",
            "needs_review", "precedent", "component", "reject_reason", "context", "gram", "detector_version"]
    c = c[cols]
    c.to_parquet(out / "candidates.parquet", index=False)
    c.drop(columns=["gram"]).to_csv(out / "candidates.csv", index=False)

    # Сводка по леммам (для проверки по типам)
    ls = (c.groupby(["lemma", "det_label"], dropna=False)
            .agg(n=("token", "size"), n_books=("book_id", "nunique"),
                 n_rus_o=("group", lambda s: int((s == "RUS-O").sum())),
                 n_rus_t=("group", lambda s: int((s == "RUS-T").sum())),
                 base_lemma=("base_lemma", "first"), base_source=("base_source", "first"),
                 cand_suffix=("cand_suffix", "first"), rule_id=("rule_id", "first"),
                 nonce=("nonce", "max"), name=("name", "max"), needs_review=("needs_review", "max"),
                 example=("context", "first"))
            .reset_index().sort_values("n", ascending=False))
    ls["check_label"] = ""   # для ручной проверки по типам
    ls["check_comment"] = ""
    ls.to_csv(out / "lemma_summary.csv", index=False)

    # Отчёт
    stats = pd.read_csv(data / "data" / "book_stats.csv")
    words = stats.groupby("group").n_words.sum()
    L = [f"# Детектор диминутивов {DETECTOR_VERSION}{' (без списков прецедентов)' if lex is None else ''}: сводка\n",
         f"Кандидатов (словоупотреблений): {len(c)}; лемм: {c.lemma.nunique()}. "
         f"DerivBase.Ru: {'подключена' if args.derivbase else 'не подключена'}. "
         f"Список прецедентов: {'нет' if lex is None else f'{len(lex_df)} записей ({lex_path.name})'}.\n",
         "## Метки\n", "| Метка | Словоупотреблений | Лемм | RUS-O | RUS-T | RUS-O на 1000 слов | RUS-T на 1000 слов |",
         "|---|---|---|---|---|---|---|"]
    for lab, g in c.groupby("det_label"):
        o, t = int((g.group == "RUS-O").sum()), int((g.group == "RUS-T").sum())
        L.append(f"| {lab} | {len(g)} | {g.lemma.nunique()} | {o} | {t} | {o / words['RUS-O'] * 1000:.1f} | {t / words['RUS-T'] * 1000:.1f} |")
    L.append("\nЧастоты на 1000 слов — техническая проверка, не результат: до оценки на золотой выборке их нельзя интерпретировать.\n")
    L.append("## Правила и источники производящего\n")
    rb = c.groupby(["rule_id", "base_source"], dropna=False).size().reset_index(name="n")
    L.append("| Правило | Источник производящего | Словоупотреблений |\n|---|---|---|")
    for _, r in rb.iterrows():
        L.append(f"| {r.rule_id} | {r.base_source if pd.notna(r.base_source) else '—'} | {r.n} |")
    L.append(f"\nНа ручную проверку по контексту (CONTEXT, `needs_review = 1`): {int(c.needs_review.sum())} словоупотреблений.")
    L.append(f"Атрибуты у DIM: pej — {int(c.pej.sum())}, name — {int(c[c.det_label == 'DIM'].name.sum())}, nonce — {int(c.nonce.sum())}.\n")
    for lab in ["DIM", "LEX", "HYP", "YOUNG", "NO"]:
        top = ls[ls.det_label == lab].head(40)
        if len(top):
            L.append(f"## Самые частотные леммы: {lab}\n")
            L.append(", ".join(f"{r.lemma} ({r.n}{' → ' + str(r.base_lemma) if pd.notna(r.base_lemma) and lab != 'NO' else ''})" for r in top.itertuples()))
            L.append("")
    L.append("## Самопроверка на контрольных примерах инструкции\n")
    L.append("Это проверка работоспособности, а не оценка качества: часть этих слов есть в списке прецедентов.\n")
    L.append("| № | Слово | Ключ | Детектор | На проверку | Производящее | Правило |\n|---|---|---|---|---|---|---|")
    hits = 0
    for n, kid, form, gold in CONTROL:
        x = c[(c.kidlit_id == kid) & (c.token == form)]
        lab = x.det_label.iloc[0] if len(x) else "не кандидат"
        hits += lab == gold
        L.append(f"| {n} | {form} | {gold} | {lab}{'' if lab == gold else ' ✗'} | {'да' if len(x) and x.needs_review.iloc[0] else ''} | "
                 f"{x.base_lemma.iloc[0] if len(x) and pd.notna(x.base_lemma.iloc[0]) else ''} | {x.rule_id.iloc[0] if len(x) else ''} |")
    L.append(f"\nСовпало с ключом: {hits} из {len(CONTROL)}.")
    if args.dev_gold:
        L.extend(dev_eval(c, Path(args.dev_gold), out))
    (out / "detector_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[1.2] Готово: {len(c)} кандидатов → {out}")


if __name__ == "__main__":
    main()
