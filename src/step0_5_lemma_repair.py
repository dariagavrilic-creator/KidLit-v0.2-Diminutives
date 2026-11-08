#!/usr/bin/env python3
"""
KidLit by Numbers — шаг 0.3 v2: исправление лемм после предобработки.

    python src/step0_5_lemma_repair.py --data kidlit_dim

Правило: если леммы словоупотребления нет в
частотном словаре Ляшевской–Шарова, а у словоформы есть другой разбор pymorphy3 той же части речи с леммой из
словаря, берётся этот разбор (при нескольких — с наибольшей частотой в словаре). Для лемм на «-ие» проверяется
вариант на «-ье». Слова с заглавной буквы не в начале предложения и имена собственные не трогаются.

На KidLit v0.2 исправлено 260 словоупотреблений (54 пары лемм, reference/lemma_repairs.csv).
Исходные значения сохраняются в колонках lemma_pm, lemma_e_pm, pos_pm, gram_pm, pm_rank_pm; lemma_fixed = True
у исправленных. Скрипт перезаписывает data/tokens.parquet и пересчитывает reference/kidlit_lemma_freq.csv не
трогает (его строит шаг 0.4 до исправления; keyness считается по исправленным леммам).
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import pymorphy3

PROPER_GRAMMEMES = {"Name", "Surn", "Patr", "Geox", "Orgn", "Trad"}


def e_norm(s):
    return s.replace("ё", "е").replace("Ё", "Е") if isinstance(s, str) else s


def pm_to_rnc(pos, gram, pos_ud=None):
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


def key_pos(pos, gram, ud=None):
    k = pm_to_rnc(pos, gram, ud)
    return "s" if k == "s.PROP" else k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kidlit_dim")
    args = ap.parse_args()
    data = Path(args.data)
    tok = pd.read_parquet(data / "data" / "tokens.parquet")
    if "lemma_fixed" in tok:
        raise SystemExit("Леммы уже исправлены (есть колонка lemma_fixed).")
    ref = pd.read_csv(data / "reference" / "freqrnc2011_clean.csv")
    ref["key_pos"] = ref.pos_rnc.replace({"s.PROP": "s"})
    ref_lp = ref.groupby(["lemma_e", "key_pos"]).approx_count.sum().to_dict()
    ref_lemmas = set(ref.lemma_e)
    morph = pymorphy3.MorphAnalyzer()

    for c in ["lemma", "lemma_e", "pos", "gram", "pm_rank"]:
        tok[f"{c}_pm"] = tok[c]
    tok["lemma_fixed"] = False
    cache = {}
    for i, r in tok[tok.is_word].iterrows():
        lem = e_norm(str(r.lemma).lower())
        if lem in ref_lemmas:
            continue
        is_cap = r.token[:1].isupper()
        if (is_cap and r.tok_id > 0) or (is_cap and set(re.split(r"[ ,]", r.gram or "")) & PROPER_GRAMMEMES):
            continue
        kp = key_pos(r.pos, r.gram, r.get("pos_ud"))
        key = (r.token.lower(), lem, kp)
        if key not in cache:
            best = None
            for p in morph.parse(r.token.lower()):
                nf = e_norm(p.normal_form)
                if nf != lem and key_pos(p.tag.POS, str(p.tag)) == kp and (nf, kp) in ref_lp:
                    cand = (ref_lp[(nf, kp)], p)
                    if best is None or cand[0] > best[0]:
                        best = cand
            if best is None and lem.endswith("ие") and (lem[:-2] + "ье", kp) in ref_lp:
                for p in morph.parse(r.token.lower()):
                    if e_norm(p.normal_form) == lem[:-2] + "ье":
                        best = (0, p)
                        break
            cache[key] = best[1] if best else None
        p = cache[key]
        if p is not None:
            tok.at[i, "lemma"], tok.at[i, "lemma_e"] = p.normal_form, e_norm(p.normal_form)
            tok.at[i, "pos"], tok.at[i, "gram"], tok.at[i, "lemma_fixed"] = p.tag.POS, str(p.tag), True
    fixed = tok[tok.lemma_fixed]
    pairs = fixed.groupby(["lemma_pm", "lemma"]).size().reset_index(name="n").sort_values("n", ascending=False)
    pairs.to_csv(data / "reference" / "lemma_repairs.csv", index=False)
    tok.to_parquet(data / "data" / "tokens.parquet", index=False)
    print(f"[0.3 v2] Исправлено словоупотреблений: {len(fixed)}, пар лемм: {len(pairs)}")


if __name__ == "__main__":
    main()
