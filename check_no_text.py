#!/usr/bin/env python3
"""Check that the repository contains no fragments of the book texts.

    python tools/check_no_text.py --texts /path/to/kidlit_dim/data/texts [--n 5] [--report reports/text_leak_check.md]

The book texts are NOT part of the repository; run this script locally against your own copy
of the corpus before every release. Every text cell of every .csv/.xlsx/.json/.md/.py/.docx file
is compared with the books: any sequence of N consecutive words (default 5) that also occurs in a
book is reported. Exit code 1 if anything is found outside the allow-list.
"""
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
SKIP_DIRS = {".git", "kidlit_dim", "__pycache__"}


def words(s):
    return WORD.findall(s.lower().replace("ё", "е"))


def shingles(ws, n):
    # repetitions of one word («у у у у у», from interjections such as у-у-у) are not text fragments
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1) if len(set(ws[i:i + n])) > 1}


def file_strings(p: Path):
    suf = p.suffix.lower()
    if suf == ".csv":
        import csv
        with open(p, encoding="utf-8", errors="ignore", newline="") as fh:
            for row in csv.reader(fh):
                yield from row
    elif suf in {".md", ".py", ".json", ".txt", ".cff", ".yml", ".yaml", ".js", ".html"}:
        yield p.read_text(encoding="utf-8", errors="ignore")
    elif suf == ".xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(p, read_only=True)
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                for v in row:
                    if isinstance(v, str):
                        yield v
    elif suf == ".pdf":
        import shutil, subprocess
        if shutil.which("pdftotext"):
            yield subprocess.run(["pdftotext", "-layout", str(p), "-"], capture_output=True, text=True).stdout
        else:
            print(f"warning: pdftotext not found, {p.name} not checked")
    elif suf == ".docx":
        with zipfile.ZipFile(p) as z:
            xml = z.read("word/document.xml").decode("utf-8")
        for para in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
            yield "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", para))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, help="folder with <book_id>.txt (local copy of the corpus)")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--report", default="reports/text_leak_check.md")
    ap.add_argument("--allow", default="tools/text_check_allowlist.json",
                    help="JSON list of files whose quotations were reviewed and are allowed (e.g. the guide)")
    args = ap.parse_args()
    book = {}
    for f in sorted(Path(args.texts).glob("*.txt")):
        for s in shingles(words(f.read_text(encoding="utf-8")), args.n):
            book.setdefault(s, f.stem)
    allow = set(json.loads((ROOT / args.allow).read_text())) if (ROOT / args.allow).exists() else set()
    hits = {}
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or set(p.relative_to(ROOT).parts) & SKIP_DIRS:
            continue
        rel = str(p.relative_to(ROOT))
        if rel == args.report:
            continue
        found = []
        for s in file_strings(p):
            for sh in shingles(words(s), args.n):
                if sh in book:
                    found.append((sh, book[sh]))
        if found:
            hits[rel] = sorted(set(found))
    L = [f"# Text leak check (n = {args.n} words)\n", f"Books: {len(list(Path(args.texts).glob('*.txt')))}.\n"]
    bad = 0
    for rel, f in hits.items():
        status = "allowed (allow-list: book titles)" if rel in allow else "NOT ALLOWED"
        bad += rel not in allow
        L.append(f"## {rel} — {len(f)} matches, {status}\n")
        # the report itself must not quote the books: only the first word of each match is shown
        L += [f"- {s.split()[0]} … ({b})" for s, b in f[:200]]
        L.append("")
        if rel not in allow:
            print(f"{rel}:", *[f"«{s}» ({b})" for s, b in f[:50]], sep="\n  ")
    if not hits:
        L.append("No matches.")
    (ROOT / args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Files with matches: {len(hits)}, not allowed: {bad}. Report: {args.report}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
