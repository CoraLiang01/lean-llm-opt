#!/usr/bin/env python3
"""
fix_refdata.py
==============
Large_Scale_Or_Files/RefData.csv has a .csv name but xlsx bytes (it starts with
the ZIP magic "PK"), so pandas.read_csv and langchain's CSVLoader both fail with
UnicodeDecodeError. Most likely someone opened it in Excel and saved it back
with the workbook format while keeping the .csv file name.

This script converts it to a real UTF-8 CSV in place, keeping the file name so
that no notebook code has to change.

    python fix_refdata.py            # convert (keeps a byte-exact backup)
    python fix_refdata.py --check    # report only, write nothing

Backup: the original bytes are copied to Large_Scale_Or_Files/RefData.xlsx
(correct extension). To undo: cp RefData.xlsx RefData.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

TARGET = Path(__file__).parent / "Large_Scale_Or_Files" / "RefData.csv"
BACKUP = TARGET.with_suffix(".xlsx")


def sniff(path: Path) -> str:
    head = path.read_bytes()[:4]
    if head[:2] == b"PK":
        return "xlsx"
    return "text"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not TARGET.exists():
        print(f"not found: {TARGET}")
        return 1

    kind = sniff(TARGET)
    print(f"{TARGET.name}: content is {kind}")

    if kind == "text":
        df = pd.read_csv(TARGET)
        print(f"already a real CSV: {df.shape[0]} rows x {df.shape[1]} cols "
              f"-- nothing to do")
        return 0

    df = pd.read_excel(TARGET)
    print(f"parsed as xlsx: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"columns: {list(df.columns)}")

    if args.check:
        print("\n--check: no files written")
        return 0

    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"backup of the original bytes -> {BACKUP.name}")
    else:
        print(f"backup already exists, leaving it alone: {BACKUP.name}")

    df.to_csv(TARGET, index=False, encoding="utf-8")
    print(f"rewritten as UTF-8 CSV: {TARGET}")

    # -- verify with the two readers the notebooks actually use -------------
    back = pd.read_csv(TARGET)
    assert back.shape == df.shape, (back.shape, df.shape)
    print(f"pandas.read_csv OK: {back.shape}")

    try:
        from langchain_community.document_loaders.csv_loader import CSVLoader
        docs = CSVLoader(file_path=str(TARGET), encoding="utf-8").load()
        print(f"langchain CSVLoader OK: {len(docs)} documents")
        print("\nfirst document (what the classification agent will retrieve):")
        print("-" * 70)
        print(docs[0].page_content[:400])
        print("-" * 70)
    except ImportError:
        print("langchain_community not importable here; skipped that check")

    return 0


if __name__ == "__main__":
    sys.exit(main())
