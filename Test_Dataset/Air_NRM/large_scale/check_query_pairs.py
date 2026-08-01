"""Check that every (OD, Departure Time) in the query file exists in flight.csv.

The test is row-level co-occurrence, not two independent lookups: for a query
clause

    (OD = ('865', '678') AND Departure Time='01:30')

there must be at least one row of flight.csv whose ``Oneway_OD`` column holds
``('865', '678')`` AND whose ``Departure Time`` column holds ``01:30`` on that
same row. An OD that exists and a time that exists but never together is a
FAILURE.

Stdlib only — no pandas, no scipy.

Usage:
    python check_query_pairs.py
    python check_query_pairs.py <query.csv> <flight.csv>

Exit code 0 = all pairs found, 1 = something is missing or malformed.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Matches one clause and captures the OD tuple text and the HH:MM time.
CLAUSE = re.compile(
    r"\(OD\s*=\s*(\('[^']*',\s*'[^']*'\))\s*AND\s*Departure Time='(\d{2}:\d{2})'\)"
)
# Used only to confirm the regex above did not silently skip a clause.
LOOSE = "AND Departure Time="


def load_flight_rows(path: Path) -> tuple[set[tuple[str, str]], Counter, set[str], set[str]]:
    """Return the set of (Oneway_OD, Departure Time) pairs that share a row."""
    pairs: set[tuple[str, str]] = set()
    products: Counter = Counter()
    ods: set[str] = set()
    times: set[str] = set()

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for field in ("Oneway_OD", "Departure Time"):
            if field not in reader.fieldnames:
                sys.exit(f"ERROR: {path.name} has no '{field}' column; "
                         f"found {reader.fieldnames}")
        for row in reader:
            od = row["Oneway_OD"].strip()
            time = row["Departure Time"].strip()
            pairs.add((od, time))
            products[(od, time)] += 1
            ods.add(od)
            times.add(time)
    return pairs, products, ods, times


def load_query_clauses(path: Path) -> list[list[tuple[str, str]]]:
    """One list of (OD, time) clauses per query row."""
    out: list[list[tuple[str, str]]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "Query" not in reader.fieldnames:
            sys.exit(f"ERROR: {path.name} has no 'Query' column; "
                     f"found {reader.fieldnames}")
        for i, row in enumerate(reader, start=1):
            text = row["Query"]
            found = CLAUSE.findall(text)
            expected = text.count(LOOSE)
            if len(found) != expected:
                sys.exit(
                    f"ERROR: query {i} contains {expected} clauses but the "
                    f"pattern matched only {len(found)} — a clause is written "
                    f"in an unexpected format and would be skipped silently."
                )
            out.append([(od.strip(), t.strip()) for od, t in found])
    return out


def main() -> int:
    query_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "query_largescale_CA.csv"
    flight_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "flight.csv"
    for p in (query_path, flight_path):
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")

    valid, products, known_ods, known_times = load_flight_rows(flight_path)
    queries = load_query_clauses(query_path)

    print(f"flight.csv : {len(valid)} distinct (Oneway_OD, Departure Time) rows, "
          f"{len(known_ods)} ODs")
    print(f"{query_path.name} : {len(queries)} queries")
    print()

    failures: list[tuple[int, str, str, str]] = []
    total = 0

    for qi, clauses in enumerate(queries, start=1):
        missing = []
        for od, time in clauses:
            total += 1
            if (od, time) in valid:
                continue
            # Diagnose *why* it is missing, so a typo is distinguishable from a
            # genuinely non-existent departure.
            if od not in known_ods:
                why = "OD does not appear in flight.csv at all"
            elif time not in known_times:
                why = "departure time does not appear in flight.csv at all"
            else:
                why = ("OD and time both exist, but never on the same row "
                       "— this OD does not fly at this time")
            missing.append((od, time, why))

        seen = Counter(clauses)
        dupes = [p for p, n in seen.items() if n > 1]

        status = "OK" if not missing and not dupes else "FAIL"
        print(f"q{qi:<3} clauses={len(clauses):<4} missing={len(missing):<3} "
              f"duplicate={len(dupes):<3} {status}")
        for od, time, why in missing:
            print(f"      MISSING  (OD = {od} AND Departure Time='{time}')  -> {why}")
            failures.append((qi, od, time, why))
        for od, time in dupes:
            print(f"      DUPLICATE (OD = {od} AND Departure Time='{time}') "
                  f"appears {seen[(od, time)]}x in the same query")
            failures.append((qi, od, time, "duplicated within the query"))

    used = {p for clauses in queries for p in clauses}
    print()
    print(f"clause occurrences checked : {total}")
    print(f"distinct pairs used        : {len(used)} of {len(valid)} available")

    # Every real departure sells all four fare families, so a matched pair
    # should back exactly 4 rows of flight.csv. Anything else means the query
    # would generate a different number of variables than expected.
    odd = {p: products[p] for p in used if products.get(p) != 4}
    if odd:
        print(f"WARNING: {len(odd)} pair(s) do not have exactly 4 product rows:")
        for p, n in list(odd.items())[:10]:
            print(f"      {p} -> {n} rows")

    print()
    if failures:
        print(f"RESULT: FAIL — {len(failures)} problem(s) above.")
        return 1
    print("RESULT: PASS — every (OD, Departure Time) in the queries appears "
          "together on a row of flight.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
