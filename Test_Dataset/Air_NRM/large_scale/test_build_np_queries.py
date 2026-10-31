"""Offline checks for circulation cardinality and query/label integrity."""

import csv
import itertools
import tempfile
import unittest
from pathlib import Path

from build_np_queries import (
    FIELDS, existing_labels, maximum_circulation, normalize_time,
    parse_flights, time_window,
)


def brute_force_cardinality(flights):
    best = 0
    airports = {a for f in flights for a in f[:2]}
    for mask in itertools.product((0, 1), repeat=len(flights)):
        chosen = [f for f, y in zip(flights, mask) if y]
        if all(sum(f[0] == a for f in chosen) == sum(f[1] == a for f in chosen)
               for a in airports):
            best = max(best, len(chosen))
    return best


class QueryTests(unittest.TestCase):
    def test_circulation_against_exhaustive_search(self):
        # A 3-cycle is essential: pairing reverse ODs alone is not a general
        # maximum-circulation algorithm. Uneven parallel flights test counts.
        graphs = [
            [("A", "B", "01:00"), ("B", "C", "02:00")],
            [("A", "B", "01:00"), ("B", "A", "02:00")],
            [("A", "B", "01:00"), ("B", "C", "02:00"), ("C", "A", "03:00")],
            [("A", "B", "01:00"), ("A", "B", "02:00"), ("B", "A", "03:00"),
             ("B", "C", "04:00"), ("C", "A", "05:00"), ("D", "A", "06:00")],
        ]
        for flights in graphs:
            with self.subTest(flights=flights):
                b, witness = maximum_circulation(flights)
                self.assertEqual(b, brute_force_cardinality(flights))
                self.assertEqual(b, len(witness))
                self.assertTrue(set(witness) <= set(flights))

    def test_parser_rejects_partial_and_duplicate_flights(self):
        valid = "(OD = ('865', '678') AND Departure Time='01:30')"
        self.assertEqual(parse_flights(valid), [("865", "678", "01:30")])
        for query in [valid + ", " + valid,
                      valid + ", (OD = ('678', '865') AND Departure Time='broken')"]:
            with self.assertRaises(ValueError):
                parse_flights(query)

    def test_time_boundaries(self):
        self.assertEqual(normalize_time("8:00"), "08:00")
        for t, expected in [("00:00", "(10pm~8am)"), ("07:59", "(10pm~8am)"),
                            ("08:00", "(8am~12pm)"), ("12:00", "(12pm~6pm)"),
                            ("18:00", "(6pm~10pm)"), ("22:00", "(10pm~8am)")]:
            self.assertEqual(time_window(t), expected)
        with self.assertRaises(ValueError):
            normalize_time("24:00")

    def test_labels_preserved_only_when_query_and_size_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="np-query-test-") as directory:
            path = Path(directory) / "query.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow({"n_variables": "174", "Query": "original", "Label-objective": "123.45"})
            rows = [{"n_variables": "174", "Query": "original", "Label-objective": ""}]
            existing_labels(path, rows)
            self.assertEqual(rows[0]["Label-objective"], "123.45")
            for field, value in [("Query", "changed"), ("n_variables", "175")]:
                changed = [{**rows[0], field: value}]
                with self.assertRaises(ValueError):
                    existing_labels(path, changed)


if __name__ == "__main__":
    unittest.main()
