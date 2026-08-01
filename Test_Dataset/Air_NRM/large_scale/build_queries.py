"""Generate query_CA.csv for the sq_direct_2city3 dataset.

Mirrors ``Test_Dataset/Air_NRM/query_CA.csv`` in wording and shape, scaled up so
every query models more than 100 SBLP variables.

Variable count of a query, verified against the reference labels
(``SBLP_CA_Label/15.lp`` = 19 pairs x 2 products + 4 ODs = 42):

    n_variables = n_products * n_flight_pairs + n_distinct_ODs

with ``n_products = 4`` here (the reference has 2), so 26 pairs over 8 ODs
already clears 100.

Capacity consumption per product is set from the observed fare ladder — see
CONSUMPTION below.

Run:  python sq_direct_2city3/build_queries.py
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "query_largescale_CA.csv"

SEED = 20250729

# ── products, cheapest to dearest ─────────────────────────────────────────────
# Singapore Airlines' published economy ladder is Lite -> Value -> Standard ->
# Flexi, which is also the order of observed median fare in flight.csv
# (134.7 / 218.0 / 312.1 / 591.2 SGD).
PRODUCTS_BY_FARE = ["Eco_lite", "Eco_value", "Eco_standard", "Eco_flexi"]

# Capacity consumption a_p, one set per query, rotating over the reference
# file's three Eco_flexi anchors (2, 3, 2.2) with Eco_lite pinned at 1.
#
# The two middle products are placed by log-fare interpolation,
#     a_p = 1 + (log f_p - log f_lite) / (log f_flexi - log f_lite) * (anchor - 1),
# so the spacing tracks the fare ladder while the total spread stays inside the
# reference's 2-3x band. The raw fare ratio is 4.39x, which would be too wide.
#
_LADDER_BY_ANCHOR = [
    {"Eco_flexi": 2.0, "Eco_standard": 1.6, "Eco_value": 1.3, "Eco_lite": 1.0},
    {"Eco_flexi": 3.0, "Eco_standard": 2.1, "Eco_value": 1.7, "Eco_lite": 1.0},
    {"Eco_flexi": 2.2, "Eco_standard": 1.7, "Eco_value": 1.4, "Eco_lite": 1.0},
]

CONSUMPTION = [
    {p: f"{a:g}" for p, a in ladder.items()} for ladder in _LADDER_BY_ANCHOR
]

# ── capacity haircut ──────────────────────────────────────────────────────────
# Capacity is 'Y Seats/Week', per week, matching od_demand's per-week Avg Pax.
# (The reference's hard-coded 187 is a per-departure number set against a weekly
# market, which pins every capacity constraint tight — all seven flights in
# SBLP_CA_Label/1.lp solve to 187/2.)
#
# At full 'Y Seats/Week' the opposite happens: only ~9% of capacity constraints
# bind and capacity stops being a lever at all. The haircut below fixes that
# without touching the fare-proportional consumption ladder, which keeps
# Eco_lite at the reference's baseline of 1 unit.
#
# 0.28 is not a free knob: build_report.json records ticketed pax at 28% of the
# economy seats offered over the offer-set window (993,419 against 3,530,860).
# The feed covers 7 points of sale, so only that slice of each cabin is
# effectively available to the market od_demand describes. Sweeping the haircut
# over the 15 instances gives the share of binding capacity constraints:
#
#     1.0  9%   0.5  24%   1/3  39%   0.28  47%   0.25  51%   0.2  61%
CAPACITY_FRACTION = 0.28
CAPACITY_PCT = f"{CAPACITY_FRACTION:.0%}"

# (n_flight_pairs, n_distinct_ODs) per query -> 4*pairs + ODs variables.
LADDER = [
    (26, 8), (30, 9), (34, 10), (39, 11), (44, 12),
    (49, 13), (54, 14), (59, 16), (64, 17), (69, 19),
    (74, 21), (80, 23), (86, 25), (92, 27), (98, 29),
]

TEMPLATE = (
    "Based on all flight ticket choices in 'flight.csv' and 'od_demand.csv' "
    "with attraction values in v1 and shadow attraction value ratios in v2, "
    "develop the SBLP(sales-based linear programming) formulation for among "
    "flights {flights}  that maximize the total revenue of flight ticket sales. "
    "The SBLP should include decision variables, objective function, balance "
    "constraints, scale constraints, nonnegative constraints and selection "
    "constraints. Each flight's capacity is {cap_pct} of its weekly economy "
    "capacity 'Y Seats/Week' in 'flight_capacity.csv', matched on (Oneway_OD, "
    "Departure Time) — the seven points of sale this dataset covers account for "
    "{cap_pct} of the economy cabin. Please consider that each {a_flexi_name} "
    "ticket consumes {a_flexi} units of flight capacity, each {a_std_name} "
    "ticket consumes {a_std} units, each {a_val_name} ticket consumes {a_val} "
    "units and each {a_lite_name} ticket consumes {a_lite} unit of capacity"
)


def load_pool() -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """Selectable (OD, departure time) pairs, grouped by OD.

    Drops the two all-premium SIA 35/36 departures: their ``Y Seats/Week`` is 0,
    which would pin every variable on that flight to zero.
    """
    flight = pd.read_csv(_HERE / "flight.csv")
    cap = pd.read_csv(_HERE / "flight_capacity.csv")

    usable = cap[cap["Y Seats/Week"] > 0][["Oneway_OD", "Departure Time"]]
    pairs = (
        flight[["Oneway_OD", "Departure Time"]]
        .drop_duplicates()
        .merge(usable, on=["Oneway_OD", "Departure Time"], how="inner")
    )
    by_od: dict[str, list[str]] = {}
    for od, time in zip(pairs["Oneway_OD"], pairs["Departure Time"]):
        by_od.setdefault(od, []).append(time)
    for od in by_od:
        by_od[od].sort()
    return by_od, list(zip(pairs["Oneway_OD"], pairs["Departure Time"]))


TIME_WINDOWS = [
    ("(12pm~6pm)", 12 * 60, 18 * 60),
    ("(6pm~10pm)", 18 * 60, 22 * 60),
    ("(10pm~8am)", 22 * 60, 8 * 60),  # wraps midnight
    ("(8am~12pm)", 8 * 60, 12 * 60),
]


def window_of(time: str) -> str:
    minutes = int(time[:2]) * 60 + int(time[3:])
    for label, lo, hi in TIME_WINDOWS:
        if lo < hi:
            if lo <= minutes < hi:
                return label
        elif minutes >= lo or minutes < hi:
            return label
    raise ValueError(time)


def zero_ratio_pairs(by_od: dict[str, list[str]]) -> list[tuple[str, str]]:
    """(OD, time) pairs whose v2 ratio is exactly 0 for at least one product.

    Their term drops out of the balance constraint entirely — the same edge case
    the reference labels carry (``('B','A')`` Eco_lite at 15:40). Keeping at
    least one per query preserves it at the new scale.
    """
    v2 = pd.read_csv(_HERE / "v2.csv")
    cells = [c for c in v2.columns if c not in ("OD Pairs", "no_purchase")]
    out = []
    for _, row in v2.iterrows():
        od = row["OD Pairs"]
        if od not in by_od:
            continue
        windows = {c.split("*")[1] for c in cells if float(row[c]) == 0.0}
        out += [(od, t) for t in by_od[od] if window_of(t) in windows]
    return out


def pick(rng: random.Random, by_od: dict[str, list[str]], n_pairs: int,
         n_ods: int, must_include: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Choose ``n_pairs`` departures spread over exactly ``n_ods`` ODs."""
    seed_pair = rng.choice(must_include)
    ods = [seed_pair[0]]

    # Prefer ODs that fly several banks, so n_pairs is reachable from n_ods.
    candidates = sorted(
        (od for od in by_od if od != seed_pair[0]),
        key=lambda od: (-len(by_od[od]), od),
    )
    rich = candidates[: max(n_ods * 3, 40)]
    ods += rng.sample(rich, n_ods - 1)

    # The random draw can land on too many single-departure ODs to reach
    # n_pairs. Swap the thinnest picks for the fattest unpicked ones until the
    # selected ODs can actually supply n_pairs departures between them.
    def capacity(selected: list[str]) -> int:
        return sum(len(by_od[od]) for od in selected)

    spare = [od for od in candidates if od not in set(ods)]
    spare.sort(key=lambda od: (-len(by_od[od]), od))
    while capacity(ods) < n_pairs and spare:
        thin = min(
            (od for od in ods if od != seed_pair[0]),
            key=lambda od: (len(by_od[od]), od),
        )
        fat = spare.pop(0)
        if len(by_od[fat]) <= len(by_od[thin]):
            break
        ods[ods.index(thin)] = fat
    if capacity(ods) < n_pairs:
        raise RuntimeError(
            f"{n_ods} ODs can supply at most {capacity(ods)} departures, "
            f"need {n_pairs}"
        )

    chosen = [seed_pair]
    remaining = {od: [t for t in by_od[od] if (od, t) not in chosen] for od in ods}
    # One departure per OD first, so every OD really contributes a variable.
    for od in ods:
        if od == seed_pair[0] and not remaining[od]:
            continue
        if od != seed_pair[0]:
            time = rng.choice(remaining[od])
            remaining[od].remove(time)
            chosen.append((od, time))

    while len(chosen) < n_pairs:
        pool = [od for od in ods if remaining[od]]
        if not pool:
            raise RuntimeError(
                f"cannot reach {n_pairs} departures from {n_ods} ODs"
            )
        od = rng.choice(pool)
        time = rng.choice(remaining[od])
        remaining[od].remove(time)
        chosen.append((od, time))

    rng.shuffle(chosen)
    return chosen


def render(flights: list[tuple[str, str]], consumption: dict[str, str]) -> str:
    clause = ", ".join(
        f"(OD = {od} AND Departure Time='{time}')" for od, time in flights
    )
    return TEMPLATE.format(
        flights=clause + ",",
        cap_pct=CAPACITY_PCT,
        a_flexi_name="Eco_flexi", a_flexi=consumption["Eco_flexi"],
        a_std_name="Eco_standard", a_std=consumption["Eco_standard"],
        a_val_name="Eco_value", a_val=consumption["Eco_value"],
        a_lite_name="Eco_lite", a_lite=consumption["Eco_lite"],
    )


def main() -> None:
    rng = random.Random(SEED)
    by_od, _ = load_pool()
    must = zero_ratio_pairs(by_od)
    if not must:
        raise RuntimeError("no v2 == 0 cell found; the edge case is gone")

    rows, counts = [], []
    for i, (n_pairs, n_ods) in enumerate(LADDER):
        flights = pick(rng, by_od, n_pairs, n_ods, must)
        rows.append(render(flights, CONSUMPTION[i % len(CONSUMPTION)]))
        # n_products * n_flight_pairs + one no-purchase variable per OD.
        n_ods_used = len({od for od, _ in flights})
        n_vars = len(PRODUCTS_BY_FARE) * len(flights) + n_ods_used
        counts.append(n_vars)
        print(f"q{i + 1:>2}: pairs={len(flights):>3}  ODs={n_ods_used:>2}  "
              f"vars={n_vars:>3}")

    pd.DataFrame({"n_variables": counts, "Query": rows}).to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(rows)} queries)")


if __name__ == "__main__":
    main()
