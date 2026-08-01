"""Build Air_NRM SBLP inputs (flight / od_demand / v2) from SIA raw data.

Standalone — imports nothing from the Django app and writes nothing outside
this folder. The simulator code base is read-only input here: the economy
fare-family definition is *extracted* from
``api/data_generation/simulation_sales_generator.py::FARE_FAMILY_POLICY``
via ``ast`` so the two can never drift.

Outputs (same schema as the reference files in ``Air_NRM/``):

  flight.csv           index, Oneway_OD, Departure Time, Oneway_Product, Avg Price
  od_demand.csv        index, Oneway_OD, Avg Pax
  v1.csv               OD Pairs, <product>*(<window>) x 16, no_purchase
                       -- GAM attraction values v_j and v_0
  v2.csv               same grid -- GAM shadow attraction ratios vtilde/v
  flight_capacity.csv  Oneway_OD, Departure Time, Aircraft Type, Y Seats, ...

Plus diagnostics under ``Supplement/``: ``flight_price_coverage.csv``,
``od_market_size.csv``, ``v1_cell_status.csv``, ``offer_sets.csv`` and
``airport_code_map.csv``, and ``build_report.json`` alongside this file.

Airport and city codes are replaced by 3-digit surrogates (see ``CENSOR``).
The real-code key is written to ``airport_code_map.csv``, which is the only
file that must be withheld when sharing the outputs.

Run from anywhere (paths resolve relative to this file)::

    python Air_NRM/sq_direct_2city/build_air_nrm_inputs.py
"""

from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────── paths
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent                                   # booking_simulator/
RAW = _REPO / "app" / "backend" / "api" / "Raw_data"
SSG_PY = (_REPO / "app" / "backend" / "api" / "data_generation"
          / "simulation_sales_generator.py")

_SUPP = _HERE / "Supplement"

OUT_FLIGHT = _HERE / "flight.csv"
OUT_DEMAND = _HERE / "od_demand.csv"
OUT_V1 = _HERE / "v1.csv"
OUT_V2 = _HERE / "v2.csv"
OUT_CAPACITY = _HERE / "flight_capacity.csv"
OUT_REPORT = _HERE / "build_report.json"
OUT_PRICE_COV = _SUPP / "flight_price_coverage.csv"
OUT_MARKET = _SUPP / "od_market_size.csv"
OUT_CODE_MAP = _SUPP / "airport_code_map.csv"
OUT_V1_STATUS = _SUPP / "v1_cell_status.csv"
OUT_OFFER_SETS = _SUPP / "offer_sets.csv"

# ───────────────────────────────────────────────────────── zero-cell filling
# A (product, window) cell is zero whenever the OD has no SQ departure in that
# window — 44% of the matrix. Rather than leave a hard zero (which asserts the
# product could never be chosen), fill it from a quasi-independence fit; see
# fit_window_effect. Imputed cells are *counterfactual*: what the product would
# attract if SQ flew that bank. Observed cells and no_purchase are left exact,
# so rows sum to more than 1 by design.
IMPUTE_ZERO_CELLS = True
IMPUTE_TOL = 1e-12
IMPUTE_MAX_ITER = 5000
V_DECIMALS = 6

# ─────────────────────────────────────────────── shadow attraction (v2.csv)
# GAM (Gallego, Ratliff & Shebalov 2015 eq. 2): pi_j(S) = v_j / (v_0 + W(Sbar)
# + V(S)) with shadow attractions w_j in [0, v_j]. w = 0 recovers the BAM/MNL
# (a closed product's demand is fully recaptured by what stays open); w = v
# recovers the IDM (it is entirely lost). v2.csv carries the *ratio* the SBLP
# balance constraint consumes, i.e. vtilde/v = 1 - w/v -- see build_v2.
#
# WHY THIS IS A CONVENTION AND NOT A FIT.  Estimating w needs offer sets that
# vary (paper §2.5): the ticketed-sales feed records purchases only, never
# availability. The one offer-set axis reconstructible here -- which departure
# banks operate on each date, from Networkplanning_raw.xlsx -- is *rejected* by
# the data: dropping it outright fits better than the GAM at its most
# favourable w. assess_shadow_identifiability() reruns that test every build
# and writes the numbers to build_report.json. So v2 is set by an explicit
# assumption, and the knobs below are the assumption.
#
#   "pgam_shape"  as "pgam", but theta carries a per-cell *shape*:
#                 theta_j = kappa_i * d_j, with d built from observables and
#                 normalised to attraction-weighted mean 1 inside each OD, so
#                 kappa_i is still the single per-OD number the recapture
#                 target pins down. Default — see below.
#   "pgam"        parsimonious GAM w_j = theta*v_j, with theta solved *per OD*
#                 so the attraction-weighted mean own-product recapture rate
#                 equals TARGET_RECAPTURE_RATE. The flat-shape special case of
#                 "pgam_shape" (alpha = beta = 0 reproduces it bit-for-bit).
#   "pgam_fixed"  w_j = PGAM_THETA * v_j for every cell, no calibration.
#   "bam"         theta = 0  -> v2 == 1.0 everywhere (full recapture).
#   "idm"         theta = 1  -> v2 == 0.0 everywhere (no recapture).
SHADOW_MODE = "pgam_shape"

# ── the shape, for SHADOW_MODE="pgam_shape" ────────────────────────────────
# The parsimonious GAM's w_j = theta*v_j gives one theta per OD, so every cell
# in a row carries the same ratio. The paper's GAM (eq. 2) does NOT require
# that — it allows a free w_j per product — so a per-cell shape is *more*
# faithful to it, not less. What the shape may not do is invent information:
# it is built only from quantities already measured elsewhere in this build.
#
#   d_j = exp(alpha * tau_j + beta * B_j),  normalised to attraction-weighted
#                                           mean 1 within the OD
#
#   tau_j  time isolation. Circular minutes from that departure bank to the
#          nearest *served* other bank of the same OD, over 720 (the maximum).
#          Single-bank ODs take 1.0. From the real schedule. Isolated banks
#          have no close substitute, so their demand leaks rather than moves —
#          which is exactly what the quoted Ja et al. range conditions on
#          ("in markets in which there are multiple flight departures").
#   B_j    fare barrier. Attraction-weighted mean of max(0, log f_k - log f_j)
#          over the other 15 cells: how much dearer the surviving alternatives
#          are. Closing Eco_lite forces a buy-up and loses demand; closing
#          Eco_flexi lets the passenger buy down at no cost. This is the
#          sell-up channel w exists to carry, and it is read off the observed
#          per-cell fares, not assumed.
#
# alpha = beta = 0 collapses d to 1 and reproduces SHADOW_MODE="pgam" exactly,
# so the shipped-before build is the corner of this one and a sweep over
# (alpha, beta) is a ready-made sensitivity table.
SHADOW_SHAPE_ALPHA = 1.0
SHADOW_SHAPE_BETA = 1.0

# The 26 ODs whose market structure caps recapture below TARGET_RECAPTURE_RATE
# even at theta = 0 would otherwise clamp to the kappa = 0 *boundary*, which is
# a flat v2 == 1 row carrying no cell-level information. Target instead this
# fraction of their own attainable maximum, which is an interior point.
SHADOW_UNREACHABLE_FRACTION = 0.95

# Ja et al. (2001), quoted by the paper (p. 213): "In markets in which there
# are multiple flight departures, recapture rates typically range between
# 15%-55%." 0.35 is the midpoint. This is the single number driving v2 --
# change it and rerun to get a sensitivity case.
TARGET_RECAPTURE_RATE = 0.35

# What "recapture rate" is measured over. The quoted 15-55% is *cross-flight*
# recapture -- a whole departure disappears and the market's other departures
# pick it up -- so the target is calibrated on a whole departure bank
# ("window"). Calibrating instead on a single (fare family, bank) cell would
# read the same 35% as the recapture after closing one RBD group on one
# departure, which is a much closer substitution and would push theta higher.
RECAPTURE_SCOPE = "window"              # "window" | "cell"

PGAM_THETA = 0.20                       # used only by SHADOW_MODE="pgam_fixed"
# Matches V_DECIMALS. At 4 dp the shaped ratios collide on ~250 of the 1440
# cells purely through rounding, which would hide real cell-level structure.
SHADOW_DECIMALS = 6
_THETA_TOL = 1e-12
_THETA_MAX_ITER = 200

# ──────────────────────────────────────────── offer sets / identifiability
# Networkplanning_raw.xlsx is a snapshot taken 2025-01-11: every Eff Date sits
# in 2025-01-11..01-19 and all but a handful of Disc Dates land on or before
# the 2025-03-29 season boundary. Per-date offer sets therefore only exist on
# this sub-window of the departure window, not on all 151 days.
OFFER_START = pd.Timestamp("2025-01-13")
OFFER_END = pd.Timestamp("2025-03-29")
IDENT_IPF_ITER = 300

# ─────────────────────────────────────────────────────────────── censoring
# Replace IATA airport / city codes with 3-digit surrogates in every output.
# The mapping is seeded, so re-running reproduces it exactly; the key lands in
# airport_code_map.csv.
#
# This is obfuscation, not anonymisation — see the README. Departure times,
# fare levels and market sizes are all preserved by design, and any of them
# can be matched against a public schedule to recover a route. Withholding the
# key stops casual reading, nothing stronger.
CENSOR = True
CENSOR_SEED = 20250727
CENSOR_LOW, CENSOR_HIGH = 100, 999      # inclusive 3-digit surrogate range

# ─────────────────────────────────────────────────────────── analysis window
# The maximal *fully-booked* departure window implied by the data:
#   - Ticketed sales carry issue dates over [2024-01-01, 2025-05-31].
#   - The observed maximum advance purchase is 365 days.
# A departure D therefore has its complete booking curve inside the issue
# window iff  D - 365d >= 2024-01-01  and  D <= 2025-05-31.
DEP_START = pd.Timestamp("2025-01-01")
DEP_END = pd.Timestamp("2025-05-31")
MAX_ADVANCE_PURCHASE_DAYS = 365

# Airport -> city codes missing from Airport&CityCodelist.csv. Needed only to
# join sales (airport-keyed) onto MIDT (city-keyed).
CITY_CODE_PATCH = {"ICN": "SEL", "PVG": "SHA", "KNO": "MES"}

# ───────────────────────────────────────────────────── products and windows
# Human-readable name per economy fare family, keyed by the RBD tuple that
# FARE_FAMILY_POLICY assigns to it. Order = most to least flexible, which is
# also strictly descending in observed median fare.
FAMILY_NAMES = {
    ("E", "B", "Y"): "Eco_flexi",
    ("W", "H", "M"): "Eco_standard",
    ("N", "Q"): "Eco_value",
    ("K", "V"): "Eco_lite",
}
PRODUCT_ORDER = ["Eco_flexi", "Eco_standard", "Eco_value", "Eco_lite"]

# Departure-time buckets, in the reference files' column order. Each entry is
# (label, start_minute, end_minute); the wrap-around bucket is handled below.
TIME_WINDOWS = [
    ("(12pm~6pm)", 12 * 60, 18 * 60),
    ("(6pm~10pm)", 18 * 60, 22 * 60),
    ("(10pm~8am)", 22 * 60, 8 * 60),      # wraps midnight
    ("(8am~12pm)", 8 * 60, 12 * 60),
]


def window_of(minutes: int) -> str:
    """Bucket a minutes-past-midnight departure time into a TIME_WINDOWS label."""
    for label, lo, hi in TIME_WINDOWS:
        if lo < hi:
            if lo <= minutes < hi:
                return label
        elif minutes >= lo or minutes < hi:   # wrap-around bucket
            return label
    raise ValueError(f"unbucketable departure minute: {minutes}")


# ═══════════════════════════════════════════ fare-family policy (from source)

def load_economy_fare_families() -> dict[str, str]:
    """Return ``{RBD: product_name}`` for SQ economy, read from the simulator.

    Parses ``FARE_FAMILY_POLICY`` out of ``simulation_sales_generator.py``
    with ``ast`` rather than importing it (the module pulls in Django). Raises
    if the policy has changed shape, so this script fails loudly instead of
    silently drifting from the code base.
    """
    tree = ast.parse(SSG_PY.read_text())
    literal = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "FARE_FAMILY_POLICY" not in names:
            continue
        # FARE_FAMILY_POLICY = pd.DataFrame({...}) -> take the dict argument.
        call = node.value
        if not (isinstance(call, ast.Call) and call.args):
            raise RuntimeError("FARE_FAMILY_POLICY is not a pd.DataFrame({...}) call")
        literal = ast.literal_eval(call.args[0])
        break
    if literal is None:
        raise RuntimeError(f"FARE_FAMILY_POLICY not found in {SSG_PY}")

    cabins = literal["Cabin"]
    families = literal["Fare_Family"]
    airlines = literal["AirlineID"]
    try:
        idx = next(i for i in range(len(cabins))
                   if cabins[i] == "Y" and airlines[i] == "SQ")
    except StopIteration:
        raise RuntimeError("no (SQ, Y) row in FARE_FAMILY_POLICY")

    groups = tuple(tuple(g) for g in families[idx])
    unknown = [g for g in groups if g not in FAMILY_NAMES]
    if unknown:
        raise RuntimeError(
            f"FARE_FAMILY_POLICY (SQ, Y) has unnamed families {unknown}; "
            f"add them to FAMILY_NAMES."
        )
    return {rbd: FAMILY_NAMES[g] for g in groups for rbd in g}


# ═════════════════════════════════════════════════════════════ sales loading

_SALES_COLS = [
    "Iss Date", "Dept Date", "Trip OD Itinerary", "Flight Number (ALL)",
    "POS", "Carrier Designator", "Cabin Class", "Booking Class",
    "Ticketed OD All-In Rev (SGD)", "Ticketed Pax",
]


def _num(series: pd.Series) -> pd.Series:
    """Parse a thousands-separated numeric string column to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def load_sales(report: dict) -> pd.DataFrame:
    """Load the per-POS ticketed-sales CSVs and cut them to the study scope.

    The per-POS files — not the pre-combined ``All.csv`` — are the source:
    ``load_and_clean_sales`` drops ``Flight Number (ALL)`` when it builds
    ``All.csv``, and that column is the only bridge from a ticket to a
    scheduled departure time.
    """
    paths = sorted((RAW / "Raw_Sales").glob("POS *NUS Sales Data*.csv"))
    if not paths:
        raise FileNotFoundError(f"no per-POS sales CSVs under {RAW / 'Raw_Sales'}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=_SALES_COLS, dtype=str, low_memory=False)
        frame["pos_file"] = path.name[4:6]
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    report["sales_files"] = [p.name for p in paths]
    report["rows_raw"] = int(len(df))

    df = df[df["Carrier Designator"].str.strip() == "SQ"]
    report["rows_sq"] = int(len(df))

    # Spec 1 — direct SQ only: the trip itinerary names exactly two cities.
    df = df[df["Trip OD Itinerary"].str.count("-") == 1].copy()
    report["rows_two_city"] = int(len(df))

    # Spec 2 — per-pax price; rows with no ticketed pax carry no price.
    df["rev"] = _num(df["Ticketed OD All-In Rev (SGD)"])
    df["pax"] = _num(df["Ticketed Pax"])
    before = len(df)
    df = df[(df["pax"] > 0) & df["rev"].notna()].copy()
    report["rows_dropped_zero_pax"] = int(before - len(df))
    df["price_per_pax"] = df["rev"] / df["pax"]

    df["iss_date"] = pd.to_datetime(df["Iss Date"], format="%m/%d/%y")
    df["dep_date"] = pd.to_datetime(df["Dept Date"], format="%d %b %Y")
    advance = (df["dep_date"] - df["iss_date"]).dt.days
    report["observed_max_advance_purchase_days"] = int(advance.max())
    if advance.max() > MAX_ADVANCE_PURCHASE_DAYS:
        raise RuntimeError(
            f"advance purchase up to {advance.max()}d exceeds the "
            f"{MAX_ADVANCE_PURCHASE_DAYS}d assumption behind DEP_START; "
            f"widen the window or move DEP_START forward."
        )

    df = df[(df["dep_date"] >= DEP_START) & (df["dep_date"] <= DEP_END)].copy()
    report["rows_in_departure_window"] = int(len(df))
    report["pax_in_departure_window"] = float(df["pax"].sum())

    df["origin"] = df["Trip OD Itinerary"].str.split("-").str[0]
    df["destination"] = df["Trip OD Itinerary"].str.split("-").str[1]
    df["rbd"] = df["Booking Class"].str.strip()
    df["cabin"] = df["Cabin Class"].str.strip()
    df["flight_number"] = pd.to_numeric(
        df["Flight Number (ALL)"].str.strip().str.split().str[-1], errors="coerce",
    )
    return df


def apply_products(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Keep SQ economy and map each RBD to its fare-family product.

    RBDs absent from ``FARE_FAMILY_POLICY`` (economy X and G) are dropped, as
    are the non-economy cabins.
    """
    rbd_to_product = load_economy_fare_families()
    report["fare_families"] = {
        name: sorted(r for r, n in rbd_to_product.items() if n == name)
        for name in PRODUCT_ORDER
    }

    econ = df[df["cabin"] == "Y"].copy()
    report["pax_economy"] = float(econ["pax"].sum())

    econ["product"] = econ["rbd"].map(rbd_to_product)
    dropped = econ[econ["product"].isna()]
    report["pax_dropped_unmapped_rbd"] = float(dropped["pax"].sum())
    report["dropped_rbds"] = sorted(dropped["rbd"].dropna().unique().tolist())
    econ = econ[econ["product"].notna()].copy()
    report["pax_in_policy_economy"] = float(econ["pax"].sum())
    return econ


# ═══════════════════════════════════════════════════════ schedule departure times

# Read once — the workbook is 258k rows and three call sites need it.
_SCHEDULE_CACHE: pd.DataFrame | None = None

# Equipment and seat columns carried alongside the departure time. ``Econ`` is
# the economy-cabin seat count this study needs; ``Prem Econ`` rides along
# because SQ's all-premium A350-900ULR (SIN-LAX) has ``Econ == 0``.
_SEAT_COLS = ["Equip", "Seats", "Econ", "Prem Econ"]


def load_schedule() -> pd.DataFrame:
    """SQ rows of ``Networkplanning_raw.xlsx``, with dates parsed. Cached."""
    global _SCHEDULE_CACHE
    if _SCHEDULE_CACHE is None:
        sched = pd.read_excel(RAW / "Raw_Networkplanning" / "Networkplanning_raw.xlsx")
        sched = sched[sched["Mkt Al"] == "SQ"].copy()
        sched["Eff Date"] = pd.to_datetime(sched["Eff Date"])
        sched["Disc Date"] = pd.to_datetime(sched["Disc Date"])
        _SCHEDULE_CACHE = sched
    return _SCHEDULE_CACHE


def load_departure_times(report: dict) -> pd.DataFrame:
    """``(origin, destination, flight_number) -> departure minute`` from the schedule.

    ``Networkplanning_raw.xlsx`` is a weekly schedule extract; a flight can
    appear on several rows (different effective periods / operating days). The
    representative departure time is the ``Ops/Week``-weighted modal ``Dep
    Time`` among rows whose validity period overlaps the analysis window,
    falling back to all rows for a flight that has no overlapping row.

    Equipment and seat counts ride along on the same representative row, so
    ``flight_capacity.csv`` can never disagree with ``flight.csv`` about which
    schedule row a departure is.
    """
    sched = load_schedule()
    in_window = sched[(sched["Eff Date"] <= DEP_END) & (sched["Disc Date"] >= DEP_START)]
    report["schedule_rows_sq"] = int(len(sched))
    report["schedule_rows_in_window"] = int(len(in_window))

    _CELL = ["Orig", "Dest", "Flight", "Dep Time"]
    _CONFIG = _CELL + ["Equip", "Seats", "Econ", "Prem Econ"]

    def representative(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.assign(_econ_week=frame["Econ"] * frame["Ops/Week"])
        cell = frame.groupby(_CELL).agg(
            **{"Ops/Week": ("Ops/Week", "sum")},
            econ_week=("_econ_week", "sum"),
            equip_variants=("Equip", "nunique"),
            econ_min=("Econ", "min"),
            econ_max=("Econ", "max"),
        ).reset_index()

        # An equipment swap inside one (flight, departure time) cell is resolved
        # the same way the departure time is: by frequency. Type and seat counts
        # must come off the *same* schedule row, or a swapped cell would pair one
        # aircraft's name with another's cabin.
        config = (
            frame.groupby(_CONFIG)["Ops/Week"].sum().reset_index()
            .sort_values(_CELL + ["Ops/Week", "Equip"],
                         ascending=[True] * 4 + [False, True])
            .drop_duplicates(subset=_CELL, keep="first")
            .drop(columns="Ops/Week")
        )
        weights = cell.merge(config, on=_CELL, how="left").sort_values(
            ["Orig", "Dest", "Flight", "Ops/Week", "Dep Time"],
            ascending=[True, True, True, False, True],
        )
        return weights.drop_duplicates(subset=["Orig", "Dest", "Flight"], keep="first")

    primary = representative(in_window)
    fallback = representative(sched)
    merged = pd.concat([
        primary,
        fallback.merge(
            primary[["Orig", "Dest", "Flight"]], on=["Orig", "Dest", "Flight"],
            how="left", indicator=True,
        ).query("_merge == 'left_only'").drop(columns="_merge"),
    ], ignore_index=True)

    keep = (["Orig", "Dest", "Flight", "Dep Time", "Ops/Week"] + _SEAT_COLS
            + ["econ_week", "equip_variants", "econ_min", "econ_max"])
    out = merged[keep].rename(columns={
        "Orig": "origin", "Dest": "destination", "Flight": "flight_number",
    })
    # Dep Time is HHMM as an integer.
    out["dep_minute"] = (out["Dep Time"] // 100) * 60 + out["Dep Time"] % 100
    out["departure_time"] = out["dep_minute"].map(lambda m: f"{m // 60:02d}:{m % 60:02d}")
    return out.drop(columns="Dep Time")


def attach_departure_times(sales: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Join ticket rows onto the schedule; drop flights the schedule does not cover."""
    times = load_departure_times(report)
    merged = sales.merge(
        times, on=["origin", "destination", "flight_number"], how="left",
    )
    matched = merged[merged["dep_minute"].notna()].copy()
    unmatched = merged[merged["dep_minute"].isna()]

    report["pax_before_schedule_join"] = float(merged["pax"].sum())
    report["pax_after_schedule_join"] = float(matched["pax"].sum())
    report["schedule_join_pax_rate"] = float(
        matched["pax"].sum() / merged["pax"].sum()
    )
    # Flight numbers are withheld under CENSOR — a public timetable turns one
    # into a route, which would undo the surrogate codes.
    lost = (
        unmatched.groupby(["origin", "destination"])["pax"].sum()
        .sort_values(ascending=False)
        .rename("pax").reset_index()
    )
    report["pax_lost_to_schedule_join_by_od"] = lost.to_dict("records")

    matched["dep_minute"] = matched["dep_minute"].astype(int)
    matched["time_window"] = matched["dep_minute"].map(window_of)
    return matched


# ═════════════════════════════════════════════════════ non-SQ share from MIDT

def load_sq_market_share(report: dict) -> pd.DataFrame:
    """SQ's economy booking share per undirected city pair, from MIDT.

    ``AUG_Share_Data`` reports one canonical direction per market, so the
    share is computed on the undirected city pair and applied to both
    directions. Sales are airport-keyed (LHR / LGW are distinct ODs) while
    MIDT is city-keyed (both are LON), so several ODs can share one market.
    """
    midt = pd.read_csv(
        RAW / "AUG_Share_Data" / "All.csv",
        usecols=[
            "O-D Code (True-Dir) L2 - City", "(True) Mkt Carrier - (Dominant)",
            "MIDT Bkgs CY - Carrier (True)", "Cabin (True) (Mkt Carr-Dom)",
            "Dptr Date (True) (YYYYMMDD)",
        ],
    )
    midt = midt[midt["Cabin (True) (Mkt Carr-Dom)"] == "Economy"].copy()
    midt["pair"] = [
        "-".join(sorted(str(od).split("-")))
        for od in midt["O-D Code (True-Dir) L2 - City"]
    ]

    def share_of(frame: pd.DataFrame, label: str) -> pd.Series:
        total = frame.groupby("pair")["MIDT Bkgs CY - Carrier (True)"].sum()
        sq = (
            frame[frame["(True) Mkt Carrier - (Dominant)"] == "SQ"]
            .groupby("pair")["MIDT Bkgs CY - Carrier (True)"].sum()
        )
        out = (sq / total).dropna()
        out.name = label
        return out

    window = midt[
        (midt["Dptr Date (True) (YYYYMMDD)"] >= int(DEP_START.strftime("%Y%m%d")))
        & (midt["Dptr Date (True) (YYYYMMDD)"] <= int(DEP_END.strftime("%Y%m%d")))
    ]
    primary = share_of(window, "sq_share")
    sensitivity = share_of(midt, "sq_share_all_departures")

    shares = pd.concat([primary, sensitivity], axis=1).reset_index()
    report["midt_pairs"] = int(len(shares))
    return shares


def city_map() -> dict[str, str]:
    """Airport -> city code, with the three codes the code list is missing."""
    codes = pd.read_csv(RAW / "Airport&CityCodelist.csv")
    mapping = dict(zip(codes["Airport"], codes["City code"]))
    mapping.update(CITY_CODE_PATCH)
    return mapping


# ═══════════════════════════════════════════════════════════════ output build

def build_flight(sales: pd.DataFrame, report: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """flight.csv — pax-weighted mean per-pax fare per (OD, departure, product).

    Prices are computed independently for every OD: no fare is ever pooled
    across ODs. A (departure, product) cell with no observed ticket falls back
    to that OD's own product-level mean, so every OD emits a complete
    departure x product grid; the returned coverage frame records which cells
    were filled that way.
    """
    cell = (
        sales.groupby(["origin", "destination", "departure_time", "product"])
        .agg(rev=("rev", "sum"), pax=("pax", "sum")).reset_index()
    )
    cell["price"] = cell["rev"] / cell["pax"]

    od_product = (
        sales.groupby(["origin", "destination", "product"])
        .agg(rev=("rev", "sum"), pax=("pax", "sum")).reset_index()
    )
    od_product["od_product_price"] = od_product["rev"] / od_product["pax"]

    # Full grid: every departure time an OD flies x every product it sells.
    departures = sales[["origin", "destination", "departure_time"]].drop_duplicates()
    grid = departures.merge(
        od_product[["origin", "destination", "product", "od_product_price"]],
        on=["origin", "destination"], how="inner",
    )
    grid = grid.merge(
        cell[["origin", "destination", "departure_time", "product", "price", "pax"]],
        on=["origin", "destination", "departure_time", "product"], how="left",
    )
    grid["observed"] = grid["price"].notna()
    grid["avg_price"] = grid["price"].fillna(grid["od_product_price"]).round(2)
    grid["pax"] = grid["pax"].fillna(0.0)

    report["flight_rows"] = int(len(grid))
    report["flight_rows_observed"] = int(grid["observed"].sum())
    report["flight_rows_filled_from_od_mean"] = int((~grid["observed"]).sum())

    grid["product_rank"] = grid["product"].map(PRODUCT_ORDER.index)
    grid = grid.sort_values(
        ["origin", "destination", "departure_time", "product_rank"]
    ).reset_index(drop=True)

    coverage = grid[[
        "origin", "destination", "departure_time", "product",
        "observed", "pax", "avg_price", "od_product_price",
    ]].copy()
    flight = grid[[
        "origin", "destination", "departure_time", "product", "avg_price",
    ]].copy()
    return flight, coverage


def build_capacity(sales: pd.DataFrame, report: dict) -> pd.DataFrame:
    """flight_capacity.csv — aircraft type and economy seats per departure.

    Answers the question ``flight.csv`` left open: every one of its
    ``(Oneway_OD, Departure Time)`` cells gets the ``Equip`` and per-cabin seat
    counts of the schedule row it was built from. No haul-band approximation is
    needed — ``Networkplanning_raw.xlsx`` carries ``Equip``, ``Seats``,
    ``First``, ``Business``, ``Prem Econ`` and ``Econ`` natively.

    Two capacity units are emitted because the deliverables mix them:
    ``Y Seats`` is per departure (what a per-flight capacity constraint wants)
    while ``Y Seats/Week`` multiplies by ``Ops/Week`` (the unit ``od_demand``'s
    ``Avg Pax`` is already in).
    """
    cells = (
        sales[["origin", "destination", "departure_time", "flight_number",
               "Equip", "Seats", "Econ", "Prem Econ", "Ops/Week", "econ_week",
               "equip_variants", "econ_min", "econ_max"]]
        .drop_duplicates(subset=["origin", "destination", "departure_time"])
        .sort_values(["origin", "destination", "departure_time"])
        .reset_index(drop=True)
    )

    out = pd.DataFrame({
        "origin": cells["origin"],
        "destination": cells["destination"],
        "departure_time": cells["departure_time"],
        "Aircraft Type": cells["Equip"],
        "Total Seats": cells["Seats"].astype(int),
        "Y Seats": cells["Econ"].astype(int),
        "W Seats": cells["Prem Econ"].astype(int),
        "Ops/Week": cells["Ops/Week"].astype(int),
    })
    # Summed over the schedule rows rather than Y Seats x Ops/Week, so a cell
    # that swaps equipment mid-season gets its true weekly seat count.
    out["Y Seats/Week"] = cells["econ_week"].astype(int)
    # SQ 35/36 (SIN-LAX) is the all-premium A350-900ULR: 67J + 94W, no economy
    # cabin at all. The seat count is right and the sales feed's Y-cabin pax on
    # those two departures is not; flag rather than silently patch.
    out["all_premium_no_Y"] = out["Y Seats"] == 0
    # An equipment swap inside the cell means Y Seats is the modal type's.
    out["equip_varies"] = cells["equip_variants"].to_numpy() > 1
    out["Y Seats min"] = cells["econ_min"].astype(int)
    out["Y Seats max"] = cells["econ_max"].astype(int)

    report["capacity_cells"] = int(len(out))
    report["capacity_cells_equip_varies"] = int(out["equip_varies"].sum())
    report["capacity_cells_all_premium_no_Y"] = int(out["all_premium_no_Y"].sum())
    report["capacity_seats_by_equip"] = {
        str(k): {"departures": int(len(g)), "median_Y": float(g["Y Seats"].median()),
                 "min_Y": int(g["Y Seats"].min()), "max_Y": int(g["Y Seats"].max())}
        for k, g in out.groupby("Aircraft Type")
    }
    report["capacity_Y_seats"] = {
        "min": int(out["Y Seats"].min()), "median": float(out["Y Seats"].median()),
        "max": int(out["Y Seats"].max()),
    }
    return out


# ════════════════════════════════════════════════════════════════ offer sets

def build_offer_sets(sales: pd.DataFrame, report: dict) -> pd.DataFrame:
    """offer_sets.csv — which departure banks each OD actually flies, per date.

    This is the ``S`` of the GAM: the set of alternatives a customer shopping
    that ``(OD, departure date)`` can see. It is reconstructed by expanding the
    schedule's ``Op Days`` / ``Eff Date`` / ``Disc Date`` over the calendar, so
    it is availability at the *departure-bank* level only — the sales feed
    never records fare-family (RBD) availability, which is the axis the GAM's
    upsell story actually turns on.

    Restricted to ``[OFFER_START, OFFER_END]``: outside that range the schedule
    extract has no coverage, so an empty offer set would be an artefact.
    """
    sched = load_schedule()
    sched = sched.assign(
        dep_minute=(sched["Dep Time"] // 100) * 60 + sched["Dep Time"] % 100,
        op_days=sched["Op Days"].astype(str),
    )
    sched["window"] = sched["dep_minute"].map(window_of)
    in_scope = set(zip(sales["origin"], sales["destination"]))

    rows = []
    for day in pd.date_range(OFFER_START, OFFER_END):
        code = str(day.dayofweek + 1)            # Op Days is '1234567', Mon=1
        active = sched[
            (sched["Eff Date"] <= day) & (sched["Disc Date"] >= day)
            & sched["op_days"].str.contains(code, regex=False)
        ]
        for (origin, destination), group in active.groupby(["Orig", "Dest"]):
            if (origin, destination) not in in_scope:
                continue
            served = sorted({w for w in group["window"]},
                            key=[l for l, _, _ in TIME_WINDOWS].index)
            rows.append({
                "origin": origin, "destination": destination,
                "departure_date": day.strftime("%Y-%m-%d"),
                "day_of_week": day.dayofweek + 1,
                "offer_set": "|".join(served),
                "n_windows": len(served),
                "n_departures": int(len(group)),
                "Y_seats_offered": int(group["Econ"].sum()),
            })
    offer = pd.DataFrame(rows)

    per_od = offer.groupby(["origin", "destination"])["offer_set"].nunique()
    report["offer_set_window"] = [OFFER_START.strftime("%Y-%m-%d"),
                                  OFFER_END.strftime("%Y-%m-%d")]
    report["offer_set_od_dates"] = int(len(offer))
    report["offer_set_ods"] = int(len(per_od))
    report["offer_set_ods_with_variation"] = int((per_od > 1).sum())
    report["offer_set_distinct_sets_per_od"] = {
        int(k): int(v) for k, v in per_od.value_counts().sort_index().items()
    }
    return offer


def assess_shadow_identifiability(
    sales: pd.DataFrame, v1: pd.DataFrame, offer: pd.DataFrame, report: dict,
) -> dict:
    """Test whether the offer sets identify ``w``. They do not — this records why.

    Fits, by Poisson IPF, daily SQ pax as
    ``mu[i,d] = lambda_i * g[dow] * h[week] * V(S) / D(S)`` with
    ``D(S) = v_0 + V(S) + theta * V(Sbar)``, at the GAM's two extremes and its
    midpoint, and against a null that drops the offer-set term ``V(S)/D(S)``
    altogether.

    If the null fits *better* than the GAM at its most favourable ``theta``,
    the offer-set channel carries no usable signal and any ``w`` read off it is
    an artefact of the boundary, not an estimate. That is what happens here.
    """
    windows = [label for label, _, _ in TIME_WINDOWS]
    keyed = v1.set_index(["origin", "destination"])
    attraction = {
        od: np.array([sum(keyed.loc[od, f"{p}*{w}"] for p in PRODUCT_ORDER)
                      for w in windows])
        for od in keyed.index
    }
    no_purchase = keyed["no_purchase"].to_dict()

    frame = offer[[(o, d) in attraction
                   for o, d in zip(offer["origin"], offer["destination"])]].copy()
    if frame.empty:
        return {"verdict": "no offer-set observations"}

    daily = (
        sales.assign(date=sales["dep_date"].dt.strftime("%Y-%m-%d"))
        .groupby(["origin", "destination", "date"])["pax"].sum()
    )
    frame["pax"] = [
        daily.get((o, d, day), 0.0) for o, d, day in
        zip(frame["origin"], frame["destination"], frame["departure_date"])
    ]

    ods = sorted({(o, d) for o, d in zip(frame["origin"], frame["destination"])})
    od_index = {od: i for i, od in enumerate(ods)}
    idx = np.array([od_index[(o, d)] for o, d in
                    zip(frame["origin"], frame["destination"])])
    dow = frame["day_of_week"].to_numpy() - 1
    week = ((pd.to_datetime(frame["departure_date"]) - OFFER_START).dt.days // 7).to_numpy()
    z = frame["pax"].to_numpy(float)

    mask = np.zeros((len(frame), len(windows)), bool)
    for row, served in enumerate(frame["offer_set"]):
        for label in served.split("|"):
            mask[row, windows.index(label)] = True
    v_od = np.array([attraction[od] for od in ods])
    v0_od = np.array([no_purchase[od] for od in ods])

    def deviance(theta: float, use_offer_set: bool) -> tuple[float, float]:
        served = (mask * v_od[idx]).sum(axis=1)
        closed = ((~mask) * v_od[idx]).sum(axis=1)
        base = served / (v0_od[idx] + served + theta * closed) if use_offer_set \
            else np.ones(len(idx))
        lam = np.ones(len(ods))
        g = np.ones(7)
        h = np.ones(int(week.max()) + 1)
        for _ in range(IDENT_IPF_ITER):
            pred = g[dow] * h[week] * base
            lam = (np.bincount(idx, z, len(ods))
                   / np.maximum(np.bincount(idx, pred, len(ods)), 1e-300))
            pred = lam[idx] * h[week] * base
            g = np.bincount(dow, z, 7) / np.maximum(np.bincount(dow, pred, 7), 1e-300)
            g = g / g.mean()
            pred = lam[idx] * g[dow] * base
            h = (np.bincount(week, z, len(h))
                 / np.maximum(np.bincount(week, pred, len(h)), 1e-300))
            h = h / h.mean()
        mu = np.maximum(lam[idx] * g[dow] * h[week] * base, 1e-9)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = 2 * np.sum(
                np.where(z > 0, z * np.log(np.maximum(z, 1e-12) / mu), 0.0) - (z - mu)
            )
        pearson = float(np.sum((z - mu) ** 2 / mu))
        return float(dev), pearson

    residual_dof = max(len(idx) - len(ods) - 6 - (int(week.max())), 1)
    fits = {
        "gam_theta_0_bam_full_recapture": deviance(0.0, True),
        "gam_theta_0.5": deviance(0.5, True),
        "gam_theta_1_idm_no_recapture": deviance(1.0, True),
        "null_offer_set_dropped": deviance(0.0, False),
    }
    out = {
        "observations": int(len(idx)),
        "ods": len(ods),
        "deviance": {k: round(v[0], 1) for k, v in fits.items()},
        "pearson_dispersion": {k: round(v[1] / residual_dof, 2) for k, v in fits.items()},
    }
    best_gam = min(fits[k][0] for k in fits if k != "null_offer_set_dropped")
    gain = fits["null_offer_set_dropped"][0] - best_gam
    out["null_minus_best_gam_deviance"] = round(gain, 1)
    out["offer_set_informative"] = bool(gain > 0)
    out["verdict"] = (
        "offer sets carry usable signal — w is estimable" if gain > 0 else
        "dropping the offer set fits BETTER than the GAM at its most favourable "
        "w, so the departure-bank offer sets carry no usable information about "
        "w; v2 is set by assumption (see SHADOW_MODE), not estimated"
    )
    report["shadow_identifiability"] = out
    return out


# ════════════════════════════════════════════════════ shadow attraction (v2)

def _group_attractions(v: np.ndarray, scope: str) -> np.ndarray:
    """Attraction of each removable unit ``R``, under the chosen scope.

    ``v`` is one OD's 16 cells laid out product-major, matching the v1 column
    order ``[f"{p}*{w}" for p in PRODUCT_ORDER for w in windows]``.
    """
    if scope == "cell":
        return v
    if scope == "window":
        return v.reshape(len(PRODUCT_ORDER), len(TIME_WINDOWS)).sum(axis=0)
    raise RuntimeError(f"RECAPTURE_SCOPE={scope!r} is not 'window' or 'cell'")


def _mean_recapture(
    kappa: float, v: np.ndarray, v0: float, scope: str,
    shape: np.ndarray | None = None,
) -> float:
    """Attraction-weighted mean recapture rate at ``w_j = kappa*shape_j*v_j``.

    Removing a set ``R`` from the full set moves ``V_R/(v_0+V)`` of demand. The
    products that stay open pick up ``(V - V_R)(V_R - W_R)/(D_S * D_N)`` with
    ``D_S = v_0 + W_R + V - V_R``, so writing ``tbar_R = W_R/V_R`` for R's
    attraction-weighted mean ratio, the share of R's demand recaptured rather
    than lost to the no-purchase alternative is

        r_R = (1 - tbar_R) * (V - V_R) / (v_0 + tbar_R*V_R + V - V_R)

    which is 0 at ``tbar = 1`` (IDM) and maximal at ``tbar = 0`` (BAM). The
    scope sets what ``R`` is: a whole departure bank, or one fare family within
    one bank.

    ``shape is None`` (or all-ones) is the parsimonious case ``w = kappa*v``:
    ``tbar_R = kappa`` for every ``R`` and this reduces to the scalar form.
    """
    total = v.sum()
    if total <= 0:
        return 0.0
    theta = kappa if shape is None else np.clip(kappa * shape, 0.0, 1.0)
    group = _group_attractions(v, scope)
    shadow = _group_attractions(theta * v, scope)
    tbar = shadow / np.maximum(group, 1e-300)
    others = total - group
    rate = (1.0 - tbar) * others / (v0 + tbar * group + others)
    return float((group * rate).sum() / group.sum())


def solve_theta(
    v: np.ndarray, v0: float, target: float, scope: str,
    shape: np.ndarray | None = None, unreachable_fraction: float = 1.0,
) -> tuple[float, float, bool]:
    """Smallest ``kappa`` whose mean recapture rate is ``target``.

    ``_mean_recapture`` is continuous and strictly decreasing in ``kappa`` on
    the admissible range, so a bisection is exact. Markets where SQ is small
    relative to the no-purchase alternative cannot reach the target even at
    ``kappa = 0``; those are reported as unreachable and aim instead at
    ``unreachable_fraction`` of their own attainable maximum, so the solution
    stays interior and keeps the cell-level shape (at 1.0 they clamp to the
    ``kappa = 0`` boundary, which is the old behaviour and a flat row).
    """
    best = _mean_recapture(0.0, v, v0, scope, shape)
    reached = True
    if best <= target:
        if unreachable_fraction >= 1.0:
            return 0.0, best, False
        target = unreachable_fraction * best
        reached = False
    # kappa scales theta, which is clipped to [0, 1] cell-wise inside
    # _mean_recapture. The bracket must reach the point where *every* cell has
    # saturated (theta = 1, recapture 0), or the bisection can run out of range
    # before it reaches the target and silently return a kappa that misses it —
    # bounding by 1/max(shape) instead of 1/min(shape) left 4 of the 64
    # reachable ODs short of TARGET_RECAPTURE_RATE.
    high = 1.0 if shape is None else 1.0 / max(float(np.min(shape)), 1e-12)
    low = 0.0
    for _ in range(_THETA_MAX_ITER):
        mid = 0.5 * (low + high)
        if _mean_recapture(mid, v, v0, scope, shape) > target:
            low = mid
        else:
            high = mid
        if high - low < _THETA_TOL:
            break
    kappa = 0.5 * (low + high)
    return kappa, _mean_recapture(kappa, v, v0, scope, shape), reached


def _shadow_shape(
    v1: pd.DataFrame, coverage: pd.DataFrame, report: dict,
) -> np.ndarray:
    """Per-cell shadow multipliers ``d``, attraction-weighted mean 1 per OD.

    Built from two observables already carried by this build — see the
    SHADOW_SHAPE_* block at the top of the file for what they mean and why they
    are the right two.
    """
    windows = [label for label, _, _ in TIME_WINDOWS]
    columns = [f"{p}*{w}" for p in PRODUCT_ORDER for w in windows]
    values = v1[columns].to_numpy(float).reshape(len(v1), len(PRODUCT_ORDER), len(windows))
    order = list(zip(v1["origin"], v1["destination"]))
    position = {od: i for i, od in enumerate(order)}
    n = len(order)

    # ── tau: time isolation of each bank, from the real schedule ──────────
    dep_minutes = (
        coverage["departure_time"].str.slice(0, 2).astype(int) * 60
        + coverage["departure_time"].str.slice(3, 5).astype(int)
    )
    cov = coverage.assign(dep_minutes=dep_minutes)
    cov["time_window"] = cov["dep_minutes"].map(window_of)

    served = np.zeros((n, len(windows)), bool)
    rep = np.tile(
        np.array([_window_midpoint(w) for w in windows], float), (n, 1),
    )
    for (origin, destination, window), group in cov.groupby(
        ["origin", "destination", "time_window"]
    ):
        i = position.get((origin, destination))
        if i is None:
            continue
        t = windows.index(window)
        weight = np.maximum(group["pax"].to_numpy(float), 1e-9)
        rep[i, t] = float(np.average(group["dep_minutes"], weights=weight))
        # coverage only carries departures the OD actually flies, so presence
        # is the same served-window definition build_v1's mask uses.
        served[i, t] = True

    tau = np.zeros((n, len(windows)))
    for i in range(n):
        for t in range(len(windows)):
            others = [u for u in range(len(windows)) if u != t and served[i, u]]
            if not others:
                tau[i, t] = 1.0          # single-bank OD: maximally isolated
            else:
                gap = min(_circular_minutes(rep[i, t], rep[i, u]) for u in others)
                tau[i, t] = gap / 720.0

    # ── B: fare barrier, from the observed per-cell fares ─────────────────
    fare = np.full((n, len(PRODUCT_ORDER), len(windows)), np.nan)
    for (origin, destination, product, window), group in cov.groupby(
        ["origin", "destination", "product", "time_window"]
    ):
        i = position.get((origin, destination))
        if i is None:
            continue
        weight = np.maximum(group["pax"].to_numpy(float), 1e-9)
        fare[i, PRODUCT_ORDER.index(product), windows.index(window)] = float(
            np.average(group["avg_price"], weights=weight)
        )
    # Unserved banks have no fare of their own; the OD x product mean is the
    # stand-in flight.csv already documents for its imputed cells.
    for (origin, destination, product), group in cov.groupby(
        ["origin", "destination", "product"]
    ):
        i = position.get((origin, destination))
        if i is None:
            continue
        row = fare[i, PRODUCT_ORDER.index(product)]
        row[np.isnan(row)] = float(group["od_product_price"].iloc[0])
    if np.isnan(fare).any():
        raise RuntimeError("shadow shape: fare grid has holes after OD-mean fill")

    barrier = np.zeros((n, len(PRODUCT_ORDER) * len(windows)))
    log_fare = np.log(fare)
    for i in range(n):
        v_flat = values[i].ravel()
        lf = log_fare[i].ravel()
        for j in range(len(v_flat)):
            up = np.maximum(0.0, lf - lf[j])
            keep = np.ones(len(v_flat), bool)
            keep[j] = False
            barrier[i, j] = float(np.average(up[keep], weights=v_flat[keep]))

    tau_flat = np.repeat(tau[:, None, :], len(PRODUCT_ORDER), axis=1).reshape(n, -1)
    raw = np.exp(SHADOW_SHAPE_ALPHA * tau_flat + SHADOW_SHAPE_BETA * barrier)
    flat_values = values.reshape(n, -1)
    shape = raw / np.average(raw, axis=1, weights=flat_values)[:, None]

    report["shadow_shape"] = {
        "alpha": SHADOW_SHAPE_ALPHA,
        "beta": SHADOW_SHAPE_BETA,
        "tau_mean": round(float(tau.mean()), 4),
        "barrier_mean": round(float(barrier.mean()), 4),
        "d_min": round(float(shape.min()), 4),
        "d_median": round(float(np.median(shape)), 4),
        "d_max": round(float(shape.max()), 4),
    }
    return shape


def _window_midpoint(label: str) -> float:
    for name, low, high in TIME_WINDOWS:
        if name != label:
            continue
        return ((low + high) / 2.0) % 1440 if low < high else ((low + high + 1440) / 2.0) % 1440
    raise RuntimeError(f"unknown window {label!r}")


def _circular_minutes(a: float, b: float) -> float:
    gap = abs(a - b)
    return min(gap, 1440.0 - gap)


def build_v2(
    v1: pd.DataFrame, coverage: pd.DataFrame, report: dict,
) -> pd.DataFrame:
    """v2.csv — the shadow-attraction ratios the SBLP balance constraint uses.

    The paper's transformation (p. 215) is ``vtilde_j = v_j - w_j`` for
    ``j in N`` and ``vtilde_0 = v_0 + W(N)``, and the SBLP balance constraint
    (eq. 7) reads ``sum_k (vtilde_k/v_k) x_k + (vtilde_0/v_0) x_0 = Lambda``.
    So the coefficients this file must carry are

        cell         vtilde_j / v_j = 1 - w_j/v_j    in [0, 1]
        no_purchase  vtilde_0 / v_0 = 1 + sum_j w_j / v_0

    which is exactly the convention the reference ``Air_NRM/v1.csv`` +
    ``v2.csv`` pair uses (verified: reconstructing ``no_purchase`` from the
    reference v1/v2 as ``1 + sum_j v_j (1 - v2_j) / v_0`` reproduces its stated
    value on 3 of its 4 rows; the 4th row is internally inconsistent).

    Both columns are invariant to rescaling a v1 row, so it does not matter
    whether v1 is on the share-like scale used here or the reference's
    ``v_0 = 1``.
    """
    windows = [label for label, _, _ in TIME_WINDOWS]
    columns = [f"{p}*{w}" for p in PRODUCT_ORDER for w in windows]
    values = v1[columns].to_numpy(float)
    v0 = v1["no_purchase"].to_numpy(float)

    shape = None
    unreachable_fraction = 1.0
    if SHADOW_MODE == "pgam_shape":
        shape = _shadow_shape(v1, coverage, report)
        unreachable_fraction = SHADOW_UNREACHABLE_FRACTION

    if SHADOW_MODE == "bam":
        kappa = np.zeros(len(v1))
    elif SHADOW_MODE == "idm":
        kappa = np.ones(len(v1))
    elif SHADOW_MODE == "pgam_fixed":
        kappa = np.full(len(v1), float(PGAM_THETA))
    elif SHADOW_MODE in ("pgam", "pgam_shape"):
        kappa = np.empty(len(v1))
        achieved = np.empty(len(v1))
        reached = np.empty(len(v1), bool)
        for i in range(len(v1)):
            kappa[i], achieved[i], reached[i] = solve_theta(
                values[i], v0[i], TARGET_RECAPTURE_RATE, RECAPTURE_SCOPE,
                None if shape is None else shape[i], unreachable_fraction,
            )
        report["shadow_target_recapture_rate"] = TARGET_RECAPTURE_RATE
        report["shadow_recapture_scope"] = RECAPTURE_SCOPE
        report["shadow_ods_target_unreachable"] = int((~reached).sum())
        report["shadow_recapture_achieved"] = {
            "min": round(float(achieved.min()), 4),
            "median": round(float(np.median(achieved)), 4),
            "max": round(float(achieved.max()), 4),
        }
        # The same kappa implies a different (higher) recapture rate at the
        # finer scope; report it so the number is not read at the wrong level.
        other = "cell" if RECAPTURE_SCOPE == "window" else "window"
        implied = np.array([
            _mean_recapture(kappa[i], values[i], v0[i], other,
                            None if shape is None else shape[i])
            for i in range(len(v1))
        ])
        report[f"shadow_recapture_implied_at_{other}_scope"] = {
            "min": round(float(implied.min()), 4),
            "median": round(float(np.median(implied)), 4),
            "max": round(float(implied.max()), 4),
        }
    else:
        raise RuntimeError(
            f"SHADOW_MODE={SHADOW_MODE!r} is not one of "
            f"'pgam_shape', 'pgam', 'pgam_fixed', 'bam', 'idm'"
        )

    report["shadow_mode"] = SHADOW_MODE
    report["shadow_theta"] = {
        "min": round(float(kappa.min()), 4),
        "median": round(float(np.median(kappa)), 4),
        "max": round(float(kappa.max()), 4),
    }

    if shape is None:
        theta_cells = kappa[:, None] * np.ones_like(values)
    else:
        theta_cells = np.clip(kappa[:, None] * shape, 0.0, 1.0)
    report["shadow_theta_cells"] = {
        "min": round(float(theta_cells.min()), 4),
        "median": round(float(np.median(theta_cells)), 4),
        "max": round(float(theta_cells.max()), 4),
        "clipped_at_1": int((theta_cells >= 1.0).sum()),
    }
    ratio = np.round(1.0 - theta_cells, SHADOW_DECIMALS)
    # W(N) is recomputed from the *rounded* ratios, so a consumer who recovers
    # w_j = v_j (1 - v2_j) from the two published files reproduces no_purchase
    # exactly rather than to within the rounding of the cells.
    shadow_total = ((1.0 - ratio) * values).sum(axis=1)
    no_purchase = np.round(1.0 + shadow_total / v0, SHADOW_DECIMALS)

    if (ratio < 0).any() or (ratio > 1).any():
        raise RuntimeError("v2 cell outside [0, 1]; w must satisfy 0 <= w <= v")
    if (no_purchase < 1).any():
        raise RuntimeError("v2 no_purchase below 1; vtilde_0 = v_0 + W(N) >= v_0")

    report["shadow_no_purchase_ratio"] = {
        "min": round(float(no_purchase.min()), 4),
        "median": round(float(np.median(no_purchase)), 4),
        "max": round(float(no_purchase.max()), 4),
    }

    frame = pd.concat([
        v1[["origin", "destination"]].reset_index(drop=True),
        pd.DataFrame(ratio, columns=columns),
    ], axis=1)
    frame["no_purchase"] = no_purchase
    return frame


def build_market(sales: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Per-OD market size: SQ economy pax grossed up by the MIDT SQ share.

    Non-SQ traffic is the whole of the no-purchase alternative, so the market
    an OD faces is ``sq_pax / sq_share`` and the non-SQ volume is the
    remainder.
    """
    airport_to_city = city_map()
    sq = (
        sales.groupby(["origin", "destination"])["pax"].sum()
        .rename("sq_pax").reset_index()
    )
    sq["origin_city"] = sq["origin"].map(airport_to_city)
    sq["destination_city"] = sq["destination"].map(airport_to_city)
    missing = sq[sq["origin_city"].isna() | sq["destination_city"].isna()]
    if not missing.empty:
        raise RuntimeError(
            "airports with no city code: "
            f"{sorted(set(missing['origin']) | set(missing['destination']))}"
        )
    sq["pair"] = [
        "-".join(sorted([o, d]))
        for o, d in zip(sq["origin_city"], sq["destination_city"])
    ]

    shares = load_sq_market_share(report)
    market = sq.merge(shares, on="pair", how="left")
    unmatched = market[market["sq_share"].isna()]
    report["ods_without_midt_share"] = int(len(unmatched))
    report["pax_without_midt_share"] = float(unmatched["sq_pax"].sum())
    market = market[market["sq_share"].notna() & (market["sq_share"] > 0)].copy()

    market["market_pax"] = market["sq_pax"] / market["sq_share"]
    market["non_sq_pax"] = market["market_pax"] - market["sq_pax"]

    n_days = (DEP_END - DEP_START).days + 1
    weeks = n_days / 7.0
    report["departure_window_days"] = int(n_days)
    report["departure_window_weeks"] = round(weeks, 4)
    market["avg_pax_per_week"] = market["market_pax"] / weeks
    market["sq_pax_per_week"] = market["sq_pax"] / weeks

    report["ods_in_output"] = int(len(market))
    return market.sort_values("market_pax", ascending=False).reset_index(drop=True)


def fit_window_effect(
    pax: np.ndarray, mask: np.ndarray, report: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Quasi-independence fit ``E[pax[i,p,w]] = theta[i,p] * delta[w]``.

    The zero cells are not missing at random — an OD's bookings concentrate in
    whatever bank it happens to fly, so raw pooled window shares measure SQ's
    schedule rather than passengers' time-of-day preference. Fitting on the
    *observed support only* (an incomplete contingency table) separates the
    two: ``theta`` absorbs each OD's own product mix and size, ``delta`` is the
    availability-corrected window preference identified from the 64 ODs that
    serve two or more banks.

    ``delta`` is deliberately shared across products. A per-product
    ``delta[p, w]`` is estimable but earns nothing — hold-out error was 0.6398
    vs 0.6370 and the spread across products never exceeded 0.035 — so the
    tied form is the more robust of the two.

    Returns ``(theta, delta)`` with ``delta`` normalised to sum to 1.
    """
    pax_ip = pax.sum(axis=2)
    theta = np.ones(pax.shape[:2])
    delta = np.ones(pax.shape[2]) / pax.shape[2]
    numerator = (pax * mask[:, None, :]).sum(axis=(0, 1))

    for iteration in range(IMPUTE_MAX_ITER):
        previous = delta
        theta = pax_ip / np.maximum(
            (mask * delta[None, :]).sum(axis=1)[:, None], 1e-300,
        )
        denominator = (mask[:, None, :] * theta[:, :, None]).sum(axis=(0, 1))
        delta = numerator / np.maximum(denominator, 1e-300)
        delta = delta / delta.sum()
        if np.abs(delta - previous).max() < IMPUTE_TOL:
            break
    else:
        raise RuntimeError(
            f"window-effect fit did not converge in {IMPUTE_MAX_ITER} iterations"
        )

    report["impute_iterations"] = iteration + 1
    report["window_effect"] = {
        label: round(float(value), 6)
        for (label, _, _), value in zip(TIME_WINDOWS, delta)
    }
    raw = numerator / numerator.sum()
    report["window_share_raw_supply_confounded"] = {
        label: round(float(value), 6)
        for (label, _, _), value in zip(TIME_WINDOWS, raw)
    }
    return theta, delta


def build_v1(
    sales: pd.DataFrame, market: pd.DataFrame, report: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """v1.csv — GAM attraction values, expressed against the OD's total market.

    ``v[product x window]`` is that cell's pax over the OD's total economy
    market; ``no_purchase`` is ``v_0``, the non-SQ share. GAM choice
    probabilities are invariant to a common per-market scale, so this is the
    same model as any other normalisation — it just reports the fitted numbers
    on a share-like scale. Dividing a row through by its own ``no_purchase``
    recovers the reference files' ``v_0 = 1`` convention; ``v2.csv`` is a ratio
    and is unaffected either way.

    With ``IMPUTE_ZERO_CELLS`` the unserved-window cells are filled from
    ``fit_window_effect`` while observed cells and ``no_purchase`` stay exact.
    Rows therefore sum to more than 1: the surplus is the counterfactual
    attraction SQ would add by flying the banks it currently skips.
    """
    ods = (
        market[["origin", "destination"]]
        .sort_values(["origin", "destination"]).reset_index(drop=True)
    )
    od_position = {key: i for i, key in enumerate(zip(ods["origin"], ods["destination"]))}
    windows = [label for label, _, _ in TIME_WINDOWS]

    pax = np.zeros((len(ods), len(PRODUCT_ORDER), len(windows)))
    grouped = (
        sales.groupby(["origin", "destination", "product", "time_window"])["pax"].sum()
    )
    for (origin, destination, product, window), value in grouped.items():
        pax[
            od_position[(origin, destination)],
            PRODUCT_ORDER.index(product),
            windows.index(window),
        ] = value

    # A window is served by an OD iff it carried any pax at all; within a
    # served window every product is present, so the gap is (OD, window)-shaped.
    mask = pax.sum(axis=1) > 0
    report["v1_cells_total"] = int(pax.size)
    report["v1_cells_observed"] = int(mask.sum() * len(PRODUCT_ORDER))
    report["v1_cells_imputed"] = int(pax.size - mask.sum() * len(PRODUCT_ORDER))
    report["ods_by_windows_served"] = {
        int(k): int(v) for k, v in
        pd.Series(mask.sum(axis=1)).value_counts().sort_index().items()
    }

    keyed = market.set_index(["origin", "destination"])
    order = list(zip(ods["origin"], ods["destination"]))
    market_pax = keyed.loc[order, "market_pax"].to_numpy()
    non_sq_pax = keyed.loc[order, "non_sq_pax"].to_numpy()

    filled = pax
    if IMPUTE_ZERO_CELLS:
        theta, delta = fit_window_effect(pax, mask, report)
        counterfactual = theta[:, :, None] * delta[None, None, :]
        filled = np.where(mask[:, None, :], pax, counterfactual)
        served = (mask * delta[None, :]).sum(axis=1)
        report["sq_block_growth_factor"] = {
            "median": round(float(np.median(1.0 / served)), 4),
            "max": round(float((1.0 / served).max()), 4),
        }

    values = np.round(filled / market_pax[:, None, None], V_DECIMALS)
    no_purchase = np.round(non_sq_pax / market_pax, V_DECIMALS)

    if IMPUTE_ZERO_CELLS and not (values > 0).all():
        raise RuntimeError(
            f"{int((values <= 0).sum())} cells still round to zero at "
            f"{V_DECIMALS} decimals; raise V_DECIMALS."
        )

    # The observed block must be untouched by imputation: observed cells plus
    # no_purchase still reconstruct the original share vector exactly.
    observed_sum = (values * mask[:, None, :]).sum(axis=(1, 2)) + no_purchase
    report["v1_observed_block_plus_no_purchase_max_abs_error"] = float(
        np.abs(observed_sum - 1.0).max()
    )
    row_sums = values.sum(axis=(1, 2)) + no_purchase
    report["v1_row_sum"] = {
        "median": round(float(np.median(row_sums)), 4),
        "p95": round(float(np.percentile(row_sums, 95)), 4),
        "max": round(float(row_sums.max()), 4),
    }

    columns = [f"{p}*{w}" for p in PRODUCT_ORDER for w in windows]
    flat = values.reshape(len(ods), -1)
    frame = pd.concat([ods, pd.DataFrame(flat, columns=columns)], axis=1)
    frame["no_purchase"] = no_purchase

    status = np.where(
        np.broadcast_to(mask[:, None, :], pax.shape), "observed", "imputed",
    ).reshape(len(ods), -1)
    status_frame = pd.concat(
        [ods, pd.DataFrame(status, columns=columns)], axis=1,
    )
    return frame, status_frame


def build_od_demand(market: pd.DataFrame) -> pd.DataFrame:
    """od_demand.csv — total (SQ + non-SQ) economy pax per departure week."""
    frame = market.sort_values(["origin", "destination"]).reset_index(drop=True)
    return pd.DataFrame({
        "origin": frame["origin"],
        "destination": frame["destination"],
        "Avg Pax": frame["avg_pax_per_week"].round(2),
    })


# ═══════════════════════════════════════════════════════════════ censoring

def build_code_map(market: pd.DataFrame) -> dict[str, str]:
    """Seeded IATA-code -> 3-digit surrogate map covering airports and cities.

    Airports and cities draw from one surrogate pool so a code is unambiguous
    across both namespaces. City codes must be censored too: leaving LON or
    TYO in place would immediately re-identify the airports beneath them.
    """
    airports = sorted(set(market["origin"]) | set(market["destination"]))
    cities = sorted(
        (set(market["origin_city"]) | set(market["destination_city"]))
        - set(airports)
    )
    codes = airports + cities

    pool = list(range(CENSOR_LOW, CENSOR_HIGH + 1))
    if len(codes) > len(pool):
        raise RuntimeError(
            f"{len(codes)} codes need surrogates but the 3-digit pool holds "
            f"{len(pool)}; widen CENSOR_LOW / CENSOR_HIGH."
        )
    surrogates = random.Random(CENSOR_SEED).sample(pool, len(codes))
    return {code: str(s) for code, s in zip(codes, surrogates)}


def apply_code_map(frame: pd.DataFrame, code_map: dict[str, str]) -> pd.DataFrame:
    """Rewrite every IATA-coded column of ``frame`` through ``code_map``."""
    out = frame.copy()
    for column in ("origin", "destination", "origin_city", "destination_city"):
        if column in out.columns:
            out[column] = out[column].map(code_map)
    if "pair" in out.columns:
        out["pair"] = ["-".join(code_map[c] for c in p.split("-"))
                       for p in out["pair"]]
    return out


def to_od_string(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Collapse origin/destination into the reference files' tuple-string column."""
    out = frame.copy()
    out.insert(
        0, name,
        [str((o, d)) for o, d in zip(out.pop("origin"), out.pop("destination"))],
    )
    return out


# ═══════════════════════════════════════════════════════════════════ driver

def main() -> None:
    report: dict = {
        "departure_window": [DEP_START.strftime("%Y-%m-%d"), DEP_END.strftime("%Y-%m-%d")],
        "censored": CENSOR,
        "censor_seed": CENSOR_SEED if CENSOR else None,
    }

    sales = load_sales(report)
    sales = apply_products(sales, report)
    sales = attach_departure_times(sales, report)

    market = build_market(sales, report)
    # Keep every output on exactly the ODs that survived the market join.
    keep = set(zip(market["origin"], market["destination"]))
    sales = sales[[
        (o, d) in keep for o, d in zip(sales["origin"], sales["destination"])
    ]].copy()
    report["pax_final"] = float(sales["pax"].sum())

    flight, coverage = build_flight(sales, report)
    capacity = build_capacity(sales, report)
    v1, v1_status = build_v1(sales, market, report)
    od_demand = build_od_demand(market)

    # Offer sets are a deliverable in their own right (they are what estimating
    # w would need) and the input to the identifiability test that justifies
    # v2 being an assumption rather than a fit.
    offer = build_offer_sets(sales, report)
    assess_shadow_identifiability(sales, v1, offer, report)
    v2 = build_v2(v1, coverage, report)

    lost = pd.DataFrame(report.pop("pax_lost_to_schedule_join_by_od"))

    if CENSOR:
        code_map = build_code_map(market)
        flight = apply_code_map(flight, code_map)
        capacity = apply_code_map(capacity, code_map)
        coverage = apply_code_map(coverage, code_map)
        v1 = apply_code_map(v1, code_map)
        v1_status = apply_code_map(v1_status, code_map)
        v2 = apply_code_map(v2, code_map)
        offer = apply_code_map(offer, code_map)
        od_demand = apply_code_map(od_demand, code_map)
        market = apply_code_map(market, code_map)
        lost = apply_code_map(lost, code_map).dropna(subset=["origin"])

        pd.DataFrame({
            "iata_code": list(code_map),
            "surrogate": [code_map[c] for c in code_map],
        }).to_csv(OUT_CODE_MAP, index=False)
        report["censored_codes"] = len(code_map)
    elif OUT_CODE_MAP.exists():
        OUT_CODE_MAP.unlink()

    report["pax_lost_to_schedule_join_by_od"] = lost.to_dict("records")

    _SUPP.mkdir(exist_ok=True)
    to_od_string(flight, "Oneway_OD").rename(columns={
        "departure_time": "Departure Time", "product": "Oneway_Product",
        "avg_price": "Avg Price",
    }).to_csv(OUT_FLIGHT)
    to_od_string(od_demand, "Oneway_OD").to_csv(OUT_DEMAND)
    to_od_string(capacity, "Oneway_OD").rename(
        columns={"departure_time": "Departure Time"},
    ).to_csv(OUT_CAPACITY, index=False)
    to_od_string(v1, "OD Pairs").to_csv(OUT_V1, index=False)
    to_od_string(v2, "OD Pairs").to_csv(OUT_V2, index=False)
    to_od_string(v1_status, "OD Pairs").to_csv(OUT_V1_STATUS, index=False)
    to_od_string(offer, "Oneway_OD").to_csv(OUT_OFFER_SETS, index=False)
    coverage.to_csv(OUT_PRICE_COV, index=False)
    market.to_csv(OUT_MARKET, index=False)
    OUT_REPORT.write_text(json.dumps(report, indent=2, default=str))

    ident = report["shadow_identifiability"]
    print(f"flight.csv           {len(flight):>6} rows  -> {OUT_FLIGHT}")
    print(f"od_demand.csv        {len(od_demand):>6} rows  -> {OUT_DEMAND}")
    print(f"flight_capacity.csv  {len(capacity):>6} rows  -> {OUT_CAPACITY}")
    print(f"v1.csv               {len(v1):>6} rows  -> {OUT_V1}")
    print(f"v2.csv               {len(v2):>6} rows  -> {OUT_V2}")
    print(f"offer_sets.csv       {len(offer):>6} rows  -> {OUT_OFFER_SETS}")
    print(f"\nODs: {report['ods_in_output']}   "
          f"pax: {report['pax_final']:,.0f}   "
          f"schedule join: {report['schedule_join_pax_rate']:.4f}")
    if IMPUTE_ZERO_CELLS:
        print(f"filled {report['v1_cells_imputed']} of {report['v1_cells_total']} "
              f"v1 cells (window effect {report['window_effect']}); "
              f"row sum median {report['v1_row_sum']['median']}")
    print(f"\nv2: SHADOW_MODE={SHADOW_MODE!r}  theta median "
          f"{report['shadow_theta']['median']}  "
          f"(cells = 1 - w/v, no_purchase = 1 + sum w / v_0)")
    print(f"    offer sets: {report['offer_set_ods_with_variation']} of "
          f"{report['offer_set_ods']} ODs vary over "
          f"{report['offer_set_od_dates']} OD-dates")
    print(f"    identifiability: {ident['verdict']}")
    if CENSOR:
        print(f"\ncensored {report['censored_codes']} IATA codes "
              f"-> key at {OUT_CODE_MAP.name} (withhold when sharing)")


if __name__ == "__main__":
    main()
