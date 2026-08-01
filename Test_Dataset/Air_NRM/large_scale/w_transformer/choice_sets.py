"""Choice-set construction for the transformer shadow-attraction probe.

Assembles, for one Air_NRM OD, the choice set the v1 transformer scores:

  * **SQ block** — the 16 Air_NRM cells (4 fare families x 4 departure-time
    windows) at their own observed fares and their own real representative
    departure times. This is the grid ``v1.csv`` / ``v2.csv`` are written on,
    so a closure counterfactual maps 1:1 onto a v2 cell.
  * **OAL block** — the market's real non-SQ itineraries from the simulator's
    ``Itinerary.csv`` asset at their own ``OW_Amt`` fares, with departure times
    / durations / stops joined from ``Flight.csv`` + ``OW_all.csv``.

The OAL block is the GAM's outside option: ``v1.csv``'s ``no_purchase`` is
``non_sq_pax / market_pax``, i.e. "bought a non-SQ carrier", which is exactly
what an OAL row is here.

SCOPE LIMIT. The shipped checkpoint's vocabulary covers four cities only —
DEL, LON, SIN, SYD — and four points of sale (AU, GB, IN, SG); the live
``Itinerary.csv`` asset is on the same four-city network. Feeding it any other
OD maps origin/destination to ``__UNKNOWN__``, so only the ODs listed in
``IN_VOCAB_ODS`` can be probed. See ``README_w_transformer.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_BUILD = _HERE.parent
_BACKEND = _BUILD.parents[1] / "app" / "backend"
_NEW_DATA = _BACKEND / "api" / "new_data"
_RAW_SALES = _BACKEND / "api" / "Raw_data" / "Raw_Sales"

PRODUCTS = ["Eco_flexi", "Eco_standard", "Eco_value", "Eco_lite"]
WINDOWS = ["(12pm~6pm)", "(6pm~10pm)", "(10pm~8am)", "(8am~12pm)"]
WIN_MID = {"(12pm~6pm)": 15 * 60, "(6pm~10pm)": 20 * 60,
           "(10pm~8am)": 3 * 60, "(8am~12pm)": 10 * 60}

# Representative RBD per fare family, taken from FARE_FAMILY_POLICY's (SQ, Y)
# row in the order build_air_nrm_inputs.py names them.
FAMILY_RBD = {"Eco_flexi": "Y", "Eco_standard": "M", "Eco_value": "N", "Eco_lite": "K"}

# Air_NRM OD (airport-keyed, real IATA) -> the transformer/Itinerary market it
# belongs to: (origin city, destination city, POS). Only these are in vocab.
IN_VOCAB_ODS: dict[tuple[str, str], tuple[str, str, str]] = {
    ("DEL", "SIN"): ("DEL", "SIN", "IN"),
    ("LHR", "SIN"): ("LON", "SIN", "GB"),
    ("LGW", "SIN"): ("LON", "SIN", "GB"),
    ("SIN", "LHR"): ("SIN", "LON", "SG"),
    ("SIN", "LGW"): ("SIN", "LON", "SG"),
    ("SIN", "SYD"): ("SIN", "SYD", "SG"),
    ("SYD", "SIN"): ("SYD", "SIN", "AU"),
    # ('SIN','DEL') is in the transformer vocab but absent from Itinerary.csv,
    # which carries no SIN->DEL market. Not probeable.
}


def window_of(minutes: float) -> str:
    if 12 * 60 <= minutes < 18 * 60:
        return "(12pm~6pm)"
    if 18 * 60 <= minutes < 22 * 60:
        return "(6pm~10pm)"
    if 8 * 60 <= minutes < 12 * 60:
        return "(8am~12pm)"
    return "(10pm~8am)"


def _hhmm(minutes: float) -> int:
    m = int(round(minutes)) % 1440
    return (m // 60) * 100 + (m % 60)


# ────────────────────────────────────────────────────────── Air_NRM SQ grid

class AirNrmGrid:
    """The 16-cell SQ grid per OD: attractions, fares, representative times."""

    def __init__(self) -> None:
        code = pd.read_csv(_BUILD / "Supplement" / "airport_code_map.csv")
        self.decode = dict(zip(code.surrogate.astype(str), code.iata_code))

        self.v1 = pd.read_csv(_BUILD / "v1.csv")
        self.cols = [f"{p}*{w}" for p in PRODUCTS for w in WINDOWS]
        ods = [tuple(s.strip("()").replace("'", "").split(", "))
               for s in self.v1["OD Pairs"]]
        self.ods = [(self.decode.get(a, a), self.decode.get(b, b)) for a, b in ods]
        self.row_of = {od: i for i, od in enumerate(self.ods)}

        self.V = self.v1[self.cols].to_numpy(float).reshape(-1, 4, 4)
        self.v0 = self.v1["no_purchase"].to_numpy(float)

        status = pd.read_csv(_BUILD / "Supplement" / "v1_cell_status.csv")
        self.observed = (status[self.cols].to_numpy() == "observed").reshape(-1, 4, 4)

        mkt = pd.read_csv(_BUILD / "Supplement" / "od_market_size.csv")
        mkt["od"] = [(self.decode.get(str(a), str(a)), self.decode.get(str(b), str(b)))
                     for a, b in zip(mkt.origin, mkt.destination)]
        self.sq_share = dict(zip(mkt.od, mkt.sq_share))

        fp = pd.read_csv(_BUILD / "Supplement" / "flight_price_coverage.csv")
        fp["od"] = [(self.decode.get(str(a), str(a)), self.decode.get(str(b), str(b)))
                    for a, b in zip(fp.origin, fp.destination)]
        fp["dep_min"] = (fp.departure_time.str.slice(0, 2).astype(int) * 60
                         + fp.departure_time.str.slice(3, 5).astype(int))
        fp["win"] = fp["dep_min"].map(window_of)
        self._fp = fp

        # per-(od, product, window) fare: pax-weighted across the departures in
        # the window; the OD x product mean where the window is not served.
        self.fare = {}
        self.dep_min = {}
        for od, g in fp.groupby("od"):
            f = np.full((4, 4), np.nan)
            for (p, w), gg in g.groupby(["product", "win"]):
                f[PRODUCTS.index(p), WINDOWS.index(w)] = np.average(
                    gg.avg_price, weights=np.maximum(gg.pax, 1e-9))
            for p, gg in g.groupby("product"):
                row = f[PRODUCTS.index(p)]
                row[np.isnan(row)] = float(gg.od_product_price.iloc[0])
            self.fare[od] = f

            t = np.array([WIN_MID[w] for w in WINDOWS], float)
            for w, gg in g.groupby("win"):
                t[WINDOWS.index(w)] = np.average(gg.dep_min, weights=np.maximum(gg.pax, 1e-9))
            self.dep_min[od] = t

    def sq_duration_seconds(self, od: tuple[str, str]) -> float:
        return _flight_table().duration(od, "SQ")


# ────────────────────────────────────────────────────────── schedule / OAL

class _FlightTable:
    """Departure times, durations and stops per (airline, airport OD)."""

    def __init__(self) -> None:
        f = pd.read_csv(
            _NEW_DATA / "Flight.csv",
            usecols=["AirlineID", "Origin", "Destination", "DepartureTime",
                     "Duration", "Stops", "Cabin", "Ops/Week"],
        )
        f = f[f.Cabin == "Y"]
        self._by_leg = {
            k: g for k, g in f.groupby(["Origin", "Destination", "AirlineID"])
        }
        self._market_dur = f.groupby(["Origin", "Destination"])["Duration"].median()

    def duration(self, od: tuple[str, str], airline: str) -> float:
        g = self._by_leg.get((od[0], od[1], airline))
        if g is not None and len(g):
            return float(g.Duration.median())
        m = self._market_dur.get((od[0], od[1]))
        return float(m) if m is not None and np.isfinite(m) else 8 * 3600.0

    def departure_minutes(self, od: tuple[str, str], airline: str) -> float | None:
        g = self._by_leg.get((od[0], od[1], airline))
        if g is None or not len(g):
            return None
        hhmm = g.DepartureTime.astype(int)
        w = g["Ops/Week"].astype(float).clip(lower=1e-9)
        mins = (hhmm // 100) * 60 + (hhmm % 100)
        return float(np.average(mins, weights=w))


_FLIGHT_TABLE: _FlightTable | None = None


def _flight_table() -> _FlightTable:
    global _FLIGHT_TABLE
    if _FLIGHT_TABLE is None:
        _FLIGHT_TABLE = _FlightTable()
    return _FLIGHT_TABLE


class OalBlock:
    """Real non-SQ itineraries per (origin city, destination city, POS)."""

    def __init__(self) -> None:
        it = pd.read_csv(_NEW_DATA / "Itinerary.csv")
        it = it[(it.Cabin == "Y") & (it["OW/RT"] == "OO") & (it.AirlineID != "SQ")]
        self._by_market = {k: g for k, g in
                           it.groupby(["Origin_Trip", "Destination_Trip", "POS"])}
        ow = pd.read_csv(_NEW_DATA / "OW_all.csv")
        self._legs = {
            (r.AirlineID, r.Routing_OW): [x for x in
                                          (r.Leg1, r.Leg2, r.Leg3, r.Leg4)
                                          if isinstance(x, str) and x]
            for r in ow.itertuples()
        }

    def build(self, market: tuple[str, str, str], max_rows: int,
              rng: np.random.Generator) -> pd.DataFrame:
        g = self._by_market.get(market)
        if g is None or g.empty:
            raise KeyError(f"no OAL itineraries for market {market}")
        g = g.rename(columns={"Bkg Cls": "BkgCls"})
        if len(g) > max_rows:
            # Stratify by airline so the competitive set keeps its carrier mix
            # rather than being truncated to whoever happens to sort first.
            frac = max_rows / len(g)
            g = (g.groupby("AirlineID", group_keys=False)
                  .sample(frac=frac, random_state=int(rng.integers(1 << 31)))
                  .head(max_rows))

        ft = _flight_table()
        rows = []
        for r in g.itertuples():
            routing = str(r.Routing_Outbound)
            legs = self._legs.get((r.AirlineID, routing))
            if legs:
                dur = sum(ft.duration(tuple(l.split("-")), r.AirlineID) for l in legs)
                first = tuple(legs[0].split("-"))
            else:
                parts = routing.split("-")
                legs = [f"{a}-{b}" for a, b in zip(parts, parts[1:])] or [routing]
                dur = sum(ft.duration(tuple(l.split("-")), r.AirlineID) for l in legs)
                first = tuple(legs[0].split("-"))
            dep = ft.departure_minutes(first, r.AirlineID)
            if dep is None:
                dep = float(rng.integers(0, 1440))
            stops = max(0, len(legs) - 1)
            rows.append({
                "AirlineID": r.AirlineID,
                "Departure_Time_OWOutbound": _hhmm(dep),
                "StopsOutbound": stops,
                "Duration_OWOutbound": dur,
                "Price": float(r.OW_Amt),
                "Duration": dur,
                "Routing_Outbound": routing,
                "RBD": str(r.BkgCls),
                "is_sq": False,
                "cell": -1,
            })
        return pd.DataFrame(rows).reset_index(drop=True)


# ────────────────────────────────────────────────────────── SQ block

def sq_block(grid: AirNrmGrid, od: tuple[str, str]) -> pd.DataFrame:
    """The 16 Air_NRM cells as choice rows. ``cell`` indexes product*4+window."""
    fare = grid.fare[od]
    dep = grid.dep_min[od]
    dur = grid.sq_duration_seconds(od)
    routing = f"{od[0]}-{od[1]}"
    rows = []
    for pi, p in enumerate(PRODUCTS):
        for wi, w in enumerate(WINDOWS):
            rows.append({
                "AirlineID": "SQ",
                "Departure_Time_OWOutbound": _hhmm(dep[wi]),
                "StopsOutbound": 0,
                "Duration_OWOutbound": dur,
                "Price": float(fare[pi, wi]),
                "Duration": dur,
                "Routing_Outbound": routing,
                "RBD": FAMILY_RBD[p],
                "is_sq": True,
                "cell": pi * 4 + wi,
            })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────── customers

_SALES_CACHE: dict[str, pd.DataFrame] = {}


def _sales_for_pos(pos: str) -> pd.DataFrame:
    if pos in _SALES_CACHE:
        return _SALES_CACHE[pos]
    path = next(_RAW_SALES.glob(f"POS {pos} NUS Sales Data*.csv"))
    d = pd.read_csv(path, usecols=["Iss Date", "Dept Date", "Trip OD Itinerary",
                                   "Cabin Class", "Carrier Designator",
                                   "Ticketed Pax"], low_memory=False)
    # The raw feed pads its string columns ('SQ ', 'Y '), exactly as
    # build_air_nrm_inputs.load_sales() handles it.
    for c in ("Carrier Designator", "Cabin Class", "Trip OD Itinerary"):
        d[c] = d[c].astype(str).str.strip()
    d = d[(d["Carrier Designator"] == "SQ") & (d["Cabin Class"] == "Y")]
    d["iss"] = pd.to_datetime(d["Iss Date"], format="%m/%d/%y", errors="coerce")
    d["dep"] = pd.to_datetime(d["Dept Date"], format="%d %b %Y", errors="coerce")
    d = d.dropna(subset=["iss", "dep"])
    d["lead_days"] = (d["dep"] - d["iss"]).dt.days
    d = d[(d.lead_days >= 0) & (d.lead_days <= 365)]
    _SALES_CACHE[pos] = d
    return d


def sample_customers(od: tuple[str, str], pos: str, n: int,
                     rng: np.random.Generator) -> pd.DataFrame:
    """Draw customers with this OD's own observed lead-day distribution."""
    d = _sales_for_pos(pos)
    key = f"{od[0]}-{od[1]}"
    sub = d[d["Trip OD Itinerary"].astype(str).str.strip() == key]
    if len(sub) < 50:
        sub = d
    pax = pd.to_numeric(sub["Ticketed Pax"], errors="coerce").fillna(1).clip(lower=1)
    idx = rng.choice(len(sub), size=n, replace=True,
                     p=(pax / pax.sum()).to_numpy())
    picked = sub.iloc[idx]
    out = pd.DataFrame({
        "PNR_ID": np.arange(n),
        "POS": pos,
        "originTrip": IN_VOCAB_ODS[od][0],
        "destinationTrip": IN_VOCAB_ODS[od][1],
        "ow_rt": "OO",
        "cabin": "Y",
        "pax": np.minimum(pd.to_numeric(picked["Ticketed Pax"],
                                        errors="coerce").fillna(1).to_numpy(), 9),
        "random": rng.random(n),
        "lead_days": picked["lead_days"].to_numpy(),
        "stay_days": 0,
        "issueDate": picked["iss"].to_numpy(),
        "Outbound_date": picked["dep"].to_numpy(),
        "Inbound_date": pd.NaT,
    })
    return out
