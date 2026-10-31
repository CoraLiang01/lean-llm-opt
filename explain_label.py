#!/usr/bin/env python3
"""
explain_label.py
================
Trace an Air-NRM-CA-LS label back to the CSV cells it came from, so a person
can check it by hand.

    python explain_label.py                              # instance 1, 2 flights
    python explain_label.py --instance 15 --flights 3
    python explain_label.py --instance 1 --od "('865', '678')" --time 01:30

Why read the .lp back
---------------------
The point is to check the artefact, not the generator. This script does not
rebuild the model; it loads SBLP_CA_Label/<n>.lp -- the file that actually
ships -- pulls each coefficient out of it, and looks up independently what the
source CSV says that number should be. If build_largescale_labels.py has a bug,
re-deriving the model with the same code would reproduce the bug and agree with
itself. Reading the file and going back to flight.csv / v1.csv / v2.csv /
od_demand.csv / flight_capacity.csv cannot.

What it prints
--------------
1. structural checks that hold for every instance, machine-verified
2. a full trace of one or more flights: every coefficient, its value in the
   .lp, and the file + row + column it must equal
3. one balance constraint traced end to end
4. a short list of what this does NOT establish

Needs gurobipy to parse the .lp.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

def find_dataset(override: Path | None = None) -> Path:
    """Locate the large_scale folder, from wherever this is run.

    Tried in order: --data-dir, the script's own folder (when it lives inside
    large_scale, next to check_query_pairs.py), the repo-root-relative path
    (when it lives at the top level), then the same two relative to the current
    directory. A folder counts as the dataset only if it holds flight.csv and
    a SBLP_CA_Label directory.
    """
    rel = Path("Test_Dataset") / "Air_NRM" / "large_scale"
    here = Path(__file__).resolve().parent
    candidates = [override] if override else []
    candidates += [here, here / rel, here.parent, Path.cwd(), Path.cwd() / rel]
    for base in candidates:
        if base and (base / "flight.csv").exists() and (base / "SBLP_CA_Label").is_dir():
            return base.resolve()
    tried = "\n  ".join(str(c) for c in candidates if c)
    sys.exit("ERROR: could not find the large_scale dataset (a folder holding "
             "flight.csv and SBLP_CA_Label/).\nLooked in:\n  " + tried +
             "\nPass --data-dir /path/to/large_scale to say where it is.")

PRODUCTS = ["Eco_flexi", "Eco_standard", "Eco_value", "Eco_lite"]
SHORT = {"Eco_flexi": "f", "Eco_standard": "s", "Eco_value": "v", "Eco_lite": "l"}
TIME_WINDOWS = [
    ("(12pm~6pm)", 12 * 60, 18 * 60),
    ("(6pm~10pm)", 18 * 60, 22 * 60),
    ("(10pm~8am)", 22 * 60, 8 * 60),
    ("(8am~12pm)", 8 * 60, 12 * 60),
]
CLAUSE = re.compile(
    r"\(OD\s*=\s*(\('[^']*',\s*'[^']*'\))\s*AND\s*Departure Time='(\d{1,2}:\d{2})'\)")
CONSUME = re.compile(r"each (Eco_\w+) ticket consumes ([\d.]+) unit")
HAIRCUT = re.compile(r"capacity is (\d+(?:\.\d+)?)% of its weekly economy capacity")
TOL = 1e-6


def window_of(t: str) -> str:
    m = int(t.split(":")[0]) * 60 + int(t.split(":")[1])
    for label, lo, hi in TIME_WINDOWS:
        if lo < hi:
            if lo <= m < hi:
                return label
        elif m >= lo or m < hi:
            return label
    raise ValueError(t)


def lp_name(od: str, time: str, short: str) -> str:
    """Gurobi rewrites ':' when it writes LP format, so match on that form."""
    return f"x_{od}_{time}_{short}".replace(" ", "_").replace(":", "_")


def verdict(ok: bool) -> str:
    return "OK " if ok else "MISMATCH"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instance", type=int, default=1, help="1..15")
    ap.add_argument("--flights", type=int, default=2,
                    help="how many flights to trace in full")
    ap.add_argument("--od", default=None, help="trace this OD only, e.g. \"('865', '678')\"")
    ap.add_argument("--time", default=None, help="with --od, e.g. 01:30")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="path to the large_scale folder, if auto-detection fails")
    args = ap.parse_args()

    BASE = find_dataset(args.data_dir)
    LABEL_DIR = BASE / "SBLP_CA_Label"

    try:
        import gurobipy as gp
    except Exception:
        sys.exit("ERROR: gurobipy is needed to read the .lp file "
                 "(pip install gurobipy; no licence needed at this size)")

    # convert_uflp_lp_to_txt.py's convention is to ship the LP text as .txt, so
    # accept either extension. The contents are identical either way; gurobipy
    # needs the .lp suffix to pick its reader, so a .txt is copied to a temp
    # file first rather than renamed in place.
    lp_path = next((LABEL_DIR / f"{args.instance}{ext}"
                    for ext in (".lp", ".txt")
                    if (LABEL_DIR / f"{args.instance}{ext}").exists()), None)
    if lp_path is None:
        sys.exit(f"ERROR: neither {args.instance}.lp nor {args.instance}.txt in "
                 f"{LABEL_DIR} -- run build_largescale_labels.py")
    read_path = lp_path
    if lp_path.suffix != ".lp":
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / f"{args.instance}.lp"
        tmp.write_text(lp_path.read_text(encoding="utf-8", errors="ignore"))
        read_path = tmp

    query = pd.read_csv(BASE / "query_CA.csv")
    text = query["Query"].iloc[args.instance - 1]
    flights = [(o.strip(), t.strip()) for o, t in CLAUSE.findall(text)]
    consumption = {p: float(v) for p, v in CONSUME.findall(text)}
    cut = HAIRCUT.search(text)
    fraction = float(cut.group(1)) / 100.0 if cut else 1.0
    ods = sorted({o for o, _ in flights})

    flight_csv = pd.read_csv(BASE / "flight.csv")
    price = flight_csv.set_index(
        ["Oneway_OD", "Departure Time", "Oneway_Product"])["Avg Price"]
    cap = pd.read_csv(BASE / "flight_capacity.csv").set_index(
        ["Oneway_OD", "Departure Time"])["Y Seats/Week"]
    v1 = pd.read_csv(BASE / "v1.csv").set_index("OD Pairs")
    v2 = pd.read_csv(BASE / "v2.csv").set_index("OD Pairs")
    demand = pd.read_csv(BASE / "od_demand.csv").set_index("Oneway_OD")["Avg Pax"]

    gp.setParam("OutputFlag", 0)
    model = gp.read(str(read_path))
    obj = {v.VarName: v.Obj for v in model.getVars()}
    names = {v.VarName for v in model.getVars()}

    print(f"instance {args.instance}   {lp_path}")
    print(f"query says: {len(flights)} flights, {len(ods)} ODs, "
          f"consumption {consumption}, capacity haircut {fraction:.0%}")
    print()

    # ── 1. structural checks ────────────────────────────────────────────────
    print("1. STRUCTURE — machine-checked, must hold for every instance")
    n_cap = sum(1 for c in model.getConstrs() if c.ConstrName.startswith("capacity"))
    n_scale = sum(1 for c in model.getConstrs() if c.ConstrName.startswith("scale"))
    n_bal = sum(1 for c in model.getConstrs() if c.ConstrName.startswith("balance"))
    checks = [
        (f"variables = 4 x flights + ODs = {4*len(flights)}+{len(ods)}",
         model.NumVars, 4 * len(flights) + len(ods)),
        ("capacity constraints = flights", n_cap, len(flights)),
        ("scale constraints = 4 x flights", n_scale, 4 * len(flights)),
        ("balance constraints = ODs", n_bal, len(ods)),
    ]
    for label, got, want in checks:
        print(f"   [{verdict(got == want)}] {label:<44} .lp has {got}, expected {want}")
    print(f"   [{verdict(model.ModelSense == -1)}] objective is a maximisation")
    zero_cells = sum(1 for od, t in flights for p in PRODUCTS
                     if float(v2.loc[od, f"{p}*{window_of(t)}"]) == 0.0)
    print(f"   [note ] {zero_cells} cell(s) have v2 = 0; those terms are absent from "
          f"the balance constraint by design")
    print()

    # ── 2. per-flight trace ─────────────────────────────────────────────────
    if args.od:
        picked = [(args.od, args.time)] if args.time else \
                 [f for f in flights if f[0] == args.od]
        if not picked:
            sys.exit(f"ERROR: {args.od} {args.time or ''} is not in instance {args.instance}")
    else:
        step = max(1, len(flights) // max(args.flights, 1))
        picked = flights[::step][:args.flights]

    print("2. COEFFICIENT TRACE — open the named file, find the named row, compare")
    bad = 0
    for od, t in picked:
        w = window_of(t)
        print(f"\n   flight OD={od}  Departure Time={t}   ->  window {w}")

        print("   objective (max):")
        for p in PRODUCTS:
            nm = lp_name(od, t, SHORT[p])
            got, want = obj.get(nm), float(price[(od, t, p)])
            ok = got is not None and abs(got - want) < TOL
            bad += not ok
            print(f"     [{verdict(ok)}] {got!s:>10}  = flight.csv  Oneway_OD={od}, "
                  f"Departure Time={t}, Oneway_Product={p}  ->  Avg Price {want}")

        c = model.getConstrByName(f"capacity_constraint_{od}_{t}".replace(" ", "_").replace(":", "_"))
        row = model.getRow(c)
        got_lhs = {row.getVar(i).VarName: row.getCoeff(i) for i in range(row.size())}
        raw = float(cap[(od, t)])
        want_rhs = raw * fraction
        ok = abs(c.RHS - want_rhs) < 1e-4
        bad += not ok
        print("   capacity constraint:")
        print(f"     [{verdict(ok)}] RHS {c.RHS:,.2f}  = flight_capacity.csv  "
              f"Y Seats/Week {raw:,.0f} x {fraction:.0%}")
        for p in PRODUCTS:
            nm = lp_name(od, t, SHORT[p])
            g, wv = got_lhs.get(nm), consumption[p]
            ok = g is not None and abs(g - wv) < TOL
            bad += not ok
            print(f"     [{verdict(ok)}] {g!s:>10}  = the query text: "
                  f"'each {p} ticket consumes {wv:g} unit'")

        print("   scale constraint   v1[cell] * x_o  -  v1[no_purchase] * x  >= 0:")
        for p in PRODUCTS:
            nm = f"scale_constraint_{od}_{t}_{SHORT[p]}".replace(" ", "_").replace(":", "_")
            r = model.getRow(model.getConstrByName(nm))
            coeffs = {r.getVar(i).VarName: r.getCoeff(i) for i in range(r.size())}
            g_cell = coeffs.get(f"x_{od}_0".replace(" ", "_"))
            g_np = -coeffs.get(lp_name(od, t, SHORT[p]), 0.0)
            w_cell = float(v1.loc[od, f"{p}*{w}"])
            w_np = float(v1.loc[od, "no_purchase"])
            ok = (g_cell is not None and abs(g_cell - w_cell) < TOL
                  and abs(g_np - w_np) < TOL)
            bad += not ok
            print(f"     [{verdict(ok)}] {g_cell!s:>10} * x_o - {g_np:g} * x   "
                  f"= v1.csv row {od}: column '{p}*{w}' = {w_cell}, "
                  f"no_purchase = {w_np}")

    # ── 3. one balance constraint ───────────────────────────────────────────
    od = picked[0][0]
    c = model.getConstrByName(f"balance_constraint_{od}".replace(" ", "_"))
    r = model.getRow(c)
    coeffs = {r.getVar(i).VarName: r.getCoeff(i) for i in range(r.size())}
    print(f"\n3. BALANCE CONSTRAINT for OD={od}   (coefficients are v2, "
          f"RHS is od_demand)")
    ok = abs(c.RHS - float(demand[od])) < 1e-4
    bad += not ok
    print(f"   [{verdict(ok)}] RHS {c.RHS:,.2f}  = od_demand.csv  Oneway_OD={od} "
          f"-> Avg Pax {float(demand[od]):,.2f}")
    zero_name = f"x_{od}_0".replace(" ", "_")
    g, wv = coeffs.get(zero_name), float(v2.loc[od, "no_purchase"])
    ok = g is not None and abs(g - wv) < TOL
    bad += not ok
    print(f"   [{verdict(ok)}] x_o coefficient {g}  = v2.csv row {od} "
          f"-> no_purchase {wv}")
    shown = 0
    for t in [t for o, t in flights if o == od]:
        for p in PRODUCTS:
            wv = float(v2.loc[od, f"{p}*{window_of(t)}"])
            g = coeffs.get(lp_name(od, t, SHORT[p]))
            if wv == 0.0:
                ok = g is None
                bad += not ok
                print(f"   [{verdict(ok)}] {t} {p:<13} v2 = 0  -> term correctly "
                      f"{'absent' if ok else 'PRESENT'} from the balance constraint")
                continue
            ok = g is not None and abs(g - wv) < TOL
            bad += not ok
            if shown < 4:
                print(f"   [{verdict(ok)}] {t} {p:<13} coefficient {g}  "
                      f"= v2.csv column '{p}*{window_of(t)}' = {wv}")
                shown += 1

    print()
    print(f"traced coefficients: {'all match their source CSV' if bad == 0 else f'{bad} MISMATCH(ES)'}")
    print()
    print("4. WHAT THIS DOES NOT ESTABLISH")
    print("   - that the SBLP formulation itself is the right one. That is a")
    print("     modelling question, settled against Gallego et al. (2015) eq. (7)")
    print("     and by reproducing the small_scale reference labels 15/15.")
    print("   - that the numbers in flight.csv / v1.csv / v2.csv are correct. This")
    print("     checks the label against those files, not the files themselves.")
    print("   - that the optimum is right. Gurobi and HiGHS agree on all 15, which")
    print("     covers the arithmetic but not the formulation.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
