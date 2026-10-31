# solve_two_factory_production.py
# Linear program: two factories (A, B), products Pk, maximize profit with per-factory resource limits only.
# Variables x_A_Pk, x_B_Pk are continuous and >= 0.

from pathlib import Path
import sys
import os
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as e:
    sys.exit(
        "gurobipy is not installed. Install Gurobi & its Python bindings first. "
        "See: https://www.gurobi.com/documentation/"
    )

# -------------------------
# Configuration (edit paths if needed)
# -------------------------
DEFAULT_RR = "Resource_Requirements.csv"
DEFAULT_PU = "Profit_per_Unit.csv"

# Resource capacities (per factory, per week)
CAPACITY = {
    "A": {"Labor Hours": 100.0, "Machine Hours": 90.0,  "Raw Material Units": 80.0},
    "B": {"Labor Hours": 80.0,  "Machine Hours": 100.0, "Raw Material Units": 90.0},
}

# -------------------------
# Helpers
# -------------------------
def find_file(name: str) -> Path:
    """Resolve a data file by searching several likely locations.

    Search order:
      1. Provided path as-is (absolute or relative to CWD)
      2. Directory of this script
      3. Parent directory of this script (one level up)
      4. Directory pointed to env var DATA_DIR (if set)
      5. /mnt/data (for some notebook environments)

    Returns the first existing Path or exits with an informative message listing attempted paths.
    """
    candidates = []

    given = Path(name)
    candidates.append(given if given.is_absolute() else Path.cwd() / given)

    try:
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir / name)
        candidates.append(script_dir.parent / name)
    except Exception:
        # __file__ may not be set in some interactive contexts; ignore.
        pass

    data_dir_env = os.getenv("DATA_DIR")
    if data_dir_env:
        candidates.append(Path(data_dir_env) / name)

    candidates.append(Path("/mnt/data") / name)

    for c in candidates:
        if c.exists():
            return c

    # Deduplicate while preserving order for message
    seen = []
    display = []
    for c in candidates:
        cp = str(c)
        if cp not in seen:
            seen.append(cp)
            display.append(cp)
    sys.exit(
        "Could not find required file: "
        f"{name}. Looked in: \n  - " + "\n  - ".join(display) +
        "\nHint: run the script from the project root or pass explicit paths, e.g.\n"
        f"  python 13/solve_factory_lp.py 13/{name} Profit_per_Unit.csv"
    )

def parse_rr(rr_df: pd.DataFrame):
    """
    Parse Resource_Requirements.csv shaped as:
      Key, Labor Hours, Machine Hours, Raw Material Units
    where Key looks like 'A-P1', 'B-P1', ...
    Returns:
      requirements: dict[(factory, product)] -> dict[resource] -> requirement per unit
      resources: list of resource column names (as seen in CSV)
      pairs: sorted list of (factory, product) present in RR
    """
    if "Key" not in rr_df.columns:
        sys.exit("Resource_Requirements.csv must contain a 'Key' column (e.g., 'A-P1').")

    # Identify resource columns (everything that's not 'Key')
    resources = [c for c in rr_df.columns if c != "Key"]
    needed = {"Labor Hours", "Machine Hours", "Raw Material Units"}
    missing = needed - set(resources)
    if missing:
        sys.exit(f"Resource_Requirements.csv is missing columns: {sorted(missing)}")

    def split_key(k: str):
        k = str(k).strip()
        if "-" not in k:
            sys.exit(f"Invalid Key value '{k}' (expected like 'A-P1').")
        f, p = k.split("-", 1)
        f = f.strip()
        p = p.strip()
        if f not in {"A", "B"}:
            sys.exit(f"Factory code '{f}' in Key '{k}' must be 'A' or 'B'.")
        if not p.startswith("P"):
            sys.exit(f"Product code '{p}' in Key '{k}' must start with 'P'.")
        return f, p

    requirements = {}
    pairs = set()

    for _, row in rr_df.iterrows():
        f, p = split_key(row["Key"])
        req_map = {res: float(row[res]) for res in resources}
        requirements[(f, p)] = req_map
        pairs.add((f, p))

    # Sort products by number, e.g., P1, P2, ..., P10, ...
    def product_key(pp: str):
        try:
            return int(pp[1:])
        except Exception:
            return 10**9

    pairs = sorted(pairs, key=lambda t: (t[0], product_key(t[1])))
    return requirements, resources, pairs

def parse_profits(pu_df: pd.DataFrame):
    """
    Parse Profit_per_Unit.csv shaped as:
       Factory, P1, P2, P3, ...
       Factory A, ...
       Factory B, ...
    Returns:
      profit_df: dataframe indexed by 'Factory' with columns Pk, values are profits per unit
      a mapping from factory code ('A'/'B') -> row label in profit_df ('Factory A'/'Factory B')
    """
    if "Factory" not in pu_df.columns:
        sys.exit("Profit_per_Unit.csv must contain a 'Factory' column with 'Factory A' and 'Factory B' rows.")

    profit_df = pu_df.set_index("Factory")
    # Try to discover canonical row labels for A and B
    candidates = [idx for idx in profit_df.index]
    # We expect exact matches:
    for needed in ["Factory A", "Factory B"]:
        if needed not in candidates:
            sys.exit(f"Profit_per_Unit.csv must have a row labeled '{needed}' (found {list(profit_df.index)})")

    factory_label = {"A": "Factory A", "B": "Factory B"}
    return profit_df, factory_label

# -------------------------
# Main modeling routine
# -------------------------
def main(rr_path: str = DEFAULT_RR, pu_path: str = DEFAULT_PU):
    rr_file = find_file(rr_path)
    pu_file = find_file(pu_path)

    rr_df = pd.read_csv(rr_file)
    pu_df = pd.read_csv(pu_file)

    requirements, resources, pairs_rr = parse_rr(rr_df)
    profit_df, factory_label = parse_profits(pu_df)

    # Products present in profit table
    product_cols = [c for c in profit_df.columns if str(c).startswith("P")]
    product_set_profit = set(product_cols)

    # Only build variables for (factory, product) that exist in RR and in profits
    pairs = [(f, p) for (f, p) in pairs_rr if p in product_set_profit]

    if not pairs:
        sys.exit("No overlapping (factory, product) pairs found between RR and Profit tables.")

    # Validate capacities have the same resource names as in RR
    for f in ["A", "B"]:
        missing_res = set(resources) - set(CAPACITY[f].keys())
        if missing_res:
            sys.exit(
                f"CAPACITY for factory {f} is missing resource keys {sorted(missing_res)}. "
                f"Expected resources: {resources}"
            )

    # -------------------------
    # Build model
    # -------------------------
    m = gp.Model("TwoFactoryProduction")

    # Decision variables: x_{A,Pk}, x_{B,Pk} >= 0 (continuous by default)
    x = {}
    for (f, p) in pairs:
        var_name = f"x_{f}_{p}"
        x[(f, p)] = m.addVar(lb=0.0, name=var_name)

    m.update()

    # Objective: maximize sum(profit[f,p] * x[f,p])
    obj_terms = []
    for (f, p), var in x.items():
        flabel = factory_label[f]  # 'Factory A' or 'Factory B'
        if p not in profit_df.columns:
            # This shouldn't happen due to filtering above, but guard anyway.
            continue
        prof = float(profit_df.loc[flabel, p])
        obj_terms.append(prof * var)
    m.setObjective(gp.quicksum(obj_terms), GRB.MAXIMIZE)

    # Constraints: per-factory resource capacities
    # For each factory f and resource r:
    #   sum_p requirement(f,p,r) * x[f,p] <= CAPACITY[f][r]
    for f in ["A", "B"]:
        for r in resources:
            terms = []
            for (ff, p), var in x.items():
                if ff != f:
                    continue
                # requirement per unit for (f,p) on resource r
                req = requirements[(ff, p)][r]
                terms.append(req * var)
            m.addConstr(gp.quicksum(terms) <= CAPACITY[f][r], name=f"cap_{f}_{r.replace(' ', '_')}")

    # Optimize
    m.optimize()

    # Persist LP formulation for inspection (same pattern as other scripts)
    try:
        script_dir = Path(__file__).resolve().parent
        m.write(str(script_dir / "model.lp"))
    except Exception as e:
        print(f"Warning: could not write model.lp ({e})")

    # -------------------------
    # Reporting
    # -------------------------
    if m.status == GRB.OPTIMAL:
        total_profit = m.objVal
        print(f"\nOptimal total profit: ${total_profit:,.2f}\n")

        # Production plan (only positive values shown)
        rows = []
        for (f, p), var in x.items():
            val = var.X
            if val > 1e-9:
                rows.append({
                    "Factory": f,
                    "Product": p,
                    "Quantity": val,
                    "Profit_per_Unit": float(profit_df.loc[factory_label[f], p]),
                    "Contribution_to_Profit": float(profit_df.loc[factory_label[f], p]) * val
                })

        plan_df = pd.DataFrame(rows).sort_values(by=["Factory", "Product"], key=lambda col:
                                                 col.map(lambda s: int(s[1:]) if isinstance(s, str) and s.startswith("P") else s)
                                                 if col.name == "Product" else col)
        if plan_df.empty:
            print("No production recommended (all variables at 0).")
        else:
            print("Production plan (positive quantities only):")
            print(plan_df.to_string(index=False))

        # Resource usage summary
        print("\nResource usage by factory:")
        usage_rows = []
        for f in ["A", "B"]:
            for r in resources:
                used = 0.0
                for (ff, p), var in x.items():
                    if ff != f:
                        continue
                    used += requirements[(ff, p)][r] * var.X
                cap = CAPACITY[f][r]
                slack = cap - used
                usage_rows.append({
                    "Factory": f,
                    "Resource": r,
                    "Used": used,
                    "Capacity": cap,
                    "Slack": slack
                })
        usage_df = pd.DataFrame(usage_rows)
        print(usage_df.to_string(index=False))

    elif m.status == GRB.INFEASIBLE:
        print("\nModel is infeasible. Consider checking the input files or capacities.")
        # Optional: compute IIS to diagnose infeasibilities
        m.computeIIS()
        m.write("model.ilp")
        print("IIS written to model.ilp")
    else:
        print(f"\nSolver ended with status code: {m.status}")
        print("See Gurobi status codes for details.")

if __name__ == "__main__":
    # You can pass custom file paths as args: python solve_two_factory_production.py <RR.csv> <PU.csv>
    rr_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RR
    pu_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PU
    main(rr_arg, pu_arg)
