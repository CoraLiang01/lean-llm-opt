# 6.py
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# === Configuration ===
import os
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(script_dir, "energy.csv")  # You can also switch to energy.csv
DEMAND = 200                  # Total demand

# === Read data ===
df = pd.read_csv(DATA_FILE)

# Basic field validation 
required_cols = {"gen_per_lot", "cost_per_lot"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"CSV missing columns: {missing}")

# === Modeling ===
m = gp.Model("Energy_Procurement")

# Create an integer variable x[i] for each row
x = m.addVars(df.index, vtype=GRB.INTEGER, lb=0, name="x")

# Objective function: minimize total cost
m.setObjective(gp.quicksum(df.loc[i, "cost_per_lot"] * x[i] for i in df.index), GRB.MINIMIZE)

# Demand coverage constraint: total supply >= D
m.addConstr(gp.quicksum(df.loc[i, "gen_per_lot"] * x[i] for i in df.index) >= DEMAND,
            name="demand")

# Optional: set some solver parameters
# m.Params.MIPFocus = 1
# m.Params.TimeLimit = 60

# === Solve ===
m.optimize()

# === Export model ===
m.write(os.path.join(script_dir, "model.lp"))

# === Output results ===
if m.status == GRB.OPTIMAL:
    print(f"Min cost Z = {m.ObjVal:.2f}")
    # Print selected options (x > 0)
    chosen = []
    for i in df.index:
        xi = x[i].X
        if xi > 1e-6:
            row = df.loc[i]
            unit_cost = row["cost_per_lot"] / row["gen_per_lot"]
            chosen.append((row.get("option", f"option_{i}"),
                           row.get("tech", ""),
                           int(round(xi)),
                           row["gen_per_lot"],
                           row["cost_per_lot"],
                           unit_cost))
    # Sort by unit cost to check if cheaper options were selected
    chosen.sort(key=lambda t: t[5])
    for name, tech, qty, gen, cost, uc in chosen:
        print(f"{name:<15} {tech:>11} | x={qty:>3} | gen/lot={gen:>4} | cost/lot={cost:>7.2f} | unit={uc:.3f}")
else:
    print(f"Model did not reach optimality (status={m.status}).")

