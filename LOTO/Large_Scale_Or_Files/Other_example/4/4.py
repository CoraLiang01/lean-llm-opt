# 4.py
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# ==== Read data ====
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'value.csv')
df = pd.read_csv(csv_path)   
values = df['value'].tolist()
weights = df['weight'].tolist()
items  = df['item'].tolist()
n = len(items)

# ==== Parameters ====
C = 15  # Capacity (weight limit)

# ==== Modeling ====
m = gp.Model("Unbounded_Knapsack")

# Unbounded knapsack: x_i is non-negative integer 
x = m.addVars(n, vtype=GRB.INTEGER, lb=0, name="x")

# Objective: maximize total value
m.setObjective(gp.quicksum(values[i] * x[i] for i in range(n)), GRB.MAXIMIZE)

# Capacity constraint
m.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= C, name="capacity")

# ==== Solve ====
m.optimize()

# ==== Export model ====
m.write(os.path.join(script_dir, "model.lp"))

# ==== Output results ====
if m.status == GRB.OPTIMAL:
    total_value  = m.objVal
    chosen = []
    total_weight = 0
    for i in range(n):
        xi = int(round(x[i].X))
        if xi > 0:
            chosen.append((items[i], xi, values[i], weights[i], xi*values[i], xi*weights[i]))
            total_weight += xi * weights[i]

    # Print in descending order by value contribution
    chosen.sort(key=lambda t: t[4], reverse=True)

    print("\n=== Optimal Solution ===")
    print(f"Total value: {total_value:.0f}")
    print(f"Total weight: {total_weight}")
    print(f"Number of selected items (types): {len(chosen)}\n")
    print("item  quantity  unit_value  unit_weight  value_contrib  weight_used")
    for (it, xi, v, w, vcontrib, wcontrib) in chosen:
        print(f"{it:>4}  {xi:>8}  {v:>12}  {w:>12}  {vcontrib:>8}  {wcontrib:>8}")
else:
    print("No optimal solution found, model status code:", m.status)

