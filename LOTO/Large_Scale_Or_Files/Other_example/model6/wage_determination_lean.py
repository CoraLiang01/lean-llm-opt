import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load work_days.csv as a DataFrame
df = pd.read_csv(os.path.join(script_dir, 'work_days.csv'), index_col=0)
workers = list(df.index)
n = len(workers)

# d_{ij}: days worker j worked on home of worker i
d = df.values  # shape (n, n)

# Create model
m = gp.Model("Worker_Wage_Equilibrium")

# Decision variables: p_j for j=1..n (p_1 fixed at 60.00)
p = m.addVars(n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="p")

# Wage normalization
m.addConstr(p[0] == 60.00, name="wage_normalization")

# Equilibrium constraints for all i=0..n-1
for i in range(n):
    lhs = gp.quicksum(d[j, i] for j in range(n) if j != i) * p[i]
    rhs = gp.quicksum(d[i, j] * p[j] for j in range(n) if j != i)
    m.addConstr(lhs - rhs == 0, name=f"equilibrium_{i}")

# No objective (feasibility problem)
m.setObjective(0, GRB.MINIMIZE)

# Write model to LP file
m.write(os.path.join(script_dir, "model_lean.lp"))

# Solve
m.optimize()

# Output solution
if m.status == GRB.OPTIMAL:
    for j in range(n):
        print(f"{workers[j]}: {p[j].X:.4f}")
