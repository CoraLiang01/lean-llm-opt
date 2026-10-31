# model47.py
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np

# === Read distance matrix ===
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "DistanceMatrix.csv")
df = pd.read_csv(csv_path, index_col=0)
nodes = list(df.index)              # ['Depot','A',...,'J']
n = len(nodes)                      # 11
dist = df.values.astype(float)

# === Basic sets ===
V = range(n)
customers = range(1, n)             # 1..10
arcs = [(i, j) for i in V for j in V if i != j]  

# === Modeling ===
m = gp.Model("tsp")

# Binary arc variables X_ij
X = m.addVars(arcs, vtype=GRB.BINARY, name="X")

# MTZ order variables U_i (customers only)
U = m.addVars(customers, lb=1, ub=n-1, vtype=GRB.CONTINUOUS, name="U")

# Objective: total distance
m.setObjective(gp.quicksum(dist[i, j] * X[i, j] for (i, j) in arcs), GRB.MINIMIZE)

# Each node has exactly one outgoing arc
for i in V:
    m.addConstr(gp.quicksum(X[i, j] for j in V if j != i) == 1, name=f"depart_{i}")

# Each node has exactly one incoming arc
for j in V:
    m.addConstr(gp.quicksum(X[i, j] for i in V if i != j) == 1, name=f"arrive_{j}")

# MTZ subtour elimination (only between customers)
M = n - 1  # Number of customers
for i in customers:
    for j in customers:
        if i != j:
            m.addConstr(U[i] - U[j] + M * X[i, j] <= M - 1, name=f"mtz_{i}_{j}")

# Optional: for symmetric TSP, symmetry breaking can be added (to reduce equivalent optimal solutions), omitted here

m.optimize()

# Export model
m.write(os.path.join(script_dir, "model.lp"))

# === Interpret and print route ===
if m.status == GRB.OPTIMAL:
    print(f"\nOptimal distance = {m.ObjVal:.2f} km")
    # Successor table
    succ = {i: None for i in V}
    for i, j in arcs:
        if X[i, j].X > 0.5:
            succ[i] = j

    # Route starting from Depot(0)
    route = [0]
    current = 0
    for _ in range(n):  # Safety limit
        nxt = succ[current]
        route.append(nxt)
        if nxt == 0:
            break
        current = nxt

    name_map = {k: nodes[k] for k in V}
    print("Route:")
    print(" -> ".join(name_map[i] for i in route))

    # Display variable count verification
    print(f"\nBinary arc variables: {len(arcs)} (expected {n*(n-1)})")
else:
    print("No optimal solution found. Status:", m.status)

