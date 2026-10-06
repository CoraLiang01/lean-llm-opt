import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',')
locations = ['Depot', 'A', 'B', 'C']
row_ids = df['Unnamed: 0'].astype(str).str.strip().tolist()
col_ids = [col.strip() for col in df.columns if col != 'Unnamed: 0']
for loc in locations:
    if loc not in row_ids:
        raise ValueError(f"Location '{loc}' not found in CSV rows.")
    if loc not in col_ids:
        raise ValueError(f"Location '{loc}' not found in CSV columns.")
distance = {}
for i in locations:
    row = df.loc[df['Unnamed: 0'].astype(str).str.strip() == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in CSV.")
    row = row.iloc[0]
    distance[i] = {}
    for j in locations:
        val = row[j]
        if pd.isnull(val):
            raise ValueError(f'Distance from {i} to {j} is missing in CSV.')
        distance[i][j] = float(val)
arcs = [(i, j) for i in locations for j in locations if i != j]

def solve_tsp():
    m = gp.Model('TSP_SmallCourier')
    x = m.addVars(arcs, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for i, j in arcs)), gp.GRB.MINIMIZE)
    for k in locations:
        m.addConstr(gp.quicksum((x[k, j] for j in locations if j != k)) == 1, name=f'out_{k}')
        m.addConstr(gp.quicksum((x[i, k] for i in locations if i != k)) == 1, name=f'in_{k}')
    customers = [loc for loc in locations if loc != 'Depot']
    u = m.addVars(customers, vtype=gp.GRB.CONTINUOUS, lb=1, ub=len(customers), name='')
    n = len(customers)
    for i in customers:
        for j in customers:
            if i != j:
                m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f'mtz_{i}_{j}')
    m.optimize()
    return m
m = solve_tsp()