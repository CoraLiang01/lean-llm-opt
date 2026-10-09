import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
locations = [str(i) for i in range(1, 16)]
n = len(locations)
if not all((loc in df.columns for loc in locations)):
    missing_cols = [loc for loc in locations if loc not in df.columns]
    raise ValueError(f'Missing columns in distance matrix: {missing_cols}')
if not all((str(df.loc[i, 'Unnamed: 0']).strip() in locations for i in range(n))):
    missing_rows = [str(df.loc[i, 'Unnamed: 0']).strip() for i in range(n) if str(df.loc[i, 'Unnamed: 0']).strip() not in locations]
    raise ValueError(f'Missing rows in distance matrix: {missing_rows}')
distance = {}
for row_idx in range(n):
    i = str(df.loc[row_idx, 'Unnamed: 0']).strip()
    for j in locations:
        if i != j:
            val = df.loc[row_idx, j]
            try:
                distance[i, j] = float(val)
            except Exception:
                raise ValueError(f"Invalid distance value at ({i},{j}): '{val}'")
m = gp.Model('TSP_15_Cities')
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([i for i in locations if i != '1'], vtype=gp.GRB.CONTINUOUS, lb=2, ub=n, name='')
m.setObjective(gp.quicksum((distance[i, j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in locations if j != i)) == 1, name=f'leave_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in locations if i != j)) == 1, name=f'enter_{j}')
for i in locations:
    if i == '1':
        continue
    for j in locations:
        if j == '1' or i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + n * x_vars[i, j] <= n - 1, name=f'mtz_{i}_{j}')
m.optimize()