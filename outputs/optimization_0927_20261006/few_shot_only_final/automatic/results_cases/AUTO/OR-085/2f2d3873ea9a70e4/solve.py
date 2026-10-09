import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
node_ids = [str(i) for i in range(1, 16)]
distance = {}
for i in node_ids:
    distance[i] = {}
    row = df[df['Unnamed: 0'] == i]
    if row.empty:
        raise ValueError(f'Missing row for location {i} in CSV.')
    for j in node_ids:
        val = row.iloc[0][j]
        try:
            distance[i][j] = float(val)
        except Exception:
            raise ValueError(f"Non-numeric or missing distance from {i} to {j}: '{val}'")
for i in node_ids:
    for j in node_ids:
        if i != j:
            if abs(distance[i][j] - distance[j][i]) > 1e-08:
                raise ValueError(f'Distance matrix not symmetric at ({i},{j}) and ({j},{i})')
N = node_ids
m = gp.Model('TSP_15_Cities')
x_vars = m.addVars([(i, j) for i in N for j in N if i != j], vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([i for i in N if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=15, name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for i in N for j in N if i != j)), gp.GRB.MINIMIZE)
for i in N:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in N if j != i)) == 1, name=f'depart_{i}')
for j in N:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in N if i != j)) == 1, name=f'arrive_{j}')
for i in N:
    if i == '1':
        continue
    for j in N:
        if j == '1' or i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + 15 * x_vars[i, j] <= 14, name=f'mtz_{i}_{j}')
m.optimize()