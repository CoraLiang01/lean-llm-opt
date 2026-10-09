import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
location_ids = [str(i) for i in range(1, 16)]
n = len(location_ids)
loc_to_idx = {loc: idx for (idx, loc) in enumerate(location_ids)}
idx_to_loc = {idx: loc for (loc, idx) in loc_to_idx.items()}
dist = {i: {} for i in location_ids}
for row in df.itertuples(index=False):
    row_label = str(getattr(row, 'Unnamed: 0')).strip()
    for col in location_ids:
        val = getattr(row, col)
        if val == '':
            continue
        try:
            d = float(val)
        except Exception:
            raise ValueError(f'Non-numeric distance at ({row_label}, {col}): {val}')
        dist[row_label][col] = d
for i in location_ids:
    for j in location_ids:
        if i == j:
            continue
        dij = dist.get(i, {}).get(j, None)
        dji = dist.get(j, {}).get(i, None)
        if dij is not None and dji is None:
            dist[j][i] = dij
        elif dij is None and dji is not None:
            dist[i][j] = dji
        elif dij is None and dji is None:
            raise ValueError(f'Missing distance between {i} and {j}')
m = gp.Model('TSP_15_locations')
x_vars = m.addVars([(i, j) for i in location_ids for j in location_ids if i != j], vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([i for i in location_ids if i != '1'], vtype=gp.GRB.CONTINUOUS, lb=2, ub=n, name='')
m.setObjective(gp.quicksum((dist[i][j] * x_vars[i, j] for i in location_ids for j in location_ids if i != j)), gp.GRB.MINIMIZE)
for i in location_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in location_ids if j != i)) == 1, name=f'leave_{i}')
for j in location_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in location_ids if i != j)) == 1, name=f'enter_{j}')
for i in location_ids:
    if i == '1':
        continue
    for j in location_ids:
        if j == '1' or i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + n * x_vars[i, j] <= n - 1, name=f'mtz_{i}_{j}')
m.optimize()