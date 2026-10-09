import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)

def norm_id(x):
    return x.strip().casefold()
required_locs = ['Depot', 'A', 'B', 'C']
required_locs_norm = [norm_id(loc) for loc in required_locs]
row_ids = df['Unnamed: 0'].tolist()
row_ids_norm = [norm_id(x) for x in row_ids]
row_norm2orig = dict(zip(row_ids_norm, row_ids))
col_ids = list(df.columns)
col_ids_noindex = [c for c in col_ids if c != 'Unnamed: 0']
col_ids_norm = [norm_id(c) for c in col_ids_noindex]
col_norm2orig = dict(zip(col_ids_norm, col_ids_noindex))
for loc_norm in required_locs_norm:
    if loc_norm not in row_norm2orig:
        raise ValueError(f"Location '{loc_norm}' not found in distance matrix rows.")
    if loc_norm not in col_norm2orig:
        raise ValueError(f"Location '{loc_norm}' not found in distance matrix columns.")
locs = [row_norm2orig[loc_norm] for loc_norm in required_locs_norm]
distance = {}
for i in locs:
    row = df[df['Unnamed: 0'] == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in distance matrix.")
    for j in locs:
        val_str = row.iloc[0][j]
        try:
            val = float(val_str)
        except Exception:
            raise ValueError(f"Invalid distance value from '{i}' to '{j}': '{val_str}'")
        distance[i, j] = val
m = gp.Model('TSP_4node')
x_vars = m.addVars(locs, locs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((distance[i, j] * x_vars[i, j] for i in locs for j in locs)), gp.GRB.MINIMIZE)
for i in locs:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in locs if j != i)) == 1, name=f'out_{i}')
    m.addConstr(gp.quicksum((x_vars[j, i] for j in locs if j != i)) == 1, name=f'in_{i}')
for i in locs:
    m.addConstr(x_vars[i, i] == 0, name=f'no_self_{i}')
n = len(locs)
u_vars = m.addVars(locs, vtype=gp.GRB.CONTINUOUS, lb=0, ub=n - 1, name='')
m.addConstr(u_vars['Depot'] == 0, name='u_depot')
for i in locs:
    if i == 'Depot':
        continue
    m.addConstr(u_vars[i] >= 1, name=f'u_lb_{i}')
    m.addConstr(u_vars[i] <= n - 1, name=f'u_ub_{i}')
for i in locs:
    for j in locs:
        if i == j or i == 'Depot' or j == 'Depot':
            continue
        m.addConstr(u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    tour = []
    current = 'Depot'
    visited = set()
    for _ in range(n):
        visited.add(current)
        for j in locs:
            if j != current and x_vars[current, j].X > 0.5:
                tour.append((current, j))
                current = j
                break
    print('Optimal route:')
    route_str = tour[0][0]
    for (i, j) in tour:
        route_str += f' -> {j}'
    print(route_str)
else:
    print(f'No optimal solution found. Status: {m.status}')