import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']
row_ids = distance_df['Unnamed: 0'].str.strip().str.casefold().tolist()
col_ids = [col.strip().casefold() for col in distance_df.columns[1:]]
for loc in locations:
    if loc.strip().casefold() not in row_ids:
        raise ValueError(f"Row for location '{loc}' not found in DistanceMatrix.csv")
    if loc.strip().casefold() not in col_ids:
        raise ValueError(f"Column for location '{loc}' not found in DistanceMatrix.csv")
row_map = {distance_df.loc[i, 'Unnamed: 0'].strip(): i for i in range(len(distance_df))}
col_map = {col.strip(): j for (j, col) in enumerate(distance_df.columns)}
dists = {}
for i in locations:
    for j in locations:
        if i == j:
            continue
        row_idx = row_map[i]
        col_idx = col_map[j]
        val_str = distance_df.iloc[row_idx, col_idx]
        try:
            val = float(val_str)
        except Exception:
            raise ValueError(f"Distance from {i} to {j} is not a valid number: '{val_str}'")
        dists[i, j] = val
if len(dists) != len(locations) * (len(locations) - 1):
    raise ValueError('Missing distances for some location pairs.')
arc_keys = [(i, j) for i in locations for j in locations if i != j]
m = gp.Model('TSP_SmallCourier')
x_vars = m.addVars(arc_keys, vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([loc for loc in locations if loc != 'Depot'], lb=1, ub=3, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((dists[i, j] * x_vars[i, j] for (i, j) in arc_keys)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in locations if j != i)) == 1, name=f'out_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in locations if i != j)) == 1, name=f'in_{j}')
for i in [loc for loc in locations if loc != 'Depot']:
    for j in [loc for loc in locations if loc != 'Depot']:
        if i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + 3 * x_vars[i, j] <= 2, name=f'mtz_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for (i, j) in arc_keys:
        print(f'{x_vars[i, j].VarName} {x_vars[i, j].X}')
    for loc in [l for l in locations if l != 'Depot']:
        print(f'{u_vars[loc].VarName} {u_vars[loc].X}')
else:
    print(f'Solver status: {m.Status}')