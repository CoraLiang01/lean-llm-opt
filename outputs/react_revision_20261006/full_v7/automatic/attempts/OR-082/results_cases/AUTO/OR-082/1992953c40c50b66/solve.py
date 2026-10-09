import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']
row_ids = df['Unnamed: 0'].str.strip().str.casefold().tolist()
col_ids = [col.strip() for col in df.columns if col != 'Unnamed: 0']
col_ids_casefold = [col.casefold() for col in col_ids]
for loc in locations:
    if loc.casefold() not in row_ids:
        raise ValueError(f"Row for location '{loc}' not found in DistanceMatrix.csv")
    if loc.casefold() not in col_ids_casefold:
        raise ValueError(f"Column for location '{loc}' not found in DistanceMatrix.csv")
col_casefold_to_orig = {col.casefold(): col for col in col_ids}
distances = {}
for i in locations:
    row_idx = row_ids.index(i.casefold())
    row = df.iloc[row_idx]
    distances[i] = {}
    for j in locations:
        colname = col_casefold_to_orig[j.casefold()]
        val_str = row[colname]
        try:
            val = float(val_str)
        except Exception:
            raise ValueError(f"Non-numeric or missing distance from {i} to {j}: '{val_str}'")
        distances[i][j] = val
arc_keys = [(i, j) for i in locations for j in locations if i != j]
m = gp.Model('TSP_SmallCourier')
x_vars = m.addVars(arc_keys, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((distances[i][j] * x_vars[i, j] for (i, j) in arc_keys)), gp.GRB.MINIMIZE)
for k in ['A', 'B', 'C']:
    m.addConstr(gp.quicksum((x_vars[i, k] for i in locations if i != k)) == 1, name=f'arrive_{k}')
    m.addConstr(gp.quicksum((x_vars[k, j] for j in locations if j != k)) == 1, name=f'depart_{k}')
m.addConstr(gp.quicksum((x_vars['Depot', j] for j in locations if j != 'Depot')) == 1, name='depot_depart')
m.addConstr(gp.quicksum((x_vars[i, 'Depot'] for i in locations if i != 'Depot')) == 1, name='depot_arrive')
for i in locations:
    if (i, i) in x_vars:
        m.addConstr(x_vars[i, i] == 0, name=f'no_self_{i}')
n = len(locations)
u_vars = m.addVars([k for k in locations if k != 'Depot'], lb=1, ub=n - 1, vtype=gp.GRB.CONTINUOUS, name='')
for i in [k for k in locations if k != 'Depot']:
    for j in [k for k in locations if k != 'Depot']:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2, name=f'mtz_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (i, j) in arc_keys:
        print(f'x[{i},{j}] {x_vars[i, j].VarName} {x_vars[i, j].X}')
    for k in [k for k in locations if k != 'Depot']:
        print(f'u[{k}] {u_vars[k].VarName} {u_vars[k].X}')
else:
    print(f'Solver status: {m.status}')