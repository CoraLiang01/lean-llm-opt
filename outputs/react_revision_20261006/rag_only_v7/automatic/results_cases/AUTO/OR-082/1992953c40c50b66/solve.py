import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv', sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']
row_ids = distance_df['Unnamed: 0'].str.strip().tolist()
col_ids = [col.strip() for col in distance_df.columns if col != 'Unnamed: 0']
missing_rows = set(locations) - set(row_ids)
missing_cols = set(locations) - set(col_ids)
if missing_rows:
    raise ValueError(f'Missing required rows in distance matrix: {missing_rows}')
if missing_cols:
    raise ValueError(f'Missing required columns in distance matrix: {missing_cols}')
distance = {}
for i in locations:
    row = distance_df.loc[distance_df['Unnamed: 0'].str.strip() == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in distance matrix.")
    row = row.iloc[0]
    distance[i] = {}
    for j in locations:
        val = row[j]
        try:
            distance[i][j] = float(val)
        except Exception:
            raise ValueError(f"Invalid or missing distance from {i} to {j}: '{val}'")
x_keys = [(i, j) for i in locations for j in locations if i != j]
u_locs = [loc for loc in locations if loc != 'Depot']
m = gp.Model('TSP')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(x_keys, vtype=GRB.BINARY, name='')
n = len(locations)
u_vars = m.addVars(u_locs, vtype=GRB.INTEGER, lb=1, ub=n - 1, name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for (i, j) in x_keys)), GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in locations if j != i)) == 1, name='')
for j in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in locations if i != j)) == 1, name='')
for i in u_locs:
    for j in u_locs:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2, name='')
for i in locations:
    if (i, i) in x_vars:
        m.addConstr(x_vars[i, i] == 0, name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')