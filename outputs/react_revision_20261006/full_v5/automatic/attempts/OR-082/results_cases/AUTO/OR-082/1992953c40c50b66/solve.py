import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',')
locations = ['Depot', 'A', 'B', 'C']
row_labels = df['Unnamed: 0'].astype(str).str.strip().str.casefold().tolist()
col_labels = [col.strip().casefold() for col in df.columns]
for loc in locations:
    loc_cf = loc.casefold()
    if loc_cf not in row_labels:
        raise ValueError(f"Row for location '{loc}' not found in DistanceMatrix.csv")
    if loc_cf not in col_labels:
        raise ValueError(f"Column for location '{loc}' not found in DistanceMatrix.csv")
row_idx_map = {name.strip(): idx for (idx, name) in enumerate(df['Unnamed: 0'])}
distances = {}
for i in locations:
    i_idx = row_idx_map[i]
    distances[i] = {}
    for j in locations:
        distances[i][j] = float(df.at[i_idx, j])
x_keys = [(i, j) for i in locations for j in locations if i != j]
m = gp.Model('TSP_SmallCourier')
x = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((distances[i][j] * x[i, j] for (i, j) in x_keys)), gp.GRB.MINIMIZE)
for k in locations:
    m.addConstr(gp.quicksum((x[k, j] for j in locations if j != k)) == 1, name=f'out_{k}')
    m.addConstr(gp.quicksum((x[i, k] for i in locations if i != k)) == 1, name=f'in_{k}')
n = len(locations)
customer_locs = [loc for loc in locations if loc != 'Depot']
u = m.addVars(customer_locs, lb=1, ub=n - 1, vtype=gp.GRB.CONTINUOUS, name='')
for i in customer_locs:
    for j in customer_locs:
        if i != j:
            m.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2, name=f'mtz_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (i, j) in x_keys:
        print(f'x[{i},{j}] {x[i, j].VarName} {x[i, j].X}')
    for i in customer_locs:
        print(f'u[{i}] {u[i].VarName} {u[i].X}')
else:
    print(f'Solver status: {m.status}')