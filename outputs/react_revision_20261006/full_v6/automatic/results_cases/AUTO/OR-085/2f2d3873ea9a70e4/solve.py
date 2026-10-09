import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
location_ids = [str(i) for i in range(1, 16)]
row_ids = df['Unnamed: 0'].astype(str).tolist()
col_ids = [c for c in df.columns if c != 'Unnamed: 0']
if set(location_ids) != set(row_ids) or set(location_ids) != set(col_ids):
    raise ValueError('Mismatch between expected location IDs and those found in the CSV.')
dists = {}
for i in location_ids:
    dists[i] = {}
    row = df[df['Unnamed: 0'] == i]
    if row.empty:
        raise ValueError(f'Missing row for location {i} in distance matrix.')
    for j in location_ids:
        if i == j:
            dists[i][j] = 0.0
        else:
            val_ij = row[j].values[0].strip()
            if val_ij == '':
                row_j = df[df['Unnamed: 0'] == j]
                if row_j.empty:
                    raise ValueError(f'Missing row for location {j} in distance matrix.')
                val_ji = row_j[i].values[0].strip()
                if val_ji == '':
                    raise ValueError(f'Missing both d({i},{j}) and d({j},{i}) in distance matrix.')
                dists[i][j] = float(val_ji)
            else:
                dists[i][j] = float(val_ij)
            row_j = df[df['Unnamed: 0'] == j]
            if not row_j.empty:
                val_ji = row_j[i].values[0].strip()
                if val_ji != '':
                    if abs(float(val_ji) - dists[i][j]) > 1e-06:
                        raise ValueError(f'Distance matrix not symmetric at ({i},{j}) and ({j},{i}).')
arc_keys = [(i, j) for i in location_ids for j in location_ids if i != j]

def solve_tsp():
    m = gp.Model('TSP15')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(arc_keys, vtype=gp.GRB.BINARY, name='')
    u_vars = {}
    for i in location_ids:
        if i == '1':
            u_vars[i] = m.addVar(lb=1, ub=1, vtype=gp.GRB.INTEGER, name=f'u_{i}')
        else:
            u_vars[i] = m.addVar(lb=2, ub=15, vtype=gp.GRB.INTEGER, name=f'u_{i}')
    m.setObjective(gp.quicksum((dists[i][j] * x_vars[i, j] for (i, j) in arc_keys)), gp.GRB.MINIMIZE)
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
            m.addConstr(u_vars[i] - u_vars[j] + 15 * x_vars[i, j] <= 14, name=f'subtour_{i}_{j}')
    for i in location_ids:
        if (i, i) in x_vars:
            m.addConstr(x_vars[i, i] == 0, name=f'noloop_{i}')
    m.optimize()
    return m
m = solve_tsp()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')