import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
location_ids = [col for col in df.columns if col != 'Unnamed: 0']
row_ids = df['Unnamed: 0'].tolist()
if set(location_ids) != set(row_ids):
    raise ValueError('Mismatch between row and column location identifiers in distance matrix.')
location_ids = sorted(location_ids, key=lambda x: int(x))
row_ids = sorted(row_ids, key=lambda x: int(x))
N = len(location_ids)
if N != 15:
    raise ValueError(f'Expected 15 locations, found {N}.')
distance = {}
for i in location_ids:
    for j in location_ids:
        if i == j:
            continue
        row = df[df['Unnamed: 0'] == i]
        if row.empty:
            raise ValueError(f'Row for location {i} not found in distance matrix.')
        val = row.iloc[0][j]
        try:
            dval = float(val)
        except Exception:
            raise ValueError(f"Non-numeric distance from {i} to {j}: '{val}'")
        distance[i, j] = dval
x_keys = [(i, j) for i in location_ids for j in location_ids if i != j]
u_keys = [i for i in location_ids if i != '1']

def solve_tsp():
    m = gp.Model('TSP')
    x_vars = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
    u_vars = m.addVars(u_keys, vtype=gp.GRB.INTEGER, lb=2, ub=N, name='')
    m.setObjective(gp.quicksum((distance[i, j] * x_vars[i, j] for (i, j) in x_keys)), gp.GRB.MINIMIZE)
    for i in location_ids:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in location_ids if j != i)) == 1, name='out_' + i)
    for j in location_ids:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in location_ids if i != j)) == 1, name='in_' + j)
    for i in u_keys:
        for j in u_keys:
            if i == j:
                continue
            m.addConstr(u_vars[i] - u_vars[j] + N * x_vars[i, j] <= N - 1, name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_tsp()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')