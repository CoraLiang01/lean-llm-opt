import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
location_ids = [str(i) for i in range(1, 16)]
dists = {}
for i in location_ids:
    dists[i] = {}
    row = df[df['Unnamed: 0'].str.strip() == i]
    if row.empty:
        raise ValueError(f'Missing row for location {i} in distance matrix.')
    for j in location_ids:
        if i == j:
            dists[i][j] = 0.0
            continue
        val_ij = row.iloc[0][j].strip()
        if val_ij == '':
            row_j = df[df['Unnamed: 0'].str.strip() == j]
            if row_j.empty:
                raise ValueError(f'Missing row for location {j} in distance matrix.')
            val_ji = row_j.iloc[0][i].strip()
            if val_ji == '':
                raise ValueError(f'Missing distance for ({i},{j}) and ({j},{i}) in matrix.')
            val = float(val_ji)
        else:
            val = float(val_ij)
        dists[i][j] = val
for i in location_ids:
    for j in location_ids:
        if i != j:
            if not np.isclose(dists[i][j], dists[j][i]):
                raise ValueError(f'Distance matrix not symmetric at ({i},{j}): {dists[i][j]} vs {dists[j][i]}')
arc_keys = [(i, j) for i in location_ids for j in location_ids if i != j]
u_keys = [i for i in location_ids if i != '1']

def solve_tsp():
    m = gp.Model('TSP15')
    x_vars = m.addVars(arc_keys, vtype=gp.GRB.BINARY, name='')
    u_vars = m.addVars(u_keys, vtype=gp.GRB.INTEGER, lb=2, ub=15, name='')
    m.setObjective(gp.quicksum((dists[i][j] * x_vars[i, j] for (i, j) in arc_keys)), gp.GRB.MINIMIZE)
    for i in location_ids:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in location_ids if j != i)) == 1, name='leave_%s' % i)
    for j in location_ids:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in location_ids if i != j)) == 1, name='enter_%s' % j)
    n = len(location_ids)
    for i in u_keys:
        for j in u_keys:
            if i != j:
                m.addConstr(u_vars[i] - u_vars[j] + n * x_vars[i, j] <= n - 1, name='mtz_%s_%s' % (i, j))
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