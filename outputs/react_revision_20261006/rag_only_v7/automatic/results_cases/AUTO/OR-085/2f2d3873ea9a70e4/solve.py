import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
locations = [str(i) for i in range(1, 16)]
distance = {i: {} for i in locations}
for (row_idx, row) in df.iterrows():
    i = str(row['Unnamed: 0']).strip()
    if i not in locations:
        continue
    for j in locations:
        val = row[j].strip() if j in row else ''
        if val != '':
            distance[i][j] = float(val)
for i in locations:
    for j in locations:
        if i == j:
            distance[i][j] = 0.0
        else:
            dij = distance[i].get(j, None)
            dji = distance[j].get(i, None)
            if dij is not None and dji is not None:
                if abs(dij - dji) > 1e-06:
                    raise ValueError(f'Distance matrix not symmetric at ({i},{j}): {dij} vs {dji}')
            elif dij is not None:
                distance[j][i] = dij
            elif dji is not None:
                distance[i][j] = dji
            else:
                raise ValueError(f'Missing distance for ({i},{j}) and ({j},{i})')
x_keys = [(i, j) for i in locations for j in locations if i != j]
u_locs = [i for i in locations if i != '1']
m = gp.Model('TSP')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(x_keys, vtype=GRB.BINARY, name='')
u_vars = m.addVars(u_locs, vtype=GRB.INTEGER, lb=2, ub=15, name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for (i, j) in x_keys)), GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in locations if j != i)) == 1, name=f'leave_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in locations if i != j)) == 1, name=f'arrive_{j}')
for i in locations:
    if (i, i) in x_keys:
        m.addConstr(x_vars[i, i] == 0, name=f'noloop_{i}')
n = 15
for i in u_locs:
    for j in u_locs:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + n * x_vars[i, j] <= n - 1, name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')