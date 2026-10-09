import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
locations = [str(i) for i in range(1, 16)]
n = len(locations)
distance = {i: {} for i in locations}
for (row_idx, row) in df.iterrows():
    i = str(row['Unnamed: 0']).strip()
    if i not in locations:
        continue
    for j in locations:
        val = row[j].strip()
        if val == '':
            continue
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
                distance[i][j] = dij
                distance[j][i] = dij
            elif dij is not None:
                distance[j][i] = dij
            elif dji is not None:
                distance[i][j] = dji
            else:
                raise ValueError(f'Missing distance for ({i},{j}) and ({j},{i})')
x_vars = {}
model = Model()
for i in locations:
    for j in locations:
        if i != j:
            x_vars[i, j] = model.addVar(vtype=GRB.BINARY, name='x_%s_%s' % (i, j))
u_vars = {}
for i in locations:
    if i != '1':
        u_vars[i] = model.addVar(vtype=GRB.INTEGER, lb=2, ub=n, name='u_%s' % i)
model.update()
for i in locations:
    model.addConstr(quicksum((x_vars[i, j] for j in locations if j != i)) == 1)
for j in locations:
    model.addConstr(quicksum((x_vars[i, j] for i in locations if i != j)) == 1)
for i in locations:
    if i == '1':
        continue
    for j in locations:
        if j == '1' or i == j:
            continue
        model.addConstr(u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2)
model.setObjective(quicksum((distance[i][j] * x_vars[i, j] for i in locations for j in locations if i != j)), GRB.MINIMIZE)
m = model
m.optimize()