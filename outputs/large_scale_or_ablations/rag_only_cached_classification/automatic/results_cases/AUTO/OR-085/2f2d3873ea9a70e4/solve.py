import pandas as pd
import numpy as np
import re
import math
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',')
location_ids = [str(i) for i in range(1, 16)]
row_ids = df['Unnamed: 0'].astype(str).tolist()
if set(row_ids) != set(location_ids):
    raise ValueError(f'Row IDs in CSV do not match required locations: {row_ids} vs {location_ids}')
if set(df.columns[1:]) != set(location_ids):
    raise ValueError(f'Column IDs in CSV do not match required locations: {df.columns[1:]} vs {location_ids}')
distance = {i: {} for i in location_ids}
for idx, row in df.iterrows():
    i = str(row['Unnamed: 0'])
    for j in location_ids:
        val = row[j]
        if pd.isna(val):
            continue
        distance[i][j] = float(val)
for i in location_ids:
    for j in location_ids:
        if i == j:
            distance[i][j] = 0.0
        else:
            dij = distance[i].get(j, None)
            dji = distance[j].get(i, None)
            if dij is not None and dji is not None:
                if not math.isclose(dij, dji, rel_tol=1e-06, abs_tol=1e-06):
                    raise ValueError(f'Distance matrix not symmetric at ({i},{j}): {dij} vs ({j},{i}): {dji}')
            elif dij is not None:
                distance[j][i] = dij
            elif dji is not None:
                distance[i][j] = dji
            else:
                raise ValueError(f'Missing distance for both ({i},{j}) and ({j},{i})')
N = location_ids
n = len(N)
m = Model('TSP')
x = m.addVars(N, N, vtype=GRB.BINARY, name='')
u = {}
for i in N:
    if i != '1':
        u[i] = m.addVar(vtype=GRB.INTEGER, lb=2, ub=n, name=f'u_{i}')
m.setObjective(quicksum((distance[i][j] * x[i, j] for i in N for j in N if i != j)), GRB.MINIMIZE)
for i in N:
    m.addConstr(quicksum((x[i, j] for j in N if j != i)) == 1)
for j in N:
    m.addConstr(quicksum((x[i, j] for i in N if i != j)) == 1)
for i in N:
    if i == '1':
        continue
    for j in N:
        if j == '1' or i == j:
            continue
        m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)
m.optimize()