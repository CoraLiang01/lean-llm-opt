import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = [str(i) for i in range(1, 16)]
n = len(locations)
distance = {i: {} for i in locations}
for row_idx, row in df.iterrows():
    i = str(row['Unnamed: 0']).strip()
    for j in locations:
        if i == j:
            continue
        val = row[j].strip()
        if val == '':
            continue
        try:
            d = float(val)
        except Exception as e:
            raise ValueError(f"Invalid distance value at ({i},{j}): '{val}'") from e
        distance[i][j] = d
for i in locations:
    for j in locations:
        if i == j:
            continue
        if j not in distance[i]:
            if i in distance[j]:
                distance[i][j] = distance[j][i]
            else:
                raise ValueError(f'Missing distance for both ({i},{j}) and ({j},{i})')
for i in locations:
    for j in locations:
        if i == j:
            continue
        if j not in distance[i]:
            raise ValueError(f'Missing distance entry for ({i},{j}) after symmetry fill')
        if i not in distance[j]:
            raise ValueError(f'Missing distance entry for ({j},{i}) after symmetry fill')
        if not np.isclose(distance[i][j], distance[j][i], atol=1e-06):
            raise ValueError(f'Distance matrix not symmetric at ({i},{j}): {distance[i][j]} vs {distance[j][i]}')
m = gp.Model('TSP_15')
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([i for i in locations if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in locations if j != i)) == 1)
for j in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in locations if i != j)) == 1)
for i in locations:
    if i == '1':
        continue
    for j in locations:
        if j == '1' or i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + n * x_vars[i, j] <= n - 1)
m.optimize()