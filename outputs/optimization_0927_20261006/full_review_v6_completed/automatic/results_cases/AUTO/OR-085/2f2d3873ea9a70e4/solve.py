import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = [str(i) for i in range(1, 16)]
n = len(locations)
if not all((col in df.columns for col in locations)):
    missing = [col for col in locations if col not in df.columns]
    raise ValueError(f'Missing columns in distance matrix: {missing}')
if not all((str(i) in df['Unnamed: 0'].values for i in range(1, n + 1))):
    missing = [str(i) for i in range(1, n + 1) if str(i) not in df['Unnamed: 0'].values]
    raise ValueError(f'Missing rows in distance matrix: {missing}')
distance = {}
for i in locations:
    distance[i] = {}
    row = df.loc[df['Unnamed: 0'] == i]
    if row.empty:
        raise ValueError(f'Row for location {i} not found in distance matrix.')
    for j in locations:
        if i == j:
            distance[i][j] = 0.0
        else:
            val_ij = row.iloc[0][j]
            val_ji = df.loc[df['Unnamed: 0'] == j].iloc[0][i]
            if val_ij != '':
                d_ij = float(val_ij)
            elif val_ji != '':
                d_ij = float(val_ji)
            else:
                raise ValueError(f'Missing distance between {i} and {j}.')
            if val_ij != '' and val_ji != '':
                d_ji = float(val_ji)
                if abs(d_ij - d_ji) > 1e-06:
                    raise ValueError(f'Distance matrix not symmetric at ({i},{j}): {d_ij} vs {d_ji}')
            distance[i][j] = d_ij
m = gp.Model('TSP_15_Cities')
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([i for i in locations if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in locations if j != i)) == 1, name=f'leave_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in locations if i != j)) == 1, name=f'enter_{j}')
for i in locations:
    if i == '1':
        continue
    for j in locations:
        if j == '1' or i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + n * x_vars[i, j] <= n - 1, name=f'subtour_{i}_{j}')
m.optimize()