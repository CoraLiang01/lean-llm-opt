import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
locations = [str(i) for i in range(1, 16)]
n = len(locations)
if not set(locations).issubset(df.columns):
    raise ValueError('Missing required columns in distance matrix')
if not set(locations).issubset(df['Unnamed: 0'].str.strip()):
    raise ValueError('Missing required rows in distance matrix')
row_idx_by_loc = {row_id: idx for idx, row_id in enumerate(df['Unnamed: 0'].str.strip())}
distance = {}
for i in locations:
    distance[i] = {}
    for j in locations:
        if i == j:
            distance[i][j] = 0.0
        else:
            row_i = row_idx_by_loc[i]
            val_ij = df.at[row_i, j].strip()
            if val_ij == '':
                row_j = row_idx_by_loc[j]
                val_ji = df.at[row_j, i].strip()
                if val_ji == '':
                    raise ValueError(f'Missing distance between {i} and {j}')
                val = float(val_ji)
            else:
                val = float(val_ij)
            distance[i][j] = val
m = Model()
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=GRB.BINARY, name='')
u_vars = m.addVars([i for i in locations if i != '1'], vtype=GRB.INTEGER, lb=2, ub=n, name='')
m.setObjective(quicksum((distance[i][j] * x_vars[i, j] for i in locations for j in locations if i != j)), GRB.MINIMIZE)
for i in locations:
    m.addConstr(quicksum((x_vars[i, j] for j in locations if j != i)) == 1)
for j in locations:
    m.addConstr(quicksum((x_vars[i, j] for i in locations if i != j)) == 1)
for i in locations:
    if i == '1':
        continue
    for j in locations:
        if j == '1' or i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2)
m.optimize()