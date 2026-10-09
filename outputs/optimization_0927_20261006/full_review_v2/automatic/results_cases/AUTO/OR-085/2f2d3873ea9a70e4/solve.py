import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
location_ids = [str(i) for i in range(1, 16)]
n = len(location_ids)
row_id_col = 'Unnamed: 0'
row_ids = df[row_id_col].str.strip()
if not all((rid in location_ids for rid in row_ids)):
    raise ValueError('Row identifiers in CSV do not match expected location IDs.')
distance = {}
for i in location_ids:
    distance[i] = {}
    for j in location_ids:
        if i == j:
            distance[i][j] = 0.0
            continue
        row_mask = df[row_id_col].str.strip() == i
        if not row_mask.any():
            raise ValueError(f'Missing row for location {i}')
        val_ij = df.loc[row_mask, j].values[0].strip()
        if val_ij == '':
            row_mask2 = df[row_id_col].str.strip() == j
            if not row_mask2.any():
                raise ValueError(f'Missing row for location {j}')
            val_ji = df.loc[row_mask2, i].values[0].strip()
            if val_ji == '':
                raise ValueError(f'Missing distance between {i} and {j}')
            val = float(val_ji)
        else:
            val = float(val_ij)
        distance[i][j] = val
m = gp.Model('TSP_15_Cities')
x_vars = m.addVars([(i, j) for i in location_ids for j in location_ids if i != j], vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([i for i in location_ids if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for i in location_ids for j in location_ids if i != j)), gp.GRB.MINIMIZE)
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
        m.addConstr(u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2, name=f'subtour_{i}_{j}')
m.optimize()