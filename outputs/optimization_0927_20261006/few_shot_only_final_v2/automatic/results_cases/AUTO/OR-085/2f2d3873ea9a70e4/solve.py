import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
locations = [str(i) for i in range(1, 16)]
n = len(locations)
if df.shape[0] != n or len([c for c in df.columns if c in locations]) != n:
    raise ValueError('CSV does not contain the expected 15x15 distance matrix with locations 1..15.')
distance = {}
for i in locations:
    for j in locations:
        if i == j:
            distance[i, j] = 0.0
        else:
            try:
                val = df.loc[df['Unnamed: 0'] == i, j].values
                if len(val) == 0 or val[0] == '':
                    val = df.loc[df['Unnamed: 0'] == j, i].values
                if len(val) == 0 or val[0] == '':
                    raise ValueError(f'Missing distance between {i} and {j}')
                distance[i, j] = float(val[0])
            except Exception as e:
                raise ValueError(f'Error reading distance between {i} and {j}: {e}')
m = gp.Model('TSP_15_Cities')
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([i for i in locations if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
m.setObjective(gp.quicksum((distance[i, j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in locations if j != i)) == 1, name=f'out_{i}')
    m.addConstr(gp.quicksum((x_vars[j, i] for j in locations if j != i)) == 1, name=f'in_{i}')
for i in locations:
    if i == '1':
        continue
    for j in locations:
        if j == '1' or i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + n * x_vars[i, j] <= n - 1, name=f'mtz_{i}_{j}')
m.optimize()