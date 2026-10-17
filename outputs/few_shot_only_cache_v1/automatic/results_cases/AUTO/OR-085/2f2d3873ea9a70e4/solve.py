import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
locations = [str(i) for i in range(1, 16)]
n = len(locations)
d = {i: {} for i in locations}
for row in df.itertuples(index=False):
    i = getattr(row, 'Unnamed: 0')
    for j in locations:
        if i == j:
            d[i][j] = 0.0
        else:
            val = getattr(row, j)
            if val.strip() != '':
                d[i][j] = float(val)
            else:
                d[i][j] = None
for i in locations:
    for j in locations:
        if i == j:
            continue
        if d[i][j] is None and d[j][i] is not None:
            d[i][j] = d[j][i]
        elif d[i][j] is not None and d[j][i] is None:
            d[j][i] = d[i][j]
        elif d[i][j] is None and d[j][i] is None:
            raise ValueError(f'Missing distance between {i} and {j} in both directions.')
m = gp.Model('TSP_15')
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([i for i in locations if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
m.setObjective(gp.quicksum((d[i][j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
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