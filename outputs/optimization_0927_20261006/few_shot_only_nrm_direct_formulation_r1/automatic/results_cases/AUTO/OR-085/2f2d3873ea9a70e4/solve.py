import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
locations = [str(i) for i in range(1, 16)]
n = len(locations)
if not all((col in df.columns for col in locations)):
    raise ValueError('Not all required columns are present in the CSV file.')
if not all((str(i) in df['Unnamed: 0'].values for i in range(1, n + 1))):
    raise ValueError('Not all required rows are present in the CSV file.')
df = df.set_index('Unnamed: 0')
distance_matrix = {}
for i in locations:
    distance_matrix[i] = {}
    for j in locations:
        val = df.at[i, j]
        try:
            distance_matrix[i][j] = float(val)
        except Exception:
            raise ValueError(f"Invalid or missing distance value at ({i},{j}): '{val}'")
for i in locations:
    for j in locations:
        if i != j:
            if abs(distance_matrix[i][j] - distance_matrix[j][i]) > 1e-06:
                raise ValueError(f'Distance matrix is not symmetric at ({i},{j})')
m = gp.Model('TSP_15_Cities')
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([i for i in locations if i != '1'], vtype=gp.GRB.CONTINUOUS, lb=1, ub=n - 1, name='')
m.setObjective(gp.quicksum((distance_matrix[i][j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
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
        m.addConstr(u_vars[i] - u_vars[j] + n * x_vars[i, j] <= n - 1, name=f'mtz_{i}_{j}')
m.optimize()