import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
location_ids = []
for val in df['Unnamed: 0']:
    val = val.strip()
    if val != '':
        location_ids.append(val)
N = [str(i) for i in range(1, 16)]
if set(location_ids) != set(N):
    raise ValueError(f'CSV row labels do not match expected locations 1..15. Found: {location_ids}')
d = {i: {} for i in N}
col_labels = list(df.columns)
col_locations = [col.strip() for col in col_labels[1:]]
if col_locations != N:
    raise ValueError(f'CSV column labels do not match expected locations 1..15. Found: {col_locations}')
for row_idx, row in df.iterrows():
    i = row['Unnamed: 0'].strip()
    for j_idx, j in enumerate(col_locations):
        cell = row[j].strip()
        if cell != '':
            try:
                val = float(cell)
            except Exception as e:
                raise ValueError(f"Non-numeric distance at ({i},{j}): '{cell}'")
            d[i][j] = val
for i in N:
    for j in N:
        if i == j:
            d[i][j] = 0.0
        elif j not in d[i]:
            if i in d[j]:
                d[i][j] = d[j][i]
            else:
                raise ValueError(f'Missing distance for ({i},{j}) and ({j},{i}) in CSV.')
m = gp.Model('TSP_15_Cities')
x_vars = m.addVars([(i, j) for i in N for j in N if i != j], vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([i for i in N if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=15, name='')
m.setObjective(gp.quicksum((d[i][j] * x_vars[i, j] for i in N for j in N if i != j)), gp.GRB.MINIMIZE)
for i in N:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in N if j != i)) == 1, name=f'depart_{i}')
for j in N:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in N if i != j)) == 1, name=f'arrive_{j}')
for i in N:
    if i == '1':
        continue
    for j in N:
        if j == '1' or i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + 15 * x_vars[i, j] <= 14, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    tour = {}
    for i in N:
        for j in N:
            if i != j and x_vars[i, j].X > 0.5:
                tour[i] = j
    path = ['1']
    while len(path) < len(N) + 1:
        last = path[-1]
        next_node = tour[last]
        path.append(next_node)
        if next_node == '1':
            break
    print('Optimal tour (visiting order):')
    print(' -> '.join(path))
else:
    print(f'No optimal solution found. Status: {m.status}')