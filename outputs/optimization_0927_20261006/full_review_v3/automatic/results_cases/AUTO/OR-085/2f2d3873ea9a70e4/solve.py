import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
location_ids = [str(i) for i in range(1, 16)]
if not all((col in df.columns for col in location_ids)):
    missing = [col for col in location_ids if col not in df.columns]
    raise ValueError(f'Missing columns in distance matrix: {missing}')
if df.shape[0] != 15:
    raise ValueError(f'Expected 15 rows in distance matrix, got {df.shape[0]}')
distance = {i: {} for i in location_ids}
for (idx, row) in df.iterrows():
    i = str(row['Unnamed: 0']).strip()
    if i not in location_ids:
        raise ValueError(f'Unexpected row identifier: {i}')
    for j in location_ids:
        val = row[j].strip()
        if val == '':
            continue
        try:
            distance[i][j] = float(val)
        except Exception as e:
            raise ValueError(f'Invalid distance value at ({i},{j}): {val}') from e
for i in location_ids:
    for j in location_ids:
        if i == j:
            distance[i][j] = 0.0
        elif j not in distance[i]:
            if i in distance[j]:
                distance[i][j] = distance[j][i]
            else:
                raise ValueError(f'Missing distance for ({i},{j}) and ({j},{i})')
N = location_ids
m = gp.Model('TSP_15_Cities')
x_vars = m.addVars([(i, j) for i in N for j in N if i != j], vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([i for i in N if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=15, name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for i in N for j in N if i != j)), gp.GRB.MINIMIZE)
for i in N:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in N if j != i)) == 1, name=f'leave_{i}')
for j in N:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in N if i != j)) == 1, name=f'enter_{j}')
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
    succ = {}
    for i in N:
        for j in N:
            if i != j and x_vars[i, j].X > 0.5:
                succ[i] = j
    tour = ['1']
    while len(tour) < len(N):
        last = tour[-1]
        next_city = succ[last]
        tour.append(next_city)
    tour.append('1')
    print('Optimal tour:')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')