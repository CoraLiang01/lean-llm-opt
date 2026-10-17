import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = [str(i) for i in range(1, 16)]
n = len(locations)
if not all((col in df.columns for col in locations)):
    raise ValueError('Not all required location columns are present in the CSV.')
if not all((str(i) in df['Unnamed: 0'].values for i in range(1, n + 1))):
    raise ValueError('Not all required location rows are present in the CSV.')
distance = {}
for i in locations:
    distance[i] = {}
    for j in locations:
        if i == j:
            distance[i][j] = 0.0
            continue
        row_i = df[df['Unnamed: 0'] == i]
        val_ij = row_i[j].values[0] if not row_i.empty else ''
        row_j = df[df['Unnamed: 0'] == j]
        val_ji = row_j[i].values[0] if not row_j.empty else ''
        v = None
        if val_ij.strip() != '':
            v = float(val_ij)
        elif val_ji.strip() != '':
            v = float(val_ji)
        else:
            raise ValueError(f'Missing distance between {i} and {j} in both directions.')
        distance[i][j] = v
N = locations
m = gp.Model('TSP_15_Cities')
x_vars = m.addVars([(i, j) for i in N for j in N if i != j], vtype=gp.GRB.BINARY, name='')
u_vars = m.addVars([i for i in N if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for i in N for j in N if i != j)), gp.GRB.MINIMIZE)
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
        m.addConstr(u_vars[i] - u_vars[j] + n * x_vars[i, j] <= n - 1, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    succ = {}
    for i in N:
        for j in N:
            if i != j and x_vars[i, j].X > 0.5:
                succ[i] = j
    tour = ['1']
    while len(tour) < n + 1:
        last = tour[-1]
        next_city = succ[last]
        tour.append(next_city)
        if next_city == '1':
            break
    print('Optimal tour (visiting order):')
    print(' -> '.join(tour))
    print('Tour length:', m.objVal)
else:
    print(f'No optimal solution found. Status: {m.status}')