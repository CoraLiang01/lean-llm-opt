import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
node_ids = [str(i) for i in range(1, 16)]
if not all((col in df.columns for col in node_ids)):
    missing = [col for col in node_ids if col not in df.columns]
    raise KeyError(f'Missing columns in CSV: {missing}')
if df.shape[0] < 15:
    raise ValueError('CSV does not contain all 15 required rows.')
distance = {i: {} for i in node_ids}
for (idx, row) in df.iterrows():
    i = str(row['Unnamed: 0']).strip()
    if i not in node_ids:
        continue
    for j in node_ids:
        val = row[j].strip()
        if val == '':
            continue
        try:
            distance[i][j] = float(val)
        except ValueError:
            raise ValueError(f"Non-numeric distance at ({i},{j}): '{val}'")
for i in node_ids:
    for j in node_ids:
        if i == j:
            distance[i][j] = 0.0
        elif j not in distance[i] or distance[i][j] == '':
            if i in distance[j] and distance[j][i] != '':
                distance[i][j] = float(distance[j][i])
            else:
                raise ValueError(f'Missing distance for ({i},{j}) and ({j},{i})')
m = gp.Model('TSP_15_Cities')
x_vars = m.addVars(node_ids, node_ids, vtype=gp.GRB.BINARY, name='')
for i in node_ids:
    m.addConstr(x_vars[i, i] == 0)
for i in node_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in node_ids if j != i)) == 1)
for j in node_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in node_ids if i != j)) == 1)
u_ids = [str(i) for i in range(2, 16)]
u_vars = m.addVars(u_ids, vtype=gp.GRB.INTEGER, lb=2, ub=15, name='')
for i in u_ids:
    for j in u_ids:
        if i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + 15 * x_vars[i, j] <= 14)
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for i in node_ids for j in node_ids)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    succ = {}
    for i in node_ids:
        for j in node_ids:
            if i != j and x_vars[i, j].X > 0.5:
                succ[i] = j
                break
    tour = ['1']
    while len(tour) < len(node_ids):
        next_city = succ[tour[-1]]
        tour.append(next_city)
    tour.append('1')
    print('--- Optimal Tour ---')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')