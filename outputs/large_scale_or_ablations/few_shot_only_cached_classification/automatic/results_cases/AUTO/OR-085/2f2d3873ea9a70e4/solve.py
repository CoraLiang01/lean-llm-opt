import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',')
location_ids = [str(i) for i in range(1, 16)]
distance = {}
for idx, row in df.iterrows():
    i = str(row['Unnamed: 0'])
    distance[i] = {}
    for j in location_ids:
        val = row[j] if not pd.isnull(row[j]) else None
        distance[i][j] = val
for i in location_ids:
    for j in location_ids:
        if i == j:
            distance[i][j] = 0.0
        else:
            dij = distance[i][j]
            dji = distance[j][i]
            if dij is None and dji is not None:
                distance[i][j] = dji
            elif dji is None and dij is not None:
                distance[j][i] = dij
            elif dij is not None and dji is not None:
                if abs(dij - dji) > 1e-06:
                    raise ValueError(f'Distance matrix not symmetric at ({i},{j}): {dij} vs {dji}')
            elif dij is None and dji is None:
                raise ValueError(f'Missing distance for both ({i},{j}) and ({j},{i})')
m = gp.Model('TSP_15_locations')
x = m.addVars(location_ids, location_ids, vtype=gp.GRB.BINARY, name='')
for i in location_ids:
    m.addConstr(x[i, i] == 0, name=f'no_self_{i}')
for i in location_ids:
    m.addConstr(gp.quicksum((x[i, j] for j in location_ids if j != i)) == 1, name=f'leave_{i}')
for j in location_ids:
    m.addConstr(gp.quicksum((x[i, j] for i in location_ids if i != j)) == 1, name=f'arrive_{j}')
u = m.addVars(location_ids, vtype=gp.GRB.INTEGER, lb=1, ub=15, name='')
m.addConstr(u['1'] == 1, name='u_start')
for i in location_ids:
    if i == '1':
        continue
    for j in location_ids:
        if j == '1' or i == j:
            continue
        m.addConstr(u[i] - u[j] + 15 * x[i, j] <= 14, name=f'mtz_{i}_{j}')
m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for i in location_ids for j in location_ids)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    succ = {}
    for i in location_ids:
        for j in location_ids:
            if i != j and x[i, j].X > 0.5:
                succ[i] = j
                break
    tour = ['1']
    while len(tour) < len(location_ids) + 1:
        last = tour[-1]
        next_loc = succ[last]
        tour.append(next_loc)
        if next_loc == '1':
            break
    print('--- Optimal Tour ---')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')