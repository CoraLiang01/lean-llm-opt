import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',')
locations = ['Depot', 'A', 'B', 'C']

def normalize_id(x):
    return str(x).strip()
row_labels = df['Unnamed: 0'].apply(normalize_id).tolist()
col_labels = [normalize_id(col) for col in df.columns[1:]]
for loc in locations:
    if loc not in row_labels:
        raise KeyError(f"Location '{loc}' not found in row labels of DistanceMatrix.csv")
    if loc not in col_labels:
        raise KeyError(f"Location '{loc}' not found in column labels of DistanceMatrix.csv")
distance = {}
for i in locations:
    row_idx = row_labels.index(i)
    for j in locations:
        col_idx = col_labels.index(j)
        val = df.iloc[row_idx, col_idx + 1]
        distance[i, j] = float(val)
m = gp.Model('TSP_4node')
x = m.addVars(locations, locations, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((distance[i, j] * x[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1, name=f'leave_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1, name=f'enter_{j}')
u = m.addVars(locations, vtype=gp.GRB.CONTINUOUS, lb=0, ub=len(locations) - 1, name='')
m.addConstr(u['Depot'] == 0, name='u_depot')
for i in locations:
    for j in locations:
        if i != j and i != 'Depot' and (j != 'Depot'):
            m.addConstr(u[i] - u[j] + (len(locations) - 1) * x[i, j] <= len(locations) - 2, name=f'subtour_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} km')
    tour = []
    current = 'Depot'
    visited = set()
    for _ in range(len(locations)):
        for j in locations:
            if current != j and x[current, j].X > 0.5:
                tour.append((current, j))
                visited.add(current)
                current = j
                break
    print('\n--- Optimal Route ---')
    route_str = tour[0][0]
    for arc in tour:
        route_str += f' -> {arc[1]}'
    print(route_str)
else:
    print(f'No optimal solution found. Status: {m.status}')