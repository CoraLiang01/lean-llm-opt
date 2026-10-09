import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']
row_ids = df['Unnamed: 0'].tolist()
col_ids = list(df.columns)
col_locs = [c for c in col_ids if c != 'Unnamed: 0']
for loc in locations:
    if loc not in row_ids:
        raise KeyError(f"Location '{loc}' not found in DistanceMatrix.csv rows.")
    if loc not in col_locs:
        raise KeyError(f"Location '{loc}' not found in DistanceMatrix.csv columns.")
distance = {}
for i in locations:
    row = df.loc[df['Unnamed: 0'] == i]
    if row.empty:
        raise KeyError(f"Row for location '{i}' not found in DistanceMatrix.csv.")
    for j in locations:
        val_str = row.iloc[0][j]
        try:
            distance[i, j] = float(val_str)
        except Exception:
            raise ValueError(f"Invalid distance value for ({i},{j}): '{val_str}'")
m = gp.Model('TSP_4node')
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
n = len(locations)
u_locs = [loc for loc in locations if loc != 'Depot']
u_vars = m.addVars(u_locs, vtype=gp.GRB.CONTINUOUS, lb=1, ub=n - 1, name='')
m.setObjective(gp.quicksum((distance[i, j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in locations if j != i)) == 1, name=f'out_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in locations if i != j)) == 1, name=f'in_{j}')
for i in u_locs:
    for j in u_locs:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + n * x_vars[i, j] <= n - 1, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    tour = []
    current = 'Depot'
    visited = set()
    for _ in range(n):
        for j in locations:
            if j != current and x_vars[current, j].X > 0.5:
                tour.append((current, j))
                visited.add(current)
                current = j
                break
    print('Optimal route (in order):')
    route = ['Depot']
    next_node = 'Depot'
    for _ in range(n):
        for j in locations:
            if j != next_node and x_vars[next_node, j].X > 0.5:
                route.append(j)
                next_node = j
                break
    print(' -> '.join(route + ['Depot']))
else:
    print(f'No optimal solution found. Status: {m.status}')