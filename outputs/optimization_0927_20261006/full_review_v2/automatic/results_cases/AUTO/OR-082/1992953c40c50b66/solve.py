import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)

def norm(s):
    return s.strip().casefold()
locations = ['Depot', 'A', 'B', 'C']
locations_norm = [norm(l) for l in locations]
row_id_map = {}
for (idx, val) in df['Unnamed: 0'].items():
    row_id_map[norm(val)] = val
col_id_map = {}
for col in df.columns:
    col_id_map[norm(col)] = col
for l in locations_norm:
    if l not in row_id_map:
        raise KeyError(f"Location '{l}' not found in DistanceMatrix.csv rows.")
    if l not in col_id_map:
        raise KeyError(f"Location '{l}' not found in DistanceMatrix.csv columns.")
distance = {}
for i_norm in locations_norm:
    i_row = row_id_map[i_norm]
    distance[i_row] = {}
    row = df.loc[df['Unnamed: 0'] == i_row].iloc[0]
    for j_norm in locations_norm:
        j_col = col_id_map[j_norm]
        val = row[j_col]
        try:
            distance[i_row][j_col] = float(val)
        except Exception:
            raise ValueError(f"Invalid distance value from '{i_row}' to '{j_col}': {val}")
m = gp.Model('TSP_3customer')
x_vars = m.addVars(locations, locations, vtype=gp.GRB.BINARY, name='')
n = len(locations)
u_locs = [l for l in locations if l != 'Depot']
u_vars = m.addVars(u_locs, vtype=gp.GRB.CONTINUOUS, lb=1, ub=n - 1, name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in locations if j != i)) == 1, name=f'depart_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in locations if i != j)) == 1, name=f'arrive_{j}')
for i in u_locs:
    for j in u_locs:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2, name=f'mtz_{i}_{j}')
for i in locations:
    m.addConstr(x_vars[i, i] == 0, name=f'no_self_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    next_loc = {}
    for i in locations:
        for j in locations:
            if i != j and x_vars[i, j].X > 0.5:
                next_loc[i] = j
    tour = ['Depot']
    while True:
        last = tour[-1]
        if last not in next_loc:
            break
        nxt = next_loc[last]
        tour.append(nxt)
        if nxt == 'Depot':
            break
        if len(tour) > n + 1:
            print('Warning: Tour reconstruction exceeded expected length.')
            break
    print('Optimal route:')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')