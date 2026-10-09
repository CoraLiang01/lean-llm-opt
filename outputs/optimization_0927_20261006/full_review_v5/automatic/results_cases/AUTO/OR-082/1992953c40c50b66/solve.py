import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']

def norm(s):
    return s.strip().casefold()
row_labels = [str(r) for r in df['Unnamed: 0']]
row_norm_map = {norm(r): r for r in row_labels}
col_labels = list(df.columns)
col_norm_map = {norm(c): c for c in col_labels}
for loc in locations:
    if norm(loc) not in row_norm_map:
        raise KeyError(f"Location '{loc}' not found as a row in DistanceMatrix.csv")
    if norm(loc) not in col_norm_map:
        raise KeyError(f"Location '{loc}' not found as a column in DistanceMatrix.csv")
distance = {}
for i in locations:
    i_row = row_norm_map[norm(i)]
    distance[i] = {}
    for j in locations:
        j_col = col_norm_map[norm(j)]
        val = df.loc[df['Unnamed: 0'] == i_row, j_col]
        if val.empty:
            raise KeyError(f"Missing distance from '{i}' to '{j}' in DistanceMatrix.csv")
        try:
            distance[i][j] = float(val.values[0])
        except Exception as e:
            raise ValueError(f"Invalid distance value from '{i}' to '{j}': {val.values[0]}") from e
m = gp.Model('TSP_4node')
x_vars = m.addVars(locations, locations, vtype=gp.GRB.BINARY, name='')
n = len(locations)
u_locs = [loc for loc in locations if loc != 'Depot']
u_vars = m.addVars(u_locs, vtype=gp.GRB.CONTINUOUS, lb=1, ub=n - 1, name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for k in u_locs:
    m.addConstr(gp.quicksum((x_vars[i, k] for i in locations if i != k)) == 1, name=f'in_{k}')
    m.addConstr(gp.quicksum((x_vars[k, j] for j in locations if j != k)) == 1, name=f'out_{k}')
m.addConstr(gp.quicksum((x_vars['Depot', j] for j in locations if j != 'Depot')) == 1, name='out_depot')
m.addConstr(gp.quicksum((x_vars[i, 'Depot'] for i in locations if i != 'Depot')) == 1, name='in_depot')
for i in locations:
    m.addConstr(x_vars[i, i] == 0, name=f'no_self_{i}')
for i in u_locs:
    for j in u_locs:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    tour = []
    current = 'Depot'
    visited = set()
    for _ in range(n):
        for j in locations:
            if current != j and x_vars[current, j].X > 0.5:
                tour.append((current, j))
                visited.add(current)
                current = j
                break
    print('Optimal route:')
    route_str = tour[0][0]
    for arc in tour:
        route_str += f' -> {arc[1]}'
    print(route_str)
else:
    print(f'No optimal solution found. Status: {m.status}')