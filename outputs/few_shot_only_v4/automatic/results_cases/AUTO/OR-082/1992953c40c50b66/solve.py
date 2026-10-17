import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
distance_df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)

def norm_id(x):
    return x.strip().casefold()
row_ids = distance_df.iloc[:, 0].apply(norm_id).tolist()
col_ids = [norm_id(c) for c in distance_df.columns[1:]]
row_id_map = {norm_id(distance_df.iloc[i, 0]): distance_df.iloc[i, 0] for i in range(len(distance_df))}
col_id_map = {norm_id(c): c for c in distance_df.columns[1:]}
required_locs = ['Depot', 'A', 'B', 'C']
required_locs_norm = [norm_id(x) for x in required_locs]
for loc in required_locs_norm:
    if loc not in row_ids or loc not in col_ids:
        raise ValueError(f"Location '{loc}' not found in both rows and columns of DistanceMatrix.csv.")
locations = [row_id_map[loc] for loc in required_locs_norm]
distances = {}
for i_norm, i_orig in zip(required_locs_norm, locations):
    row_idx = row_ids.index(i_norm)
    row = distance_df.iloc[row_idx, 1:]
    distances[i_orig] = {}
    for j_norm, j_orig in zip(required_locs_norm, locations):
        col_idx = col_ids.index(j_norm)
        cell_val = row.iloc[col_idx]
        try:
            dist = float(cell_val)
        except Exception:
            raise ValueError(f"Invalid distance value from '{i_orig}' to '{j_orig}': '{cell_val}'")
        distances[i_orig][j_orig] = dist
arcs = [(i, j) for i in locations for j in locations if i != j]
m = gp.Model('TSP_4node')
x_vars = m.addVars(arcs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((distances[i][j] * x_vars[i, j] for i, j in arcs)), gp.GRB.MINIMIZE)
for k in locations:
    if norm_id(k) == norm_id('Depot'):
        m.addConstr(gp.quicksum((x_vars[k, j] for j in locations if j != k)) == 1, name=f'depart_{k}')
        m.addConstr(gp.quicksum((x_vars[i, k] for i in locations if i != k)) == 1, name=f'arrive_{k}')
    else:
        m.addConstr(gp.quicksum((x_vars[i, k] for i in locations if i != k)) == 1, name=f'arrive_{k}')
        m.addConstr(gp.quicksum((x_vars[k, j] for j in locations if j != k)) == 1, name=f'depart_{k}')
customer_locs = [k for k in locations if norm_id(k) != norm_id('Depot')]
u_vars = m.addVars(customer_locs, vtype=gp.GRB.CONTINUOUS, lb=1, ub=len(customer_locs), name='')
n = len(locations)
for i in customer_locs:
    for j in customer_locs:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + n * x_vars[i, j] <= n - 1, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f}')
    next_loc = {}
    for i, j in arcs:
        if x_vars[i, j].X > 0.5:
            next_loc[i] = j
    tour = []
    current = [k for k in locations if norm_id(k) == norm_id('Depot')][0]
    for _ in range(len(locations)):
        tour.append(current)
        current = next_loc[current]
    tour.append(tour[0])
    print('Optimal route:')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')