import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)

def norm(s):
    return s.strip().casefold()
locations = ['Depot', 'A', 'B', 'C']
locations_norm = [norm(loc) for loc in locations]
row_id_map = {i: norm(row['Unnamed: 0']) for i, row in df.iterrows()}
row_name_to_idx = {norm(row['Unnamed: 0']): i for i, row in df.iterrows()}
col_name_to_orig = {norm(col): col for col in df.columns if col != 'Unnamed: 0'}
for loc, loc_norm in zip(locations, locations_norm):
    if loc_norm not in row_name_to_idx:
        raise ValueError(f"Location '{loc}' not found as a row in the distance matrix.")
    if loc_norm not in col_name_to_orig:
        raise ValueError(f"Location '{loc}' not found as a column in the distance matrix.")
distance = {}
for i in locations:
    i_norm = norm(i)
    row_idx = row_name_to_idx[i_norm]
    for j in locations:
        j_norm = norm(j)
        col = col_name_to_orig[j_norm]
        val_str = df.at[row_idx, col]
        try:
            val = float(val_str)
        except Exception:
            raise ValueError(f"Invalid distance value for ({i},{j}): '{val_str}'")
        distance[i, j] = val
m = gp.Model('TSP_4node')
x_vars = m.addVars(locations, locations, vtype=gp.GRB.BINARY, name='')
u_locs = [loc for loc in locations if loc != 'Depot']
u_vars = m.addVars(u_locs, vtype=gp.GRB.CONTINUOUS, lb=1, ub=len(locations) - 1, name='')
m.setObjective(gp.quicksum((distance[i, j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in locations if j != i)) == 1, name=f'out_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in locations if i != j)) == 1, name=f'in_{j}')
n = len(locations)
for i in u_locs:
    for j in u_locs:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    succ = {}
    for i in locations:
        for j in locations:
            if i != j and x_vars[i, j].X > 0.5:
                succ[i] = j
    tour = ['Depot']
    while True:
        last = tour[-1]
        next_loc = succ.get(last, None)
        if next_loc is None or next_loc == 'Depot':
            break
        tour.append(next_loc)
    tour.append('Depot')
    print('Optimal route:')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')