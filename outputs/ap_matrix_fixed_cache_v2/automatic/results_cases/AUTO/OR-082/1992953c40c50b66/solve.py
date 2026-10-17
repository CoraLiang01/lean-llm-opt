import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']

def norm(s):
    return s.strip().casefold()
row_id_map = {norm(r): r for r in df['Unnamed: 0']}
col_id_map = {norm(c): c for c in df.columns if c != 'Unnamed: 0'}
for loc in locations:
    if norm(loc) not in row_id_map:
        raise KeyError(f"Location '{loc}' not found as a row in DistanceMatrix.csv")
    if norm(loc) not in col_id_map:
        raise KeyError(f"Location '{loc}' not found as a column in DistanceMatrix.csv")
df_sub = df.set_index('Unnamed: 0')
df_sub = df_sub.loc[[row_id_map[norm(loc)] for loc in locations], [col_id_map[norm(loc)] for loc in locations]]
distance = {}
for i in locations:
    for j in locations:
        val = df_sub.loc[row_id_map[norm(i)], col_id_map[norm(j)]]
        try:
            distance[i, j] = float(val)
        except Exception:
            raise ValueError(f"Invalid distance value for ({i},{j}): '{val}'")
m = gp.Model('TSP_4node')
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
u_locs = [loc for loc in locations if loc != 'Depot']
u_vars = m.addVars(u_locs, vtype=gp.GRB.CONTINUOUS, lb=1, ub=len(locations) - 1, name='')
m.setObjective(gp.quicksum((distance[i, j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in locations if j != i)) == 1, name=f'out_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in locations if i != j)) == 1, name=f'in_{j}')
for i in u_locs:
    for j in u_locs:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + (len(locations) - 1) * x_vars[i, j] <= len(locations) - 2, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    tour = {}
    for i in locations:
        for j in locations:
            if i != j and x_vars[i, j].X > 0.5:
                tour[i] = j
    sequence = ['Depot']
    current = 'Depot'
    while True:
        next_loc = tour.get(current)
        if next_loc is None or next_loc == 'Depot':
            break
        sequence.append(next_loc)
        current = next_loc
    sequence.append('Depot')
    print('Optimal route:')
    print(' -> '.join(sequence))
    print('Visit order:')
    for idx, loc in enumerate(sequence):
        print(f'  {idx + 1}: {loc}')
else:
    print(f'No optimal solution found. Status: {m.status}')