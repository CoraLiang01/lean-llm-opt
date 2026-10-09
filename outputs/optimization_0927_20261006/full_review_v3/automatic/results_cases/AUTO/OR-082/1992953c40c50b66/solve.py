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
row_ids = df['Unnamed: 0'].apply(norm).tolist()
row_id_map = dict(zip(df['Unnamed: 0'], row_ids))
col_id_map = {col: norm(col) for col in df.columns}
norm_to_row = {v: k for (k, v) in row_id_map.items()}
norm_to_col = {v: k for (k, v) in col_id_map.items()}
for loc in locations_norm:
    if loc not in norm_to_row:
        raise KeyError(f"Location '{loc}' not found in DistanceMatrix.csv rows.")
    if loc not in norm_to_col:
        raise KeyError(f"Location '{loc}' not found in DistanceMatrix.csv columns.")
distance = {}
for i_norm in locations_norm:
    i_row = norm_to_row[i_norm]
    row = df[df['Unnamed: 0'].apply(norm) == i_norm].iloc[0]
    distance[i_row] = {}
    for j_norm in locations_norm:
        j_col = norm_to_col[j_norm]
        val = row[j_col]
        try:
            distance[i_row][j_col] = float(val)
        except Exception:
            raise ValueError(f"Invalid distance value for ({i_row},{j_col}): '{val}'")
nodes = [norm_to_row[loc] for loc in locations_norm]
m = gp.Model('TSP_SmallCourier')
x_vars = m.addVars([(i, j) for i in nodes for j in nodes if i != j], vtype=gp.GRB.BINARY, name='')
u_nodes = [n for n in nodes if n != 'Depot']
u_vars = m.addVars(u_nodes, vtype=gp.GRB.INTEGER, lb=1, ub=len(u_nodes), name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for i in nodes for j in nodes if i != j)), gp.GRB.MINIMIZE)
for i in nodes:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in nodes if j != i)) == 1, name=f'depart_{i}')
for j in nodes:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in nodes if i != j)) == 1, name=f'arrive_{j}')
n_customers = len(u_nodes)
for i in u_nodes:
    for j in u_nodes:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + n_customers * x_vars[i, j] <= n_customers - 1, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    succ = {}
    for i in nodes:
        for j in nodes:
            if i != j and x_vars[i, j].X > 0.5:
                succ[i] = j
    tour = ['Depot']
    while True:
        last = tour[-1]
        next_node = succ.get(last)
        if next_node is None or next_node == 'Depot':
            tour.append('Depot')
            break
        tour.append(next_node)
    print('Optimal route:')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')