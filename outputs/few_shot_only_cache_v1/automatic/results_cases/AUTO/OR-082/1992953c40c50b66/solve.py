import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
distance_df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']
row_ids = distance_df['Unnamed: 0'].str.strip()
col_ids = [col.strip() for col in distance_df.columns if col != 'Unnamed: 0']
for loc in locations:
    if loc not in row_ids.values:
        raise ValueError(f"Location '{loc}' not found in DistanceMatrix.csv rows.")
    if loc not in col_ids:
        raise ValueError(f"Location '{loc}' not found in DistanceMatrix.csv columns.")
d = {}
for i in locations:
    row = distance_df.loc[row_ids == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in DistanceMatrix.csv.")
    for j in locations:
        if i == j:
            continue
        val = row.iloc[0][j]
        try:
            d[i, j] = float(val)
        except Exception:
            raise ValueError(f"Invalid distance value from '{i}' to '{j}': '{val}'")
m = gp.Model('TSP_SmallCourier')
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((d[i, j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for k in ['A', 'B', 'C']:
    m.addConstr(gp.quicksum((x_vars[i, k] for i in locations if i != k)) == 1, name=f'arrive_{k}')
    m.addConstr(gp.quicksum((x_vars[k, j] for j in locations if j != k)) == 1, name=f'depart_{k}')
m.addConstr(gp.quicksum((x_vars['Depot', j] for j in locations if j != 'Depot')) == 1, name='depart_depot')
m.addConstr(gp.quicksum((x_vars[i, 'Depot'] for i in locations if i != 'Depot')) == 1, name='arrive_depot')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    tour_arcs = {(i, j): x_vars[i, j].X for i, j in x_vars.keys() if x_vars[i, j].X > 0.5}
    tour = ['Depot']
    current = 'Depot'
    while True:
        next_nodes = [j for (i, j), v in tour_arcs.items() if i == current]
        if not next_nodes:
            break
        next_node = next_nodes[0]
        tour.append(next_node)
        current = next_node
        if current == 'Depot':
            break
    print('Optimal route:')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')