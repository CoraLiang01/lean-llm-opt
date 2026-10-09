import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']
row_ids = df['Unnamed: 0'].str.strip()
col_ids = [col.strip() for col in df.columns if col != 'Unnamed: 0']
for loc in locations:
    if loc not in row_ids.values:
        raise ValueError(f"Location '{loc}' not found in DistanceMatrix.csv rows.")
    if loc not in col_ids:
        raise ValueError(f"Location '{loc}' not found in DistanceMatrix.csv columns.")
distance = {}
for i in locations:
    row = df.loc[row_ids == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in DistanceMatrix.csv.")
    row = row.iloc[0]
    for j in locations:
        if i == j:
            continue
        val = row[j]
        try:
            distance[i, j] = float(val)
        except Exception:
            raise ValueError(f"Invalid or missing distance from '{i}' to '{j}': '{val}'")
m = gp.Model('TSP_4node')
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((distance[i, j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for k in ['A', 'B', 'C']:
    m.addConstr(gp.quicksum((x_vars[i, k] for i in locations if i != k)) == 1, name=f'arrive_{k}')
    m.addConstr(gp.quicksum((x_vars[k, j] for j in locations if j != k)) == 1, name=f'depart_{k}')
m.addConstr(gp.quicksum((x_vars['Depot', j] for j in locations if j != 'Depot')) == 1, name='depart_depot')
m.addConstr(gp.quicksum((x_vars[i, 'Depot'] for i in locations if i != 'Depot')) == 1, name='arrive_depot')
u_vars = m.addVars(['A', 'B', 'C'], vtype=gp.GRB.CONTINUOUS, lb=1, ub=3, name='')
for i in ['A', 'B', 'C']:
    for j in ['A', 'B', 'C']:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + 3 * x_vars[i, j] <= 2, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    succ = {}
    for (i, j) in x_vars.keys():
        if x_vars[i, j].X > 0.5:
            succ[i] = j
    tour = ['Depot']
    while True:
        last = tour[-1]
        if last not in succ:
            break
        nxt = succ[last]
        tour.append(nxt)
        if nxt == 'Depot':
            break
        if len(tour) > 5:
            print('Warning: Tour reconstruction exceeded expected length.')
            break
    print('Optimal route:')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')