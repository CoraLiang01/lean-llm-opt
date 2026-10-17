import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']
row_ids = df['Unnamed: 0'].astype(str).str.strip()
df = df.set_index(row_ids)
df = df.loc[locations, locations]
distance = {}
for i in locations:
    for j in locations:
        val = df.loc[i, j]
        try:
            distance[i, j] = float(val)
        except Exception:
            raise ValueError(f'Invalid distance value for ({i},{j}): {val}')
m = gp.Model('TSP_SmallCourier')
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
n = len(locations)
u_locs = [loc for loc in locations if loc != 'Depot']
u_vars = m.addVars(u_locs, lb=1, ub=n - 1, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((distance[i, j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in locations if j != i)) == 1, name=f'out_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in locations if i != j)) == 1, name=f'in_{j}')
for i in u_locs:
    for j in u_locs:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2, name=f'mtz_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    succ = {}
    for i, j in x_vars.keys():
        if x_vars[i, j].X > 0.5:
            succ[i] = j
    tour = ['Depot']
    while True:
        last = tour[-1]
        next_loc = succ.get(last, None)
        if next_loc is None or next_loc == 'Depot':
            tour.append('Depot')
            break
        tour.append(next_loc)
    print('Optimal route:')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')