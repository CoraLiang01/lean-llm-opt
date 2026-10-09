import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']
distance = {}
for i in locations:
    row = df[df['Unnamed: 0'].str.strip().casefold() == i.casefold()]
    if row.empty:
        raise KeyError(f"Row for location '{i}' not found in DistanceMatrix.csv")
    row = row.iloc[0]
    for j in locations:
        val = row[j]
        try:
            distance[i, j] = float(val)
        except Exception:
            raise ValueError(f"Distance from {i} to {j} is not a valid number: '{val}'")
m = gp.Model('TSP_4node')
x_vars = m.addVars(locations, locations, vtype=gp.GRB.BINARY, name='')
n_customers = 3
u_vars = m.addVars([loc for loc in locations if loc != 'Depot'], vtype=gp.GRB.INTEGER, lb=1, ub=n_customers, name='')
m.setObjective(gp.quicksum((distance[i, j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for k in [loc for loc in locations if loc != 'Depot']:
    m.addConstr(gp.quicksum((x_vars[i, k] for i in locations if i != k)) == 1, name=f'in_{k}')
for k in [loc for loc in locations if loc != 'Depot']:
    m.addConstr(gp.quicksum((x_vars[k, j] for j in locations if j != k)) == 1, name=f'out_{k}')
m.addConstr(gp.quicksum((x_vars['Depot', j] for j in locations if j != 'Depot')) == 1, name='depot_depart')
m.addConstr(gp.quicksum((x_vars[i, 'Depot'] for i in locations if i != 'Depot')) == 1, name='depot_arrive')
for i in locations:
    m.addConstr(x_vars[i, i] == 0, name=f'noloop_{i}')
customers = [loc for loc in locations if loc != 'Depot']
for i in customers:
    for j in customers:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + n_customers * x_vars[i, j] <= n_customers - 1, name=f'subtour_{i}_{j}')
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
        if last not in succ:
            break
        nxt = succ[last]
        tour.append(nxt)
        if nxt == 'Depot':
            break
    print('Optimal route:')
    print(' -> '.join(tour))
else:
    print(f'No optimal solution found. Status: {m.status}')