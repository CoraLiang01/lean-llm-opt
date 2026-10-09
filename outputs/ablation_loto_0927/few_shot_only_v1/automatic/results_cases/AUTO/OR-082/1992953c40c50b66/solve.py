import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv', sep=',')
locations = ['Depot', 'A', 'B', 'C']
distance_df = distance_df.set_index('Unnamed: 0')
distance_df.index = distance_df.index.astype(str)
distance_df.columns = distance_df.columns.astype(str)
for loc in locations:
    if loc not in distance_df.index or loc not in distance_df.columns:
        raise ValueError(f"Location '{loc}' not found in both rows and columns of DistanceMatrix.csv.")
distance = {}
for i in locations:
    distance[i] = {}
    for j in locations:
        if i != j:
            val = distance_df.loc[i, j]
            try:
                distance[i][j] = float(val)
            except Exception:
                raise ValueError(f'Distance from {i} to {j} is not a valid number: {val}')
m = gp.Model('TSP_3_Customers')
arcs = [(i, j) for i in locations for j in locations if i != j]
x = m.addVars(arcs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for (i, j) in arcs)), gp.GRB.MINIMIZE)
for k in ['A', 'B', 'C']:
    m.addConstr(gp.quicksum((x[i, k] for i in locations if i != k)) == 1, name=f'enter_{k}')
    m.addConstr(gp.quicksum((x[k, j] for j in locations if j != k)) == 1, name=f'exit_{k}')
m.addConstr(gp.quicksum((x['Depot', j] for j in locations if j != 'Depot')) == 1, name='depot_depart')
m.addConstr(gp.quicksum((x[i, 'Depot'] for i in locations if i != 'Depot')) == 1, name='depot_return')
u = m.addVars(['A', 'B', 'C'], vtype=gp.GRB.CONTINUOUS, lb=1, ub=3, name='')
for k in ['A', 'B', 'C']:
    for l in ['A', 'B', 'C']:
        if k != l:
            m.addConstr(u[k] - u[l] + 3 * x[k, l] <= 2, name=f'mtz_{k}_{l}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    succ = {}
    for (i, j) in arcs:
        if x[i, j].X > 0.5:
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
        if len(tour) > len(locations) + 2:
            print('Error: Tour reconstruction exceeded expected length.')
            break
    print('Optimal route:')
    print(' -> '.join(tour))
    print('Visit order:')
    for (idx, loc) in enumerate(tour):
        print(f'  {idx + 1}: {loc}')
else:
    print(f'No optimal solution found. Status: {m.status}')