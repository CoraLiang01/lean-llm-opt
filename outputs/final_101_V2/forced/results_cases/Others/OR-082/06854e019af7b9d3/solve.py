import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv', sep=',')
locations = ['Depot', 'A', 'B', 'C']

def normalize_id(x):
    return str(x).strip().casefold()
row_ids = [normalize_id(x) for x in distance_df['Unnamed: 0']]
col_ids = [normalize_id(x) for x in distance_df.columns[1:len(locations) + 1]]
row_id_map = {normalize_id(x): x for x in distance_df['Unnamed: 0']}
col_id_map = {normalize_id(x): x for x in distance_df.columns}
for loc in locations:
    if normalize_id(loc) not in row_id_map or normalize_id(loc) not in col_id_map:
        raise ValueError(f"Location '{loc}' not found in both rows and columns of the distance matrix.")
distance = {}
for i in locations:
    distance[i] = {}
    row = distance_df.loc[distance_df['Unnamed: 0'].apply(lambda x: normalize_id(x) == normalize_id(i))]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in distance matrix.")
    for j in locations:
        colname = col_id_map[normalize_id(j)]
        val = row.iloc[0][colname]
        if pd.isnull(val):
            raise ValueError(f"Distance from '{i}' to '{j}' is missing in the matrix.")
        distance[i][j] = float(val)
m = gp.Model('TSP_4node')
arcs = [(i, j) for i in locations for j in locations if i != j]
x = m.addVars(arcs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for i, j in arcs)), gp.GRB.MINIMIZE)
for k in locations:
    m.addConstr(gp.quicksum((x[k, j] for j in locations if j != k)) == 1, name=f'out_{k}')
    m.addConstr(gp.quicksum((x[i, k] for i in locations if i != k)) == 1, name=f'in_{k}')
n = len(locations)
u = m.addVars([loc for loc in locations if loc != 'Depot'], lb=1, ub=n - 1, vtype=gp.GRB.CONTINUOUS, name='')
for i in locations:
    for j in locations:
        if i != j and i != 'Depot' and (j != 'Depot'):
            m.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2, name=f'subtour_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    succ = {}
    for i, j in arcs:
        if x[i, j].X > 0.5:
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
    print('Visit order:')
    for idx, loc in enumerate(tour):
        print(f'  {idx + 1}: {loc}')
else:
    print(f'No optimal solution found. Status: {m.status}')