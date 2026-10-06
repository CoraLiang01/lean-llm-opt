import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',')
locations = ['Depot', 'A', 'B', 'C']

def normalize_id(x):
    return re.sub('\\s+', '', str(x)).casefold()
loc_norm = [normalize_id(l) for l in locations]
row_id_map = {normalize_id(idx): idx for idx in df['Unnamed: 0']}
col_id_map = {normalize_id(col): col for col in df.columns if col != 'Unnamed: 0'}
for l in loc_norm:
    if l not in row_id_map:
        raise KeyError(f"Location '{l}' not found in CSV rows.")
    if l not in col_id_map:
        raise KeyError(f"Location '{l}' not found in CSV columns.")
distance = {}
for i in locations:
    for j in locations:
        if i != j:
            row = row_id_map[normalize_id(i)]
            col = col_id_map[normalize_id(j)]
            val = df.loc[df['Unnamed: 0'] == row, col].values
            if len(val) != 1:
                raise ValueError(f'Distance from {i} to {j} not found or ambiguous in CSV.')
            distance[i, j] = float(val[0])
m = gp.Model('TSP_4node')
x = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((distance[i, j] * x[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1, name=f'leave_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1, name=f'enter_{j}')
u = m.addVars([loc for loc in locations if loc != 'Depot'], vtype=gp.GRB.CONTINUOUS, lb=1, ub=len(locations) - 1, name='')
for i in [loc for loc in locations if loc != 'Depot']:
    for j in [loc for loc in locations if loc != 'Depot']:
        if i != j:
            m.addConstr(u[i] - u[j] + (len(locations) - 1) * x[i, j] <= len(locations) - 2, name=f'subtour_{i}_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total travel distance: {m.objVal:.2f} km')
    succ = {}
    for i, j in x.keys():
        if x[i, j].X > 0.5:
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
    print('Visit order:')
    for idx, loc in enumerate(tour):
        print(f'  {idx + 1}: {loc}')
else:
    print(f'No optimal solution found. Status: {m.status}')