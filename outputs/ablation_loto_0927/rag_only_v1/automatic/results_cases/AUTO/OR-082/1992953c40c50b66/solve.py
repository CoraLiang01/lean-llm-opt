import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv', sep=',')
locations = ['Depot', 'A', 'B', 'C']
row_ids = distance_df['Unnamed: 0'].astype(str).str.strip().tolist()
col_ids = [col.strip() for col in distance_df.columns if col != 'Unnamed: 0']
for loc in locations:
    if loc not in row_ids:
        raise ValueError(f"Location '{loc}' not found in distance matrix rows.")
    if loc not in col_ids:
        raise ValueError(f"Location '{loc}' not found in distance matrix columns.")
distance = {}
for i in locations:
    row = distance_df.loc[distance_df['Unnamed: 0'].astype(str).str.strip() == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in distance matrix.")
    for j in locations:
        if i == j:
            continue
        val = row[j].values[0]
        if pd.isnull(val):
            raise ValueError(f'Distance from {i} to {j} is missing in the matrix.')
        distance[i, j] = float(val)
m = Model('TSP_Courier')
x = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=GRB.BINARY, name='')
u = m.addVars([loc for loc in locations if loc != 'Depot'], vtype=GRB.CONTINUOUS, lb=1, ub=len(locations) - 1, name='')
m.setObjective(quicksum((distance[i, j] * x[i, j] for i in locations for j in locations if i != j)), GRB.MINIMIZE)
for i in locations:
    m.addConstr(quicksum((x[i, j] for j in locations if j != i)) == 1, name='')
for j in locations:
    m.addConstr(quicksum((x[i, j] for i in locations if i != j)) == 1, name='')
for i in [loc for loc in locations if loc != 'Depot']:
    for j in [loc for loc in locations if loc != 'Depot']:
        if i == j:
            continue
        m.addConstr(u[i] - u[j] + (len(locations) - 1) * x[i, j] <= len(locations) - 2, name='')
m.optimize()
if m.status == GRB.OPTIMAL:
    tour = []
    current = 'Depot'
    visited = set()
    for _ in range(len(locations)):
        tour.append(current)
        visited.add(current)
        for j in locations:
            if j != current and x[current, j].X > 0.5:
                current = j
                break
    tour.append('Depot')
    print('Optimal tour:', ' -> '.join(tour))
    print('Total travel distance:', m.objVal)
else:
    print('No optimal solution found.')