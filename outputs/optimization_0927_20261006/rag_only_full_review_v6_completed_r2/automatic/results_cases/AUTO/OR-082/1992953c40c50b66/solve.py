import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv', dtype=str, keep_default_na=False)
location_ids = ['Depot', 'A', 'B', 'C']
row_ids = distance_df['Unnamed: 0'].str.strip()
col_ids = [col.strip() for col in distance_df.columns if col != 'Unnamed: 0']
for loc in location_ids:
    if loc not in row_ids.values:
        raise ValueError(f"Location '{loc}' not found in distance matrix rows.")
    if loc not in col_ids:
        raise ValueError(f"Location '{loc}' not found in distance matrix columns.")
distance_dict = {}
for i in location_ids:
    row = distance_df.loc[row_ids == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in distance matrix.")
    for j in location_ids:
        val = row.iloc[0][j]
        try:
            distance = float(val)
        except Exception:
            raise ValueError(f"Invalid distance value for ({i},{j}): '{val}'")
        distance_dict[i, j] = distance
N = location_ids
m = Model()
x_vars = m.addVars(N, N, vtype=GRB.BINARY, name='')
u_vars = {}
for i in N:
    if i != 'Depot':
        u_vars[i] = m.addVar(lb=1, ub=len(N) - 1, vtype=GRB.CONTINUOUS, name=f'u_{i}')
m.setObjective(quicksum((distance_dict[i, j] * x_vars[i, j] for i in N for j in N)), GRB.MINIMIZE)
for i in N:
    m.addConstr(quicksum((x_vars[i, j] for j in N if j != i)) == 1, name=f'depart_{i}')
for j in N:
    m.addConstr(quicksum((x_vars[i, j] for i in N if i != j)) == 1, name=f'arrive_{j}')
for i in N:
    if i == 'Depot':
        continue
    for j in N:
        if j == 'Depot' or i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + (len(N) - 1) * x_vars[i, j] <= len(N) - 2, name=f'subtour_{i}_{j}')
m.optimize()