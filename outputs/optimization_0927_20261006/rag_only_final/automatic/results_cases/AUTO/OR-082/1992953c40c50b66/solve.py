import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv', sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']
distance_matrix = {}
for i in locations:
    row = distance_df.loc[distance_df['Unnamed: 0'] == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in DistanceMatrix.csv")
    for j in locations:
        val = row.iloc[0][j]
        try:
            distance_matrix[i, j] = float(val)
        except Exception:
            raise ValueError(f"Invalid or missing distance from '{i}' to '{j}': {val}")
m = Model()
x_vars = m.addVars(locations, locations, vtype=GRB.BINARY, name='')
u_locs = [loc for loc in locations if loc != 'Depot']
u_vars = m.addVars(u_locs, vtype=GRB.INTEGER, lb=1, ub=len(u_locs), name='')
m.setObjective(quicksum((distance_matrix[i, j] * x_vars[i, j] for i in locations for j in locations if i != j)), GRB.MINIMIZE)
for i in locations:
    m.addConstr(quicksum((x_vars[i, j] for j in locations if j != i)) == 1, name=f'leave_{i}')
for j in locations:
    m.addConstr(quicksum((x_vars[i, j] for i in locations if i != j)) == 1, name=f'arrive_{j}')
for i in u_locs:
    for j in u_locs:
        if i != j:
            m.addConstr(u_vars[i] - u_vars[j] + len(u_locs) * x_vars[i, j] <= len(u_locs) - 1, name=f'subtour_{i}_{j}')
m.optimize()