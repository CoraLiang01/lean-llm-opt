import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
distance_df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']
row_ids = distance_df['Unnamed: 0'].tolist()
col_ids = list(distance_df.columns)
for loc in locations:
    if loc not in row_ids:
        raise ValueError(f"Location '{loc}' not found among DistanceMatrix rows: {row_ids}")
    if loc not in col_ids:
        raise ValueError(f"Location '{loc}' not found among DistanceMatrix columns: {col_ids}")
distance = {}
for i in locations:
    row = distance_df.loc[distance_df['Unnamed: 0'] == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in DistanceMatrix.csv")
    row = row.iloc[0]
    distance[i] = {}
    for j in locations:
        val = row[j]
        try:
            distance[i][j] = float(val)
        except Exception:
            raise ValueError(f"Invalid or missing distance from '{i}' to '{j}': '{val}'")
m = gp.Model('TSP_SmallCourier')
x_vars = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
n = len(locations)
customers = [loc for loc in locations if loc != 'Depot']
u_vars = m.addVars(customers, vtype=gp.GRB.CONTINUOUS, lb=1, ub=n - 1, name='')
m.setObjective(gp.quicksum((distance[i][j] * x_vars[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
for k in locations:
    m.addConstr(gp.quicksum((x_vars[k, j] for j in locations if j != k)) == 1, name=f'out_{k}')
    m.addConstr(gp.quicksum((x_vars[i, k] for i in locations if i != k)) == 1, name=f'in_{k}')
for i in customers:
    for j in customers:
        if i == j:
            continue
        m.addConstr(u_vars[i] - u_vars[j] + (n - 1) * x_vars[i, j] <= n - 2, name=f'mtz_{i}_{j}')
m.optimize()