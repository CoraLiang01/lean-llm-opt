import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
distance_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv', sep=',')
locations = ['Depot', 'A', 'B', 'C']
distance_df['Unnamed: 0'] = distance_df['Unnamed: 0'].astype(str).str.strip()
distance_df.columns = [str(col).strip() for col in distance_df.columns]
row_ids = set(distance_df['Unnamed: 0'])
col_ids = set(distance_df.columns)
for loc in locations:
    if loc not in row_ids:
        raise ValueError(f"Row for location '{loc}' not found in DistanceMatrix.csv")
    if loc not in col_ids:
        raise ValueError(f"Column for location '{loc}' not found in DistanceMatrix.csv")
distance = {}
for i in locations:
    row = distance_df.loc[distance_df['Unnamed: 0'] == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in DistanceMatrix.csv")
    distance[i] = {}
    for j in locations:
        val = row[j].values[0]
        if pd.isnull(val):
            raise ValueError(f"Distance from '{i}' to '{j}' is missing in DistanceMatrix.csv")
        distance[i][j] = float(val)
m = Model('TSP_SmallCourier')
x = m.addVars(locations, locations, vtype=GRB.BINARY, name='')
n = len(locations)
u = m.addVars([loc for loc in locations if loc != 'Depot'], vtype=GRB.CONTINUOUS, lb=1, ub=n - 1, name='')
m.setObjective(quicksum((distance[i][j] * x[i, j] for i in locations for j in locations if i != j)), GRB.MINIMIZE)
for i in locations:
    m.addConstr(quicksum((x[i, j] for j in locations if j != i)) == 1, name=f'depart_{i}')
for j in locations:
    m.addConstr(quicksum((x[i, j] for i in locations if i != j)) == 1, name=f'arrive_{j}')
for i in [loc for loc in locations if loc != 'Depot']:
    for j in [loc for loc in locations if loc != 'Depot']:
        if i != j:
            m.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2, name=f'mtz_{i}_{j}')
m.optimize()