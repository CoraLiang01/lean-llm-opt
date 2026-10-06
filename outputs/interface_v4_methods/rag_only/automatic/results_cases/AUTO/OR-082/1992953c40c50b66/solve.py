import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
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
        val = row[j].values[0]
        if pd.isnull(val):
            raise ValueError(f'Distance from {i} to {j} is missing in the matrix.')
        distance[i, j] = float(val)
customers = [loc for loc in locations if loc != 'Depot']
m = gp.Model('TSP_Courier')
x = m.addVars(locations, locations, vtype=GRB.BINARY, name='')
u = m.addVars(customers, vtype=GRB.CONTINUOUS, lb=1, ub=len(customers), name='')
m.setObjective(gp.quicksum((distance[i, j] * x[i, j] for i in locations for j in locations)), GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1, name=f'depart_{i}')
for j in locations:
    m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1, name=f'arrive_{j}')
for i in customers:
    for j in customers:
        if i != j:
            m.addConstr(u[i] - u[j] + len(customers) * x[i, j] <= len(customers) - 1, name=f'mtz_{i}_{j}')
for i in locations:
    m.addConstr(x[i, i] == 0, name=f'no_self_{i}')
m.optimize()