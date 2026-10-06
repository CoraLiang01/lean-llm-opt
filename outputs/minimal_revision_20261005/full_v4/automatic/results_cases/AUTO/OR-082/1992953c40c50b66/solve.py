import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
df = pd.read_csv(distance_path, sep=',')
locations = ['Depot', 'A', 'B', 'C']
row_ids = df['Unnamed: 0'].astype(str).str.strip().str.casefold().tolist()
col_ids = [col.strip().casefold() for col in df.columns]
for loc in locations:
    if loc.casefold() not in row_ids:
        raise ValueError(f"Row for location '{loc}' not found in DistanceMatrix.csv")
    if loc.casefold() not in col_ids:
        raise ValueError(f"Column for location '{loc}' not found in DistanceMatrix.csv")
row_map = {row.strip(): idx for (idx, row) in enumerate(df['Unnamed: 0'].astype(str))}
col_map = {col.strip(): idx for (idx, col) in enumerate(df.columns)}
distance = {}
for i in locations:
    for j in locations:
        if i == j:
            continue
        row_idx = row_map[i]
        col_idx = col_map[j]
        val = df.iloc[row_idx, col_idx]
        if pd.isnull(val):
            raise ValueError(f'Missing distance from {i} to {j} in DistanceMatrix.csv')
        distance[i, j] = float(val)
arcs = [(i, j) for i in locations for j in locations if i != j]
m = gp.Model('TSP_SmallCourier')
x = m.addVars(arcs, vtype=gp.GRB.BINARY, name='')
customers = [loc for loc in locations if loc != 'Depot']
u = m.addVars(customers, vtype=gp.GRB.INTEGER, lb=1, ub=len(customers), name='')
m.setObjective(gp.quicksum((distance[i, j] * x[i, j] for (i, j) in arcs)), gp.GRB.MINIMIZE)
for i in customers:
    m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1, name=f'depart_{i}')
for j in customers:
    m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1, name=f'arrive_{j}')
m.addConstr(gp.quicksum((x['Depot', j] for j in locations if j != 'Depot')) == 1, name='depart_depot')
m.addConstr(gp.quicksum((x[i, 'Depot'] for i in locations if i != 'Depot')) == 1, name='arrive_depot')
for i in customers:
    for j in customers:
        if i == j:
            continue
        m.addConstr(u[i] - u[j] + len(customers) * x[i, j] <= len(customers) - 1, name=f'subtour_{i}_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (i, j) in arcs:
        print(f'x[{i},{j}] = {x[i, j].X}')
    for i in customers:
        print(f'u[{i}] = {u[i].X}')
else:
    print(f'Solver status: {m.status}')