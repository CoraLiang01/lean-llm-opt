import gurobipy as gp
import pandas as pd
import numpy as np
import re
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
distance_df = pd.read_csv(distance_path, sep=',', dtype=str, keep_default_na=False)
locations = ['Depot', 'A', 'B', 'C']
row_ids = distance_df['Unnamed: 0'].tolist()
col_ids = [col for col in distance_df.columns if col in locations]
if set(locations) - set(row_ids):
    raise ValueError(f'Missing required rows in DistanceMatrix.csv: {set(locations) - set(row_ids)}')
if set(locations) - set(col_ids):
    raise ValueError(f'Missing required columns in DistanceMatrix.csv: {set(locations) - set(col_ids)}')
distance = {}
for i in locations:
    row = distance_df.loc[distance_df['Unnamed: 0'] == i]
    if row.empty:
        raise ValueError(f"Row for location '{i}' not found in DistanceMatrix.csv")
    for j in locations:
        val = row.iloc[0][j]
        try:
            distance_ij = float(val)
        except Exception:
            raise ValueError(f"Invalid distance value from '{i}' to '{j}': '{val}'")
        distance[i, j] = distance_ij
arc_keys = [(i, j) for i in locations for j in locations if i != j]

def solve_tsp():
    m = gp.Model('TSP_SmallCourier')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(arc_keys, vtype=gp.GRB.BINARY, name='')
    customers = [loc for loc in locations if loc != 'Depot']
    u_vars = m.addVars(customers, lb=1, ub=len(customers), vtype=gp.GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((distance[i, j] * x_vars[i, j] for (i, j) in arc_keys)), gp.GRB.MINIMIZE)
    for i in locations:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in locations if j != i)) == 1, name=f'out_{i}')
    for j in locations:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in locations if i != j)) == 1, name=f'in_{j}')
    n = len(customers)
    for i in customers:
        for j in customers:
            if i != j:
                m.addConstr(u_vars[i] - u_vars[j] + n * x_vars[i, j] <= n - 1, name=f'mtz_{i}_{j}')
    m.optimize()
    return m
m = solve_tsp()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')