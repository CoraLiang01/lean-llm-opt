import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_tsp():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
    df = pd.read_csv(path, sep=',')
    locations = ['Depot', 'A', 'B', 'C']
    df['row_id'] = df['Unnamed: 0'].astype(str).str.strip()
    distance = {}
    for i in locations:
        row = df[df['row_id'].str.casefold() == i.casefold()]
        if row.empty:
            raise ValueError(f"Missing row for location '{i}' in DistanceMatrix.csv")
        row = row.iloc[0]
        distance[i] = {}
        for j in locations:
            if j not in df.columns:
                raise ValueError(f"Missing column for location '{j}' in DistanceMatrix.csv")
            val = row[j]
            try:
                distance[i][j] = float(val)
            except Exception:
                raise ValueError(f'Non-numeric distance from {i} to {j}: {val}')
    arcs = [(i, j) for i in locations for j in locations if i != j]
    m = gp.Model('TSP')
    x = m.addVars(arcs, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for (i, j) in arcs)), gp.GRB.MINIMIZE)
    for i in locations:
        m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1, name=f'out_{i}')
    for j in locations:
        m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1, name=f'in_{j}')
    n = len(locations)
    u = m.addVars([loc for loc in locations if loc != 'Depot'], lb=1, ub=n - 1, vtype=gp.GRB.CONTINUOUS, name='')
    for i in locations:
        if i == 'Depot':
            continue
        for j in locations:
            if j == 'Depot' or i == j:
                continue
            m.addConstr(u[i] - u[j] + (n - 1) * x[i, j] <= n - 2, name=f'subtour_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_tsp()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')