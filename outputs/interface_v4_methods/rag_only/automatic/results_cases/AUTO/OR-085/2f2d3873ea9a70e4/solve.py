import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
    df = pd.read_csv(path, sep=',')
    location_ids = [str(i) for i in range(1, 16)]
    n = len(location_ids)
    distance = {i: {j: None for j in location_ids} for i in location_ids}
    for idx, row in df.iterrows():
        i = str(row['Unnamed: 0'])
        for j in location_ids:
            val = row[j] if j in row and (not pd.isnull(row[j])) else None
            if val is not None:
                distance[i][j] = float(val)
    for i in location_ids:
        for j in location_ids:
            if i == j:
                distance[i][j] = 0.0
            elif distance[i][j] is None and distance[j][i] is not None:
                distance[i][j] = distance[j][i]
            elif distance[i][j] is not None and distance[j][i] is None:
                distance[j][i] = distance[i][j]
            elif distance[i][j] is None and distance[j][i] is None:
                raise ValueError(f'Missing distance between {i} and {j}')
    for i in location_ids:
        for j in location_ids:
            if distance[i][j] is None:
                raise ValueError(f'Distance between {i} and {j} is missing after symmetry fill.')
    m = gp.Model('TSP')
    x = m.addVars(location_ids, location_ids, vtype=GRB.BINARY, name='')
    for i in location_ids:
        m.addConstr(x[i, i] == 0)
    u = {}
    for i in location_ids:
        if i != '1':
            u[i] = m.addVar(vtype=GRB.INTEGER, lb=2, ub=n, name=f'u_{i}')
    m.update()
    for i in location_ids:
        m.addConstr(gp.quicksum((x[i, j] for j in location_ids if j != i)) == 1)
    for j in location_ids:
        m.addConstr(gp.quicksum((x[i, j] for i in location_ids if i != j)) == 1)
    for i in location_ids:
        if i == '1':
            continue
        for j in location_ids:
            if j == '1' or i == j:
                continue
            m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)
    m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for i in location_ids for j in location_ids)), GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem()