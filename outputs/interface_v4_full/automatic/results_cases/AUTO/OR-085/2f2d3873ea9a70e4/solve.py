import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_tsp():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
    df = pd.read_csv(path, sep=',')
    locations = [str(i) for i in range(1, 16)]
    n = len(locations)
    d = {}
    for idx, row in df.iterrows():
        i = str(int(row['Unnamed: 0']))
        for j in locations:
            if j == i:
                continue
            val = row.get(j, np.nan)
            if pd.notnull(val):
                d[i, j] = float(val)
    for i in locations:
        for j in locations:
            if i == j:
                continue
            if (i, j) not in d and (j, i) in d:
                d[i, j] = d[j, i]
            elif (i, j) not in d and (j, i) not in d:
                raise ValueError(f'Missing distance between {i} and {j} in both directions.')
    m = gp.Model('TSP_15_MTZ')
    x = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
    u = m.addVars([i for i in locations if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
    m.setObjective(gp.quicksum((d[i, j] * x[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
    for i in locations:
        m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1, name=f'leave_{i}')
    for j in locations:
        m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1, name=f'enter_{j}')
    for i in locations:
        if i == '1':
            continue
        for j in locations:
            if j == '1' or i == j:
                continue
            m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f'mtz_{i}_{j}')
    m.optimize()
    return m
m = solve_tsp()