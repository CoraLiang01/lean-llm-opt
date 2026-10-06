import gurobipy as gp
import pandas as pd
import numpy as np
import re

def read_distance_matrix(path):
    df = pd.read_csv(path, sep=',')
    locations = [str(i) for i in range(1, 16)]
    row_ids = df['Unnamed: 0'].astype(str).tolist()
    d = {}
    for i_idx, i in enumerate(row_ids):
        d[i] = {}
        for j in locations:
            if i == j:
                continue
            val = df.at[i_idx, j]
            if pd.isna(val):
                j_idx = row_ids.index(j)
                val_sym = df.at[j_idx, i]
                if pd.isna(val_sym):
                    raise ValueError(f'Distance missing for both ({i},{j}) and ({j},{i})')
                val = val_sym
            d[i][j] = float(val)
    return (locations, d)

def solve_tsp():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
    locations, d = read_distance_matrix(path)
    n = len(locations)
    loc_set = set(locations)
    m = gp.Model('TSP_15_MTZ')
    x = m.addVars([(i, j) for i in locations for j in locations if i != j], vtype=gp.GRB.BINARY, name='')
    u = m.addVars([i for i in locations if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
    m.setObjective(gp.quicksum((d[i][j] * x[i, j] for i in locations for j in locations if i != j)), gp.GRB.MINIMIZE)
    for i in locations:
        m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1, name=f'depart_{i}')
    for j in locations:
        m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1, name=f'arrive_{j}')
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