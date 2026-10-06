import gurobipy as gp
import pandas as pd
import numpy as np
import math
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(distance_path, sep=',')
location_ids = [str(i) for i in range(1, 16)]
if not set(location_ids).issubset(df.columns):
    missing = set(location_ids) - set(df.columns)
    raise ValueError(f'Missing columns in distance matrix: {missing}')
if not set(range(1, 16)).issubset(df['Unnamed: 0'].astype(int)):
    missing = set(range(1, 16)) - set(df['Unnamed: 0'].astype(int))
    raise ValueError(f'Missing rows in distance matrix: {missing}')
d = {}
for (idx, row) in df.iterrows():
    i = str(int(row['Unnamed: 0']))
    d[i] = {}
    for j in location_ids:
        val = row[j]
        if pd.isnull(val):
            d[i][j] = None
        else:
            d[i][j] = float(val)
for i in location_ids:
    for j in location_ids:
        if i == j:
            d[i][j] = 0.0
        else:
            dij = d[i][j]
            dji = d[j][i]
            if dij is None and dji is not None:
                d[i][j] = dji
            elif dji is None and dij is not None:
                d[j][i] = dij
            elif dij is None and dji is None:
                raise ValueError(f'Missing distance for both ({i},{j}) and ({j},{i})')
            elif abs(dij - dji) > 1e-06:
                raise ValueError(f'Distance matrix not symmetric at ({i},{j}): {dij} vs {dji}')
N = 15
nodes = location_ids
nodes_2plus = [str(i) for i in range(2, N + 1)]

def solve_tsp():
    m = gp.Model('TSP15')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in nodes for j in nodes if i != j]
    x = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
    u = m.addVars(nodes_2plus, vtype=gp.GRB.INTEGER, lb=2, ub=N, name='')
    m.setObjective(gp.quicksum((d[i][j] * x[i, j] for (i, j) in x_keys)), gp.GRB.MINIMIZE)
    for i in nodes:
        m.addConstr(gp.quicksum((x[i, j] for j in nodes if j != i)) == 1, name='leave_%s' % i)
    for j in nodes:
        m.addConstr(gp.quicksum((x[i, j] for i in nodes if i != j)) == 1, name='enter_%s' % j)
    for i in nodes_2plus:
        for j in nodes_2plus:
            if i != j:
                m.addConstr(u[i] - u[j] + N * x[i, j] <= N - 1, name='mtz_%s_%s' % (i, j))
    for i in nodes:
        if (i, i) in x:
            m.addConstr(x[i, i] == 0, name='noloop_%s' % i)
    m.optimize()
    return m
m = solve_tsp()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')