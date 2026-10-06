import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_tsp():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
    df = pd.read_csv(path, sep=',')
    node_labels = [str(i) for i in range(1, 16)]
    nodes = [int(lbl) for lbl in node_labels]
    n = len(nodes)
    df = df.set_index('Unnamed: 0')
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    missing_rows = set(node_labels) - set(df.index)
    missing_cols = set(node_labels) - set(df.columns)
    if missing_rows or missing_cols:
        raise ValueError(f'Missing rows: {missing_rows}, Missing columns: {missing_cols} in distance matrix')
    distance = {}
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            try:
                val = df.loc[str(i), str(j)]
            except KeyError:
                raise ValueError(f'Missing distance entry for ({i},{j})')
            if pd.isna(val):
                val = df.loc[str(j), str(i)]
            if pd.isna(val):
                raise ValueError(f'Missing symmetric distance for ({i},{j}) and ({j},{i})')
            distance[i, j] = float(val)
    m = gp.Model('TSP15')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in nodes for j in nodes if i != j]
    x = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
    u_nodes = [i for i in nodes if i != 1]
    u = m.addVars(u_nodes, vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
    m.setObjective(gp.quicksum((distance[i, j] * x[i, j] for (i, j) in x_keys)), gp.GRB.MINIMIZE)
    for i in nodes:
        m.addConstr(gp.quicksum((x[i, j] for j in nodes if j != i)) == 1, name='out_%d' % i)
    for j in nodes:
        m.addConstr(gp.quicksum((x[i, j] for i in nodes if i != j)) == 1, name='in_%d' % j)
    for i in u_nodes:
        for j in u_nodes:
            if i == j:
                continue
            m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name='mtz_%d_%d' % (i, j))
    m.optimize()
    return m
m = solve_tsp()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')