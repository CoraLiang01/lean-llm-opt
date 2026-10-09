import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_tsp():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
    df = pd.read_csv(path, sep=',', dtype=str, keep_default_na=False)
    node_labels = [col for col in df.columns if col != 'Unnamed: 0']
    nodes = [int(label) for label in node_labels]
    n = len(nodes)
    if n != 15 or sorted(nodes) != list(range(1, 16)):
        raise ValueError('Expected exactly 15 locations labeled 1..15 in columns.')
    distance = {}
    for (idx, row) in df.iterrows():
        i = int(row['Unnamed: 0'])
        for j_label in node_labels:
            j = int(j_label)
            val = row[j_label].strip()
            if val == '':
                continue
            try:
                d = float(val)
            except Exception:
                raise ValueError(f"Non-numeric distance at ({i},{j}): '{val}'")
            distance[i, j] = d
    for i in nodes:
        for j in nodes:
            if i == j:
                distance[i, j] = 0.0
            elif (i, j) not in distance and (j, i) in distance:
                distance[i, j] = distance[j, i]
            elif (i, j) not in distance:
                raise ValueError(f'Missing distance for ({i},{j}) and ({j},{i})')
    x_keys = [(i, j) for i in nodes for j in nodes if i != j]
    u_nodes = [i for i in nodes if i != 1]
    m = gp.Model('TSP15')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
    u_vars = m.addVars(u_nodes, vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
    m.setObjective(gp.quicksum((distance[i, j] * x_vars[i, j] for (i, j) in x_keys)), gp.GRB.MINIMIZE)
    for i in nodes:
        m.addConstr(gp.quicksum((x_vars[i, j] for j in nodes if j != i)) == 1, name='out_%d' % i)
    for j in nodes:
        m.addConstr(gp.quicksum((x_vars[i, j] for i in nodes if i != j)) == 1, name='in_%d' % j)
    for i in u_nodes:
        for j in u_nodes:
            if i != j:
                m.addConstr(u_vars[i] - u_vars[j] + n * x_vars[i, j] <= n - 1, name='mtz_%d_%d' % (i, j))
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for (i, j) in x_keys:
            if x_vars[i, j].X > 0.5:
                print(f'x[{i},{j}] {x_vars[i, j].VarName} {x_vars[i, j].X}')
        for i in u_nodes:
            print(f'u[{i}] {u_vars[i].VarName} {u_vars[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_tsp()