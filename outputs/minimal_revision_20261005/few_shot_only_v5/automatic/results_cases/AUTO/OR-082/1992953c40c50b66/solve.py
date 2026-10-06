import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_tsp():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others5/DistanceMatrix.csv'
    df = pd.read_csv(path, sep=',')
    node_names = ['Depot', 'A', 'B', 'C']
    df_nodes = df['Unnamed: 0'].astype(str).tolist()
    node_set = set(node_names)
    node_indices = [i for (i, name) in enumerate(df_nodes) if name in node_set]
    if len(node_indices) != len(node_names):
        missing = node_set - set(df_nodes)
        raise ValueError(f'Missing required nodes in CSV: {missing}')
    distance = {}
    for i_name in node_names:
        i_idx = df_nodes.index(i_name)
        for j_name in node_names:
            if j_name not in df.columns:
                raise ValueError(f'Missing column for node {j_name} in CSV')
            val = df.at[i_idx, j_name]
            try:
                dist = float(val)
            except Exception:
                raise ValueError(f'Non-numeric distance for ({i_name},{j_name}): {val}')
            distance[i_name, j_name] = dist
    arcs = [(i, j) for i in node_names for j in node_names if i != j]
    m = gp.Model('TSP4')
    x = m.addVars(arcs, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((distance[i, j] * x[i, j] for (i, j) in arcs)), gp.GRB.MINIMIZE)
    for k in node_names:
        m.addConstr(gp.quicksum((x[k, j] for j in node_names if j != k)) == 1, name=f'out_{k}')
    for k in node_names:
        m.addConstr(gp.quicksum((x[i, k] for i in node_names if i != k)) == 1, name=f'in_{k}')
    customers = [n for n in node_names if n != 'Depot']
    u = m.addVars(customers, lb=1, ub=len(customers), vtype=gp.GRB.CONTINUOUS, name='')
    n_cust = len(customers)
    for i in customers:
        for j in customers:
            if i != j:
                m.addConstr(u[i] - u[j] + n_cust * x[i, j] <= n_cust - 1, name=f'mtz_{i}_{j}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_tsp()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')