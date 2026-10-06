import gurobipy as gp
import pandas as pd
import numpy as np
import math
distance_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(distance_path, sep=',')
location_ids = [str(i) for i in range(1, 16)]
if not set(location_ids).issubset(df.columns):
    missing_cols = set(location_ids) - set(df.columns)
    raise ValueError(f'Missing columns in distance matrix: {missing_cols}')
if not set(range(1, 16)).issubset(df['Unnamed: 0'].astype(int)):
    missing_rows = set(range(1, 16)) - set(df['Unnamed: 0'].astype(int))
    raise ValueError(f'Missing rows in distance matrix: {missing_rows}')
d = {}
for i in location_ids:
    for j in location_ids:
        if i == j:
            d[i, j] = 0.0
        else:
            val_ij = df.loc[df['Unnamed: 0'] == int(i), j].values
            val_ji = df.loc[df['Unnamed: 0'] == int(j), i].values
            v_ij = float(val_ij[0]) if len(val_ij) > 0 and (not pd.isnull(val_ij[0])) else None
            v_ji = float(val_ji[0]) if len(val_ji) > 0 and (not pd.isnull(val_ji[0])) else None
            if v_ij is not None and v_ji is not None:
                if abs(v_ij - v_ji) > 1e-06:
                    raise ValueError(f'Distance matrix not symmetric at ({i},{j}): {v_ij} vs {v_ji}')
                d[i, j] = v_ij
            elif v_ij is not None:
                d[i, j] = v_ij
            elif v_ji is not None:
                d[i, j] = v_ji
            else:
                raise ValueError(f'Missing distance for ({i},{j}) and ({j},{i})')
x_keys = [(i, j) for i in location_ids for j in location_ids if i != j]
u_keys = [i for i in location_ids]

def solve_tsp():
    m = gp.Model('TSP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
    u = m.addVars(u_keys, vtype=gp.GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((d[i, j] * x[i, j] for (i, j) in x_keys)), gp.GRB.MINIMIZE)
    for i in location_ids:
        m.addConstr(gp.quicksum((x[i, j] for j in location_ids if j != i)) == 1, name=f'leave_{i}')
    for j in location_ids:
        m.addConstr(gp.quicksum((x[i, j] for i in location_ids if i != j)) == 1, name=f'enter_{j}')
    m.addConstr(u['1'] == 1, name='u1_fixed')
    for i in location_ids:
        if i != '1':
            m.addConstr(u[i] >= 2, name=f'u_lb_{i}')
            m.addConstr(u[i] <= 15, name=f'u_ub_{i}')
    for i in location_ids:
        if i == '1':
            continue
        for j in location_ids:
            if j == '1' or i == j:
                continue
            m.addConstr(u[i] - u[j] + 15 * x[i, j] <= 14, name=f'mtz_{i}_{j}')
    m.optimize()
    return m
m = solve_tsp()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')