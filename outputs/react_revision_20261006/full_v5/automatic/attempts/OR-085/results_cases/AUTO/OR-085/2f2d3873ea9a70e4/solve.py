import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',')
location_ids = df['Unnamed: 0'].astype(str).tolist()
col_ids = [col for col in df.columns if col != 'Unnamed: 0']
if set(location_ids) != set(col_ids):
    raise ValueError('Mismatch between row and column location IDs in distance matrix.')
location_ids = sorted(location_ids, key=lambda x: int(x))
n = len(location_ids)
distance = {}
for i in location_ids:
    distance[i] = {}
    for j in location_ids:
        if i == j:
            distance[i][j] = 0.0
        else:
            val_ij = df.loc[df['Unnamed: 0'].astype(str) == i, j].values
            val_ji = df.loc[df['Unnamed: 0'].astype(str) == j, i].values
            v_ij = float(val_ij[0]) if len(val_ij) > 0 and (not pd.isnull(val_ij[0])) else None
            v_ji = float(val_ji[0]) if len(val_ji) > 0 and (not pd.isnull(val_ji[0])) else None
            if v_ij is not None and v_ji is not None:
                if abs(v_ij - v_ji) > 1e-06:
                    raise ValueError(f'Distance matrix not symmetric at ({i},{j}): {v_ij} vs {v_ji}')
                distance[i][j] = v_ij
            elif v_ij is not None:
                distance[i][j] = v_ij
            elif v_ji is not None:
                distance[i][j] = v_ji
            else:
                raise ValueError(f'Missing distance for ({i},{j}) in CSV.')
arc_keys = [(i, j) for i in location_ids for j in location_ids if i != j]

def solve_tsp():
    m = gp.Model('TSP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(arc_keys, vtype=gp.GRB.BINARY, name='')
    u = {}
    for i in location_ids:
        if i != '1':
            u[i] = m.addVar(lb=2, ub=n, vtype=gp.GRB.CONTINUOUS, name=f'u_{i}')
    m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for (i, j) in arc_keys)), gp.GRB.MINIMIZE)
    for i in location_ids:
        m.addConstr(gp.quicksum((x[i, j] for j in location_ids if j != i)) == 1, name='leave')
    for j in location_ids:
        m.addConstr(gp.quicksum((x[i, j] for i in location_ids if i != j)) == 1, name='enter')
    for i in location_ids:
        if i == '1':
            continue
        for j in location_ids:
            if j == '1' or i == j:
                continue
            m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name='mtz')
    m.optimize()
    return m
m = solve_tsp()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')