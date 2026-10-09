import gurobipy as gp
import pandas as pd
import numpy as np
import math

def solve_tsp():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
    df = pd.read_csv(path, sep=',')
    loc_ids = [str(i) for i in range(1, 16)]
    n = len(loc_ids)
    distance = {}
    for i in loc_ids:
        distance[i] = {}
        for j in loc_ids:
            distance[i][j] = None
    df_rows = df.set_index('Unnamed: 0')
    for i in loc_ids:
        if int(i) not in df_rows.index:
            raise ValueError(f'Missing row for location {i} in CSV')
        for j in loc_ids:
            if i == j:
                distance[i][j] = 0.0
                continue
            val_ij = df_rows.at[int(i), j] if j in df_rows.columns else None
            val_ji = df_rows.at[int(j), i] if i in df_rows.columns and int(j) in df_rows.index else None
            if pd.notnull(val_ij):
                distance[i][j] = float(val_ij)
            elif pd.notnull(val_ji):
                distance[i][j] = float(val_ji)
            else:
                raise ValueError(f'Missing distance between locations {i} and {j} in CSV')
    for i in loc_ids:
        for j in loc_ids:
            if i != j:
                if not math.isclose(distance[i][j], distance[j][i], rel_tol=1e-08):
                    raise ValueError(f'Distance matrix not symmetric at ({i},{j}) and ({j},{i})')
    m = gp.Model('TSP')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in loc_ids for j in loc_ids if i != j]
    x = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
    u = m.addVars([i for i in loc_ids if i != '1'], vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
    m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for (i, j) in x_keys)), gp.GRB.MINIMIZE)
    for i in loc_ids:
        m.addConstr(gp.quicksum((x[i, j] for j in loc_ids if j != i)) == 1, name='deg_out_' + i)
    for j in loc_ids:
        m.addConstr(gp.quicksum((x[i, j] for i in loc_ids if i != j)) == 1, name='deg_in_' + j)
    for i in loc_ids:
        if i == '1':
            continue
        for j in loc_ids:
            if j == '1' or i == j:
                continue
            m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name='mtz_%s_%s' % (i, j))
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for (i, j) in x_keys:
            if x[i, j].X > 0.5:
                print(f'x[{i},{j}] = 1')
        for i in [i for i in loc_ids if i != '1']:
            print(f'u[{i}] = {u[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_tsp()