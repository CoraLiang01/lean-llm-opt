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
            row = df[df['Unnamed: 0'] == int(i)]
            val_ij = row[j].values[0] if not row.empty and (not pd.isna(row[j].values[0])) else None
            row_t = df[df['Unnamed: 0'] == int(j)]
            val_ji = row_t[i].values[0] if not row_t.empty and (not pd.isna(row_t[i].values[0])) else None
            if val_ij is not None and val_ji is not None:
                if abs(val_ij - val_ji) > 1e-06:
                    raise ValueError(f'Distance matrix not symmetric at ({i},{j}): {val_ij} vs {val_ji}')
                d[i, j] = float(val_ij)
            elif val_ij is not None:
                d[i, j] = float(val_ij)
            elif val_ji is not None:
                d[i, j] = float(val_ji)
            else:
                raise ValueError(f'Missing distance for ({i},{j}) and ({j},{i})')
n = len(location_ids)
nodes = location_ids
arcs = [(i, j) for i in nodes for j in nodes if i != j]
u_nodes = [i for i in nodes if i != '1']

def solve_tsp():
    m = gp.Model('TSP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(arcs, vtype=gp.GRB.BINARY, name='')
    u = m.addVars(u_nodes, vtype=gp.GRB.INTEGER, lb=2, ub=n, name='')
    m.setObjective(gp.quicksum((d[i, j] * x[i, j] for (i, j) in arcs)), gp.GRB.MINIMIZE)
    for i in nodes:
        m.addConstr(gp.quicksum((x[i, j] for j in nodes if j != i)) == 1, name=f'leave_{i}')
    for j in nodes:
        m.addConstr(gp.quicksum((x[i, j] for i in nodes if i != j)) == 1, name=f'enter_{j}')
    for i in u_nodes:
        for j in u_nodes:
            if i != j:
                m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f'subtour_{i}_{j}')
    m.optimize()
    return m
m = solve_tsp()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')