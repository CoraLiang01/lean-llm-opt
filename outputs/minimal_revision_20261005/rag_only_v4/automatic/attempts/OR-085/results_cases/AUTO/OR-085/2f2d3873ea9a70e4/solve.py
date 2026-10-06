import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',')
locations = [str(i) for i in range(1, 16)]
distance = {}
for i_row in df.itertuples(index=False):
    i = str(getattr(i_row, 'Unnamed: 0'))
    distance[i] = {}
    for j in locations:
        val = getattr(i_row, j)
        if pd.isna(val):
            distance[i][j] = None
        else:
            distance[i][j] = float(val)
for i in locations:
    for j in locations:
        if i == j:
            distance[i][j] = 0.0
        else:
            dij = distance[i][j]
            dji = distance[j][i]
            if dij is None and dji is not None:
                distance[i][j] = dji
            elif dij is not None and dji is None:
                distance[j][i] = dij
            elif dij is None and dji is None:
                raise ValueError(f'Missing distance between {i} and {j}')
for i in locations:
    for j in locations:
        if distance[i][j] is None:
            raise ValueError(f'Missing distance between {i} and {j}')
        if abs(distance[i][j] - distance[j][i]) > 1e-06:
            raise ValueError(f'Distance matrix not symmetric at ({i},{j})')
x_keys = [(i, j) for i in locations for j in locations if i != j]
u_locs = [i for i in locations if i != '1']
m = gp.Model('tsp')
m.Params.MIPGap = 0.0001
x = m.addVars(x_keys, vtype=GRB.BINARY, lb=0, ub=1, name='')
u = m.addVars(u_locs, vtype=GRB.INTEGER, lb=2, ub=15, name='')
m.setObjective(gp.quicksum((distance[i][j] * x[i, j] for (i, j) in x_keys)), GRB.MINIMIZE)
for i in locations:
    m.addConstr(gp.quicksum((x[i, j] for j in locations if j != i)) == 1, name='')
for j in locations:
    m.addConstr(gp.quicksum((x[i, j] for i in locations if i != j)) == 1, name='')
n = len(locations)
for i in u_locs:
    for j in u_locs:
        if i != j:
            m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')