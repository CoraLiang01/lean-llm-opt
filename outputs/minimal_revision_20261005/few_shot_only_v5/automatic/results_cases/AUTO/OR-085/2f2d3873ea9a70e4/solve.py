import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others7/20.csv'
df = pd.read_csv(csv_path, sep=',')
locations = [str(i) for i in range(1, 16)]
N = locations
d = {}
for i in N:
    d[i] = {}
    row = df[df['Unnamed: 0'].astype(str) == i]
    if row.empty:
        raise ValueError(f'Missing row for location {i} in distance matrix.')
    for j in N:
        try:
            val = float(row.iloc[0][j])
        except KeyError:
            raise ValueError(f'Missing column for location {j} in distance matrix.')
        d[i][j] = val
for i in N:
    for j in N:
        if i == j:
            if d[i][j] != 0:
                raise ValueError(f'Distance from {i} to itself should be 0, got {d[i][j]}')
        elif abs(d[i][j] - d[j][i]) > 1e-06:
            raise ValueError(f'Distance matrix is not symmetric: d[{i}][{j}] != d[{j}][{i}]')
x_keys = [(i, j) for i in N for j in N if i != j]
u_keys = [i for i in N if i != '1']

def solve_tsp(N, d):
    m = gp.Model('TSP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
    u = m.addVars(u_keys, vtype=gp.GRB.INTEGER, lb=2, ub=len(N), name='')
    m.setObjective(gp.quicksum((d[i][j] * x[i, j] for (i, j) in x_keys)), gp.GRB.MINIMIZE)
    for i in N:
        m.addConstr(gp.quicksum((x[i, j] for j in N if j != i)) == 1, name=f'dep_{i}')
    for j in N:
        m.addConstr(gp.quicksum((x[i, j] for i in N if i != j)) == 1, name=f'arr_{j}')
    n = len(N)
    for i in u_keys:
        for j in u_keys:
            if i != j:
                m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f'subtour_{i}_{j}')
    m.optimize()
    return m
m = solve_tsp(N, d)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')