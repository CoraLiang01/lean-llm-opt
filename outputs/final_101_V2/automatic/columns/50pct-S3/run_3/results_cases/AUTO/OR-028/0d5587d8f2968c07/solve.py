import gurobipy as gp
from gurobipy import GRB
warehouses = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11']
stores = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11']
f = {'W1': 3000, 'W2': 3200, 'W3': 3100, 'W4': 2800, 'W5': 3500, 'W6': 2700, 'W7': 2900, 'W8': 3050, 'W9': 3100, 'W10': 2200, 'W11': 2890}
cap = {'W1': 180, 'W2': 160, 'W3': 200, 'W4': 150, 'W5': 170, 'W6': 190, 'W7': 160, 'W8': 175, 'W9': 170, 'W10': 180, 'W11': 190}
d = {'W1': 30, 'W2': 40, 'W3': 20, 'W4': 35, 'W5': 20, 'W6': 25, 'W7': 45, 'W8': 38, 'W9': 32, 'W10': 41, 'W11': 44}
c = {'W1': {'W1': 12, 'W2': 11, 'W3': 14, 'W4': 15, 'W5': 17, 'W6': 13, 'W7': 12, 'W8': 16, 'W9': 16, 'W10': 14, 'W11': 15}, 'W2': {'W1': 17, 'W2': 19, 'W3': 15, 'W4': 20, 'W5': 18, 'W6': 14, 'W7': 17, 'W8': 15, 'W9': 13, 'W10': 15, 'W11': 16}, 'W3': {'W1': 13, 'W2': 14, 'W3': 12, 'W4': 14, 'W5': 16, 'W6': 15, 'W7': 11, 'W8': 14, 'W9': 16, 'W10': 18, 'W11': 17}, 'W4': {'W1': 18, 'W2': 16, 'W3': 17, 'W4': 13, 'W5': 18, 'W6': 17, 'W7': 14, 'W8': 19, 'W9': 16, 'W10': 13, 'W11': 18}, 'W5': {'W1': 10, 'W2': 13, 'W3': 12, 'W4': 19, 'W5': 15, 'W6': 11, 'W7': 12, 'W8': 14, 'W9': 12, 'W10': 15, 'W11': 17}, 'W6': {'W1': 15, 'W2': 12, 'W3': 14, 'W4': 16, 'W5': 13, 'W6': 17, 'W7': 16, 'W8': 16, 'W9': 14, 'W10': 18, 'W11': 19}, 'W7': {'W1': 14, 'W2': 13, 'W3': 15, 'W4': 17, 'W5': 12, 'W6': 13, 'W7': 14, 'W8': 15, 'W9': 12, 'W10': 16, 'W11': 14}, 'W8': {'W1': 19, 'W2': 16, 'W3': 18, 'W4': 20, 'W5': 17, 'W6': 19, 'W7': 16, 'W8': 18, 'W9': 15, 'W10': 15, 'W11': 18}, 'W9': {'W1': 17, 'W2': 18, 'W3': 12, 'W4': 14, 'W5': 16, 'W6': 15, 'W7': 14, 'W8': 17, 'W9': 21, 'W10': 15, 'W11': 18}, 'W10': {'W1': 14, 'W2': 13, 'W3': 15, 'W4': 17, 'W5': 16, 'W6': 18, 'W7': 14, 'W8': 19, 'W9': 15, 'W10': 17, 'W11': 19}, 'W11': {'W1': 15, 'W2': 13, 'W3': 16, 'W4': 17, 'W5': 11, 'W6': 13, 'W7': 14, 'W8': 15, 'W9': 19, 'W10': 21, 'W11': 13}}
for i in warehouses:
    if i not in f or i not in cap or i not in c:
        raise ValueError(f'Missing data for warehouse {i}')
    for j in stores:
        if j not in c[i]:
            raise ValueError(f'Missing transportation cost for ({i},{j})')
for j in stores:
    if j not in d:
        raise ValueError(f'Missing demand for store {j}')
m = gp.Model('Warehouse_Location')
x = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
y = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i][j] * x[i, j] for i in warehouses for j in stores)) + gp.quicksum((f[i] * y[i] for i in warehouses)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in warehouses)) == d[j] for j in stores), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in stores)) <= cap[i] * y[i] for i in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')