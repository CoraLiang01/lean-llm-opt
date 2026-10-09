import gurobipy as gp
from gurobipy import GRB
warehouses = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11']
stores = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']
f = {'W1': 3000, 'W2': 3200, 'W3': 3100, 'W4': 2800, 'W5': 3500, 'W6': 2700, 'W7': 2900, 'W8': 3050, 'W9': 3100, 'W10': 2200, 'W11': 2890}
u = {'W1': 180, 'W2': 160, 'W3': 200, 'W4': 150, 'W5': 170, 'W6': 190, 'W7': 160, 'W8': 175, 'W9': 170, 'W10': 180, 'W11': 190}
d = {'S1': 30, 'S2': 40, 'S3': 20, 'S4': 35, 'S5': 20, 'S6': 25, 'S7': 45, 'S8': 38, 'S9': 32, 'S10': 41, 'S11': 44}
c = {'W1': {'S1': 12, 'S2': 17, 'S3': 13, 'S4': 18, 'S5': 10, 'S6': 15, 'S7': 14, 'S8': 19, 'S9': 17, 'S10': 14, 'S11': 15}, 'W2': {'S1': 11, 'S2': 19, 'S3': 14, 'S4': 16, 'S5': 13, 'S6': 12, 'S7': 13, 'S8': 16, 'S9': 18, 'S10': 13, 'S11': 13}, 'W3': {'S1': 14, 'S2': 15, 'S3': 12, 'S4': 17, 'S5': 12, 'S6': 14, 'S7': 15, 'S8': 18, 'S9': 12, 'S10': 15, 'S11': 16}, 'W4': {'S1': 15, 'S2': 20, 'S3': 14, 'S4': 13, 'S5': 19, 'S6': 16, 'S7': 17, 'S8': 20, 'S9': 14, 'S10': 17, 'S11': 17}, 'W5': {'S1': 17, 'S2': 18, 'S3': 16, 'S4': 18, 'S5': 15, 'S6': 13, 'S7': 12, 'S8': 17, 'S9': 16, 'S10': 16, 'S11': 11}, 'W6': {'S1': 13, 'S2': 14, 'S3': 15, 'S4': 17, 'S5': 11, 'S6': 17, 'S7': 13, 'S8': 19, 'S9': 15, 'S10': 18, 'S11': 13}, 'W7': {'S1': 12, 'S2': 17, 'S3': 11, 'S4': 14, 'S5': 12, 'S6': 16, 'S7': 14, 'S8': 16, 'S9': 14, 'S10': 14, 'S11': 14}, 'W8': {'S1': 16, 'S2': 15, 'S3': 14, 'S4': 19, 'S5': 14, 'S6': 16, 'S7': 15, 'S8': 18, 'S9': 17, 'S10': 19, 'S11': 15}, 'W9': {'S1': 16, 'S2': 13, 'S3': 16, 'S4': 16, 'S5': 12, 'S6': 14, 'S7': 12, 'S8': 15, 'S9': 21, 'S10': 15, 'S11': 19}, 'W10': {'S1': 14, 'S2': 15, 'S3': 18, 'S4': 13, 'S5': 15, 'S6': 18, 'S7': 16, 'S8': 15, 'S9': 15, 'S10': 17, 'S11': 21}, 'W11': {'S1': 15, 'S2': 16, 'S3': 17, 'S4': 18, 'S5': 17, 'S6': 19, 'S7': 14, 'S8': 18, 'S9': 18, 'S10': 19, 'S11': 13}}
for i in warehouses:
    if i not in f or i not in u or i not in c:
        raise ValueError(f'Missing data for warehouse {i}')
    for j in stores:
        if j not in c[i]:
            raise ValueError(f'Missing transportation cost for warehouse {i}, store {j}')
for j in stores:
    if j not in d:
        raise ValueError(f'Missing demand for store {j}')
m = gp.Model('Warehouse_Location')
x_vars = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((f[i] * y_vars[i] for i in warehouses)) + gp.quicksum((c[i][j] * x_vars[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouses)) == d[j] for j in stores), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in stores)) <= u[i] * y_vars[i] for i in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')