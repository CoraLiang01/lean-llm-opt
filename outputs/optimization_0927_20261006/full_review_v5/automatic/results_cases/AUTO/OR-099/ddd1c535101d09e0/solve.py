import gurobipy as gp
from gurobipy import GRB
warehouses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
stores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
f = {1: 3000, 2: 3200, 3: 3100, 4: 2800, 5: 3500, 6: 2700, 7: 2900, 8: 3050, 9: 3100, 10: 2200, 11: 2890}
u = {1: 180, 2: 160, 3: 200, 4: 150, 5: 170, 6: 190, 7: 160, 8: 175, 9: 170, 10: 180, 11: 190}
d = {1: 30, 2: 40, 3: 20, 4: 35, 5: 20, 6: 25, 7: 45, 8: 38, 9: 32, 10: 41, 11: 44}
c = {1: {1: 12, 2: 11, 3: 14, 4: 15, 5: 17, 6: 13, 7: 12, 8: 16, 9: 16, 10: 14, 11: 15}, 2: {1: 17, 2: 19, 3: 15, 4: 20, 5: 18, 6: 14, 7: 17, 8: 15, 9: 13, 10: 15, 11: 16}, 3: {1: 13, 2: 14, 3: 12, 4: 14, 5: 16, 6: 15, 7: 11, 8: 14, 9: 16, 10: 18, 11: 17}, 4: {1: 18, 2: 16, 3: 17, 4: 13, 5: 18, 6: 17, 7: 14, 8: 19, 9: 16, 10: 13, 11: 18}, 5: {1: 10, 2: 13, 3: 12, 4: 19, 5: 15, 6: 11, 7: 12, 8: 14, 9: 12, 10: 15, 11: 17}, 6: {1: 15, 2: 12, 3: 14, 4: 16, 5: 13, 6: 17, 7: 16, 8: 16, 9: 14, 10: 18, 11: 19}, 7: {1: 14, 2: 13, 3: 15, 4: 17, 5: 12, 6: 13, 7: 14, 8: 15, 9: 12, 10: 16, 11: 14}, 8: {1: 19, 2: 16, 3: 18, 4: 20, 5: 17, 6: 19, 7: 16, 8: 18, 9: 15, 10: 15, 11: 18}, 9: {1: 17, 2: 18, 3: 12, 4: 14, 5: 16, 6: 15, 7: 14, 8: 17, 9: 21, 10: 15, 11: 18}, 10: {1: 14, 2: 13, 3: 15, 4: 17, 5: 16, 6: 18, 7: 14, 8: 19, 9: 15, 10: 17, 11: 19}, 11: {1: 15, 2: 13, 3: 16, 4: 17, 5: 11, 6: 13, 7: 14, 8: 15, 9: 19, 10: 21, 11: 13}}
for i in warehouses:
    if i not in f or i not in u or i not in c:
        raise ValueError(f'Missing opening cost, capacity, or transportation cost row for warehouse {i}')
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