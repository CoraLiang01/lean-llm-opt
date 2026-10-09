import gurobipy as gp
from gurobipy import GRB
warehouses = [str(i) for i in range(1, 12)]
stores = [str(j) for j in range(1, 12)]
f = [3000, 3200, 3100, 2800, 3500, 2700, 2900, 3050, 3100, 2200, 2890]
K = [180, 160, 200, 150, 170, 190, 160, 175, 170, 180, 190]
d = [30, 40, 20, 35, 20, 25, 45, 38, 32, 41, 44]
C = [[12, 11, 14, 15, 17, 13, 12, 16, 16, 14, 15], [17, 19, 15, 20, 18, 14, 17, 15, 13, 15, 16], [13, 14, 12, 14, 16, 15, 11, 14, 16, 18, 17], [18, 16, 17, 13, 18, 17, 14, 19, 16, 13, 18], [10, 13, 12, 19, 15, 11, 12, 14, 12, 15, 17], [15, 12, 14, 16, 13, 17, 16, 16, 14, 18, 19], [14, 13, 15, 17, 12, 13, 14, 15, 12, 16, 14], [19, 16, 18, 20, 17, 19, 16, 18, 15, 15, 18], [17, 18, 12, 14, 16, 15, 14, 17, 21, 15, 18], [14, 13, 15, 17, 16, 18, 14, 19, 15, 17, 19], [15, 13, 16, 17, 11, 13, 14, 15, 19, 21, 13]]
if not len(f) == len(K) == len(warehouses) == len(C):
    raise ValueError('Mismatch in number of warehouses')
if not len(d) == len(stores):
    raise ValueError('Mismatch in number of stores')
for row in C:
    if len(row) != len(stores):
        raise ValueError('Each row of C must have length equal to number of stores')
cost = {}
for (i_idx, i) in enumerate(warehouses):
    cost[i] = {}
    for (j_idx, j) in enumerate(stores):
        cost[i][j] = C[i_idx][j_idx]
fixed_cost = {warehouses[i]: f[i] for i in range(len(warehouses))}
capacity = {warehouses[i]: K[i] for i in range(len(warehouses))}
demand = {stores[j]: d[j] for j in range(len(stores))}
m = gp.Model('Warehouse_Location')
x_vars = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in warehouses)) + gp.quicksum((cost[i][j] * x_vars[i, j] for i in warehouses for j in stores)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouses)) == demand[j] for j in stores), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in stores)) <= capacity[i] * y_vars[i] for i in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')