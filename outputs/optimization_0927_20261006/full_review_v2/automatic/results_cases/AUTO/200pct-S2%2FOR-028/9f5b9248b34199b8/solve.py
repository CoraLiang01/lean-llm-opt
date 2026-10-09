import gurobipy as gp
from gurobipy import GRB
warehouses = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
stores = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
f_i = {'1': 3000, '2': 3200, '3': 3100, '4': 2800, '5': 3500, '6': 2700, '7': 2900, '8': 3050, '9': 3100, '10': 2200, '11': 2890}
u_i = {'1': 180, '2': 160, '3': 200, '4': 150, '5': 170, '6': 190, '7': 160, '8': 175, '9': 170, '10': 180, '11': 190}
d_j = {'1': 30, '2': 40, '3': 20, '4': 35, '5': 20, '6': 25, '7': 45, '8': 38, '9': 32, '10': 41, '11': 44}
c_ij_matrix = [[12, 17, 13, 18, 10, 15, 14, 19, 17, 14, 15], [11, 19, 14, 16, 13, 12, 13, 16, 18, 13, 13], [14, 15, 12, 17, 12, 14, 15, 18, 12, 15, 16], [15, 20, 14, 13, 19, 16, 17, 20, 14, 17, 17], [17, 18, 16, 18, 15, 13, 12, 17, 16, 16, 11], [13, 14, 15, 17, 11, 17, 13, 19, 15, 18, 13], [12, 17, 11, 14, 12, 16, 14, 16, 14, 14, 14], [16, 15, 14, 19, 14, 16, 15, 18, 17, 19, 15], [16, 13, 16, 16, 12, 14, 12, 15, 21, 15, 19], [14, 15, 18, 13, 15, 18, 16, 15, 15, 17, 21], [15, 16, 17, 18, 17, 19, 14, 18, 18, 19, 13]]
c_ij = {}
for (i_idx, i) in enumerate(warehouses):
    c_ij[i] = {}
    for (j_idx, j) in enumerate(stores):
        c_ij[i][j] = c_ij_matrix[i_idx][j_idx]
if set(f_i.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse opening cost keys and warehouse list')
if set(u_i.keys()) != set(warehouses):
    raise ValueError('Mismatch in warehouse capacity keys and warehouse list')
if set(d_j.keys()) != set(stores):
    raise ValueError('Mismatch in store demand keys and store list')
if len(c_ij_matrix) != len(warehouses) or any((len(row) != len(stores) for row in c_ij_matrix)):
    raise ValueError('Transportation cost matrix dimensions do not match warehouses and stores')
m = gp.Model('Warehouse_Location')
x_vars = m.addVars(warehouses, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(warehouses, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((c_ij[i][j] * x_vars[i, j] for i in warehouses for j in stores)) + gp.quicksum((f_i[i] * y_vars[i] for i in warehouses)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in warehouses)) == d_j[j] for j in stores), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in stores)) <= u_i[i] * y_vars[i] for i in warehouses), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')