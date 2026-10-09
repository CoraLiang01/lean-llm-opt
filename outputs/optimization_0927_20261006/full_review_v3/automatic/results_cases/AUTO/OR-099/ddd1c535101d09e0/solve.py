import gurobipy as gp
from gurobipy import GRB
warehouses = [{'i': 1, 'f_i': 3000, 'u_i': 180}, {'i': 2, 'f_i': 3200, 'u_i': 160}, {'i': 3, 'f_i': 3100, 'u_i': 200}, {'i': 4, 'f_i': 2800, 'u_i': 150}, {'i': 5, 'f_i': 3500, 'u_i': 170}, {'i': 6, 'f_i': 2700, 'u_i': 190}, {'i': 7, 'f_i': 2900, 'u_i': 160}, {'i': 8, 'f_i': 3050, 'u_i': 175}, {'i': 9, 'f_i': 3100, 'u_i': 170}, {'i': 10, 'f_i': 2200, 'u_i': 180}, {'i': 11, 'f_i': 2890, 'u_i': 190}]
stores = [{'j': 1, 'd_j': 30}, {'j': 2, 'd_j': 40}, {'j': 3, 'd_j': 20}, {'j': 4, 'd_j': 35}, {'j': 5, 'd_j': 20}, {'j': 6, 'd_j': 25}, {'j': 7, 'd_j': 45}, {'j': 8, 'd_j': 38}, {'j': 9, 'd_j': 32}, {'j': 10, 'd_j': 41}, {'j': 11, 'd_j': 44}]
transportation_cost = [[12, 11, 14, 15, 17, 13, 12, 16, 16, 14, 15], [17, 19, 15, 20, 18, 14, 17, 15, 13, 15, 16], [13, 14, 12, 14, 16, 15, 11, 14, 16, 18, 17], [18, 16, 17, 13, 18, 17, 14, 19, 16, 13, 18], [10, 13, 12, 19, 15, 11, 12, 14, 12, 15, 17], [15, 12, 14, 16, 13, 17, 16, 16, 14, 18, 19], [14, 13, 15, 17, 12, 13, 14, 15, 12, 16, 14], [19, 16, 18, 20, 17, 19, 16, 18, 15, 15, 18], [17, 18, 12, 14, 16, 15, 14, 17, 21, 15, 18], [14, 13, 15, 17, 16, 18, 14, 19, 15, 17, 19], [15, 13, 16, 17, 11, 13, 14, 15, 19, 21, 13]]
I = [w['i'] for w in warehouses]
J = [s['j'] for s in stores]
f_i = {w['i']: w['f_i'] for w in warehouses}
u_i = {w['i']: w['u_i'] for w in warehouses}
d_j = {s['j']: s['d_j'] for s in stores}
c_ij = {(I[i], J[j]): transportation_cost[i][j] for i in range(len(I)) for j in range(len(J))}
if set(f_i.keys()) != set(I):
    raise ValueError('Missing warehouse opening costs for some warehouses.')
if set(u_i.keys()) != set(I):
    raise ValueError('Missing warehouse capacities for some warehouses.')
if set(d_j.keys()) != set(J):
    raise ValueError('Missing store demands for some stores.')
for i in I:
    for j in J:
        if (i, j) not in c_ij:
            raise ValueError(f'Missing transportation cost for warehouse {i}, store {j}.')
m = gp.Model('Warehouse_Location')
x_vars = m.addVars(I, J, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(I, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((c_ij[i, j] * x_vars[i, j] for i in I for j in J)) + gp.quicksum((f_i[i] * y_vars[i] for i in I)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in I)) == d_j[j] for j in J), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in J)) <= u_i[i] * y_vars[i] for i in I), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')