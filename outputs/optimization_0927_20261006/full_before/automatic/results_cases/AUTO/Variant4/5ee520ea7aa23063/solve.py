import gurobipy as gp
from gurobipy import GRB
I = ['SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6', 'SC7', 'SC8']
J = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
c = {'SC1': 12, 'SC2': 15, 'SC3': 18, 'SC4': 10, 'SC5': 14, 'SC6': 13, 'SC7': 16, 'SC8': 11}
a = {'SC1': {'D1': 1, 'D2': 1, 'D3': 0, 'D4': 1, 'D5': 0, 'D6': 0, 'D7': 0, 'D8': 0, 'D9': 0, 'D10': 0}, 'SC2': {'D1': 0, 'D2': 1, 'D3': 1, 'D4': 0, 'D5': 1, 'D6': 0, 'D7': 0, 'D8': 0, 'D9': 0, 'D10': 0}, 'SC3': {'D1': 0, 'D2': 0, 'D3': 0, 'D4': 1, 'D5': 1, 'D6': 1, 'D7': 0, 'D8': 0, 'D9': 0, 'D10': 0}, 'SC4': {'D1': 0, 'D2': 0, 'D3': 0, 'D4': 0, 'D5': 0, 'D6': 1, 'D7': 1, 'D8': 0, 'D9': 0, 'D10': 0}, 'SC5': {'D1': 0, 'D2': 0, 'D3': 0, 'D4': 0, 'D5': 0, 'D6': 0, 'D7': 1, 'D8': 1, 'D9': 0, 'D10': 1}, 'SC6': {'D1': 0, 'D2': 0, 'D3': 0, 'D4': 0, 'D5': 0, 'D6': 0, 'D7': 0, 'D8': 1, 'D9': 1, 'D10': 0}, 'SC7': {'D1': 1, 'D2': 0, 'D3': 0, 'D4': 0, 'D5': 0, 'D6': 0, 'D7': 0, 'D8': 0, 'D9': 1, 'D10': 1}, 'SC8': {'D1': 0, 'D2': 0, 'D3': 1, 'D4': 1, 'D5': 0, 'D6': 0, 'D7': 0, 'D8': 1, 'D9': 0, 'D10': 0}}
for i in I:
    if i not in c:
        raise ValueError(f'Missing opening cost for {i}')
    if i not in a:
        raise ValueError(f'Missing coverage row for {i}')
    for j in J:
        if j not in a[i]:
            raise ValueError(f'Missing coverage entry for ({i},{j})')
m = gp.Model('ServiceCenterSetCover')
y = m.addVars(I, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i] * y[i] for i in I)), GRB.MINIMIZE)
for j in J:
    m.addConstr(gp.quicksum((a[i][j] * y[i] for i in I)) >= 1, name=f'cov_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')