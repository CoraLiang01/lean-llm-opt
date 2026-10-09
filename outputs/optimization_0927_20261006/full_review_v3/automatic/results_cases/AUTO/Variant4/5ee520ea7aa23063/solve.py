import gurobipy as gp
from gurobipy import GRB
centers = ['SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6', 'SC7', 'SC8']
districts = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
opening_cost = {'SC1': 12, 'SC2': 15, 'SC3': 18, 'SC4': 10, 'SC5': 14, 'SC6': 13, 'SC7': 16, 'SC8': 11}
coverage = {'SC1': ['D1', 'D2', 'D4'], 'SC2': ['D2', 'D3', 'D5'], 'SC3': ['D4', 'D5', 'D6'], 'SC4': ['D6', 'D7'], 'SC5': ['D7', 'D8', 'D10'], 'SC6': ['D8', 'D9'], 'SC7': ['D1', 'D9', 'D10'], 'SC8': ['D3', 'D4', 'D8']}
a = {}
for i in centers:
    for j in districts:
        a[i, j] = 1 if j in coverage[i] else 0
for i in centers:
    if i not in opening_cost:
        raise ValueError(f'Missing opening cost for center {i}')
    for j in districts:
        if (i, j) not in a:
            raise ValueError(f'Missing coverage entry for center {i}, district {j}')
m = gp.Model('ServiceCenterSetCover')
y_vars = m.addVars(centers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y_vars[i] for i in centers)), GRB.MINIMIZE)
for j in districts:
    m.addConstr(gp.quicksum((a[i, j] * y_vars[i] for i in centers)) >= 1, name=f'cov_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')