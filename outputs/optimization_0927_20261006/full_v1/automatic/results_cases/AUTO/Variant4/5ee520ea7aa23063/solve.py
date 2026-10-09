import gurobipy as gp
from gurobipy import GRB
centers = ['SC1', 'SC2', 'SC3', 'SC4', 'SC5', 'SC6', 'SC7', 'SC8']
districts = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
opening_cost = {'SC1': 12, 'SC2': 15, 'SC3': 18, 'SC4': 10, 'SC5': 14, 'SC6': 13, 'SC7': 16, 'SC8': 11}
coverage = {'SC1': {'D1': 1, 'D2': 1, 'D3': 0, 'D4': 1, 'D5': 0, 'D6': 0, 'D7': 0, 'D8': 0, 'D9': 0, 'D10': 0}, 'SC2': {'D1': 0, 'D2': 1, 'D3': 1, 'D4': 0, 'D5': 1, 'D6': 0, 'D7': 0, 'D8': 0, 'D9': 0, 'D10': 0}, 'SC3': {'D1': 0, 'D2': 0, 'D3': 0, 'D4': 1, 'D5': 1, 'D6': 1, 'D7': 0, 'D8': 0, 'D9': 0, 'D10': 0}, 'SC4': {'D1': 0, 'D2': 0, 'D3': 0, 'D4': 0, 'D5': 0, 'D6': 1, 'D7': 1, 'D8': 0, 'D9': 0, 'D10': 0}, 'SC5': {'D1': 0, 'D2': 0, 'D3': 0, 'D4': 0, 'D5': 0, 'D6': 0, 'D7': 1, 'D8': 1, 'D9': 0, 'D10': 1}, 'SC6': {'D1': 0, 'D2': 0, 'D3': 0, 'D4': 0, 'D5': 0, 'D6': 0, 'D7': 0, 'D8': 1, 'D9': 1, 'D10': 0}, 'SC7': {'D1': 1, 'D2': 0, 'D3': 0, 'D4': 0, 'D5': 0, 'D6': 0, 'D7': 0, 'D8': 0, 'D9': 1, 'D10': 1}, 'SC8': {'D1': 0, 'D2': 0, 'D3': 1, 'D4': 1, 'D5': 0, 'D6': 0, 'D7': 0, 'D8': 1, 'D9': 0, 'D10': 0}}
for i in centers:
    if i not in opening_cost:
        raise ValueError(f'Missing opening cost for center {i}')
    if i not in coverage:
        raise ValueError(f'Missing coverage row for center {i}')
    for j in districts:
        if j not in coverage[i]:
            raise ValueError(f'Missing coverage entry for center {i}, district {j}')
m = gp.Model('ServiceCenterSetCover')
y_vars = m.addVars(centers, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y_vars[i] for i in centers)), GRB.MINIMIZE)
for j in districts:
    m.addConstr(gp.quicksum((coverage[i][j] * y_vars[i] for i in centers)) >= 1, name=f'cov_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')