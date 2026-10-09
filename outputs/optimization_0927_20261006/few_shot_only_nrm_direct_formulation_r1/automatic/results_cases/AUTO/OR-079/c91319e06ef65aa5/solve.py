import gurobipy as gp
from gurobipy import GRB
factories = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15']
dcs = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
fixed_cost = {'A1': 0, 'A2': 175, 'A3': 300, 'A4': 375, 'A5': 500, 'A6': 200, 'A7': 260, 'A8': 220, 'A9': 320, 'A10': 280, 'A11': 350, 'A12': 420, 'A13': 470, 'A14': 520, 'A15': 560}
capacity = {'A1': 30, 'A2': 10, 'A3': 20, 'A4': 30, 'A5': 40, 'A6': 20, 'A7': 25, 'A8': 30, 'A9': 35, 'A10': 20, 'A11': 40, 'A12': 25, 'A13': 30, 'A14': 50, 'A15': 45}
demand = {'B1': 30, 'B2': 25, 'B3': 20, 'B4': 35, 'B5': 25, 'B6': 30, 'B7': 25, 'B8': 30}
shipping_cost = {'A1': {'B1': 8, 'B2': 4, 'B3': 3, 'B4': 6, 'B5': 7, 'B6': 5, 'B7': 9, 'B8': 8}, 'A2': {'B1': 5, 'B2': 2, 'B3': 3, 'B4': 5, 'B5': 6, 'B6': 4, 'B7': 7, 'B8': 6}, 'A3': {'B1': 4, 'B2': 3, 'B3': 4, 'B4': 6, 'B5': 5, 'B6': 5, 'B7': 6, 'B8': 7}, 'A4': {'B1': 9, 'B2': 7, 'B3': 5, 'B4': 8, 'B5': 9, 'B6': 6, 'B7': 10, 'B8': 7}, 'A5': {'B1': 10, 'B2': 4, 'B3': 2, 'B4': 6, 'B5': 8, 'B6': 5, 'B7': 7, 'B8': 3}, 'A6': {'B1': 6, 'B2': 5, 'B3': 4, 'B4': 5, 'B5': 7, 'B6': 6, 'B7': 8, 'B8': 5}, 'A7': {'B1': 7, 'B2': 6, 'B3': 5, 'B4': 4, 'B5': 6, 'B6': 7, 'B7': 9, 'B8': 6}, 'A8': {'B1': 5, 'B2': 4, 'B3': 6, 'B4': 3, 'B5': 5, 'B6': 6, 'B7': 7, 'B8': 6}, 'A9': {'B1': 8, 'B2': 7, 'B3': 6, 'B4': 7, 'B5': 9, 'B6': 8, 'B7': 10, 'B8': 7}, 'A10': {'B1': 6, 'B2': 5, 'B3': 7, 'B4': 4, 'B5': 6, 'B6': 5, 'B7': 7, 'B8': 5}, 'A11': {'B1': 9, 'B2': 6, 'B3': 4, 'B4': 6, 'B5': 8, 'B6': 7, 'B7': 9, 'B8': 6}, 'A12': {'B1': 7, 'B2': 5, 'B3': 6, 'B4': 5, 'B5': 6, 'B6': 5, 'B7': 8, 'B8': 5}, 'A13': {'B1': 8, 'B2': 6, 'B3': 5, 'B4': 6, 'B5': 7, 'B6': 6, 'B7': 8, 'B8': 7}, 'A14': {'B1': 9, 'B2': 5, 'B3': 3, 'B4': 5, 'B5': 7, 'B6': 4, 'B7': 6, 'B8': 4}, 'A15': {'B1': 10, 'B2': 6, 'B3': 4, 'B4': 5, 'B5': 8, 'B6': 5, 'B7': 7, 'B8': 5}}
for i in factories:
    if i not in fixed_cost or i not in capacity or i not in shipping_cost:
        raise ValueError(f'Missing data for factory {i}')
    for j in dcs:
        if j not in shipping_cost[i]:
            raise ValueError(f'Missing shipping cost for ({i},{j})')
for j in dcs:
    if j not in demand:
        raise ValueError(f'Missing demand for DC {j}')
m = gp.Model('ElectroTech_Facility_Location')
x_vars = m.addVars(factories, dcs, lb=0, vtype=GRB.CONTINUOUS, name='')
y_vars = m.addVars(factories, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y_vars[i] for i in factories)) + gp.quicksum((shipping_cost[i][j] * x_vars[i, j] for i in factories for j in dcs)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x_vars[i, j] for i in factories)) == demand[j] for j in dcs), name='')
m.addConstrs((gp.quicksum((x_vars[i, j] for j in dcs)) <= capacity[i] * y_vars[i] for i in factories), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')