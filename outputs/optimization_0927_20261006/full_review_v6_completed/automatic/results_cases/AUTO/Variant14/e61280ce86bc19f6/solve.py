import gurobipy as gp
from gurobipy import GRB
depots = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
service_zones = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10']
opening_cost = {'B1': 11, 'B2': 14, 'B3': 10, 'B4': 13, 'B5': 16, 'B6': 9, 'B7': 12, 'B8': 15}
coverage = {'Z1': ['B1', 'B4'], 'Z2': ['B1', 'B2'], 'Z3': ['B2', 'B5'], 'Z4': ['B3', 'B7'], 'Z5': ['B1', 'B3', 'B8'], 'Z6': ['B2', 'B4', 'B8'], 'Z7': ['B4', 'B5'], 'Z8': ['B3', 'B6'], 'Z9': ['B5', 'B6', 'B8'], 'Z10': ['B6', 'B7']}
for i in depots:
    if i not in opening_cost:
        raise ValueError(f'Missing opening cost for depot {i}')
for j in service_zones:
    if j not in coverage:
        raise ValueError(f'Missing coverage set for service zone {j}')
    for i in coverage[j]:
        if i not in depots:
            raise ValueError(f'Depot {i} in coverage for {j} not in depots list')
m = gp.Model('Depot_Set_Covering')
y_vars = m.addVars(depots, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((opening_cost[i] * y_vars[i] for i in depots)), GRB.MINIMIZE)
for j in service_zones:
    m.addConstr(gp.quicksum((y_vars[i] for i in coverage[j])) >= 1, name=f'cov_{j}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')