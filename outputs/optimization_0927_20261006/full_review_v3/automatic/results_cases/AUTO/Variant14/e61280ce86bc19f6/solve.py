import gurobipy as gp
from gurobipy import GRB
depots = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
zones = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10']
opening_cost = {'B1': 11, 'B2': 14, 'B3': 10, 'B4': 13, 'B5': 16, 'B6': 9, 'B7': 12, 'B8': 15}
coverage = {'Z1': ['B1', 'B4'], 'Z2': ['B1', 'B2'], 'Z3': ['B2', 'B5'], 'Z4': ['B3', 'B7'], 'Z5': ['B1', 'B3', 'B8'], 'Z6': ['B2', 'B4', 'B8'], 'Z7': ['B4', 'B5'], 'Z8': ['B3', 'B6'], 'Z9': ['B5', 'B6', 'B8'], 'Z10': ['B6', 'B7']}
for d in depots:
    if d not in opening_cost:
        raise ValueError(f'Missing opening cost for depot {d}')
for z in zones:
    if z not in coverage:
        raise ValueError(f'Missing coverage set for zone {z}')
    for d in coverage[z]:
        if d not in depots:
            raise ValueError(f'Depot {d} in coverage of {z} not in depots set')

def build_model():
    m = gp.Model('Depot_Set_Cover')
    y_vars = m.addVars(depots, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((opening_cost[d] * y_vars[d] for d in depots)), GRB.MINIMIZE)
    for z in zones:
        m.addConstr(gp.quicksum((y_vars[d] for d in coverage[z])) >= 1, name=f'cov_{z}')
    return m
m = build_model()
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')