import gurobipy as gp
from gurobipy import GRB
depots = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
zones = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10']
opening_cost = {'B1': 11, 'B2': 14, 'B3': 10, 'B4': 13, 'B5': 16, 'B6': 9, 'B7': 12, 'B8': 15}
depot_covers = {'B1': ['Z1', 'Z2', 'Z5'], 'B2': ['Z2', 'Z3', 'Z6'], 'B3': ['Z4', 'Z5', 'Z8'], 'B4': ['Z1', 'Z6', 'Z7'], 'B5': ['Z3', 'Z7', 'Z9'], 'B6': ['Z8', 'Z9', 'Z10'], 'B7': ['Z4', 'Z10'], 'B8': ['Z5', 'Z6', 'Z9']}
zone_covered_by = {'Z1': ['B1', 'B4'], 'Z2': ['B1', 'B2'], 'Z3': ['B2', 'B5'], 'Z4': ['B3', 'B7'], 'Z5': ['B1', 'B3', 'B8'], 'Z6': ['B2', 'B4', 'B8'], 'Z7': ['B4', 'B5'], 'Z8': ['B3', 'B6'], 'Z9': ['B5', 'B6', 'B8'], 'Z10': ['B6', 'B7']}
for i in depots:
    if i not in opening_cost or i not in depot_covers:
        raise ValueError(f'Missing data for depot {i}')
for j in zones:
    if j not in zone_covered_by:
        raise ValueError(f'Missing coverage data for zone {j}')

def build_model():
    m = gp.Model('Depot_Set_Cover')
    y_vars = m.addVars(depots, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((opening_cost[i] * y_vars[i] for i in depots)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((y_vars[i] for i in zone_covered_by[j])) >= 1 for j in zones), name='')
    m.Params.MIPGap = 0.0001
    return m
m = build_model()
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')