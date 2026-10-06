import gurobipy as gp
from gurobipy import GRB
sections = ['Section 1', 'Section 2', 'Section 3', 'Section 4', 'Section 5', 'Section 6', 'Section 7', 'Section 8']
products = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
capacity = {'Section 1': 100, 'Section 2': 150, 'Section 3': 120, 'Section 4': 130, 'Section 5': 90, 'Section 6': 110, 'Section 7': 160, 'Section 8': 140}
value = {'1': 10, '2': 15, '3': 8, '4': 12, '5': 20, '6': 25, '7': 5, '8': 30, '9': 18, '10': 22}
weight = {'1': 2, '2': 3, '3': 1, '4': 2, '5': 4, '6': 5, '7': 1, '8': 6, '9': 3, '10': 4}
for s in sections:
    if s not in capacity:
        raise ValueError(f'Missing capacity for {s}')
for p in products:
    if p not in value or p not in weight:
        raise ValueError(f'Missing value or weight for product {p}')
m = gp.Model('Supermarket_Section_Stocking')
x = m.addVars(sections, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x[s, p] for s in sections for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[p] * x[s, p] for p in products)) <= capacity[s] for s in sections), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')