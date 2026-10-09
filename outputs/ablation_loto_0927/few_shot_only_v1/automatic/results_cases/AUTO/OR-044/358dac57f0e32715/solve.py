import gurobipy as gp
from gurobipy import GRB
sections = ['1', '2', '3', '4', '5', '6', '7', '8']
products = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
section_capacity = {'1': 100, '2': 150, '3': 120, '4': 130, '5': 90, '6': 110, '7': 160, '8': 140}
product_value = {'1': 10, '2': 15, '3': 8, '4': 12, '5': 20, '6': 25, '7': 5, '8': 30, '9': 18, '10': 22}
product_weight = {'1': 2, '2': 3, '3': 1, '4': 2, '5': 4, '6': 5, '7': 1, '8': 6, '9': 3, '10': 4}
for s in sections:
    if s not in section_capacity:
        raise ValueError(f'Missing capacity for section {s}')
for p in products:
    if p not in product_value:
        raise ValueError(f'Missing value for product {p}')
    if p not in product_weight:
        raise ValueError(f'Missing weight for product {p}')
m = gp.Model('Supermarket_Section_Allocation')
x = m.addVars(sections, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((product_value[p] * x[s, p] for s in sections for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((product_weight[p] * x[s, p] for p in products)) <= section_capacity[s] for s in sections), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')