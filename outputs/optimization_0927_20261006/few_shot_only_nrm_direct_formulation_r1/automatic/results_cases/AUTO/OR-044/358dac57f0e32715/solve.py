import gurobipy as gp
from gurobipy import GRB
sections = ['1', '2', '3', '4', '5', '6', '7', '8']
products = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
capacity = {'1': 100, '2': 150, '3': 120, '4': 130, '5': 90, '6': 110, '7': 160, '8': 140}
value = {'1': 10, '2': 15, '3': 8, '4': 12, '5': 20, '6': 25, '7': 5, '8': 30, '9': 18, '10': 22}
weight = {'1': 2, '2': 3, '3': 1, '4': 2, '5': 4, '6': 5, '7': 1, '8': 6, '9': 3, '10': 4}
if set(sections) != set(capacity.keys()):
    raise ValueError('Section identifiers in capacity do not match sections list.')
if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
    raise ValueError('Product identifiers in value/weight do not match products list.')
m = gp.Model('Supermarket_Section_Stocking')
x_vars = m.addVars(sections, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in sections for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x_vars[i, j] for j in products)) <= capacity[i] for i in sections), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')