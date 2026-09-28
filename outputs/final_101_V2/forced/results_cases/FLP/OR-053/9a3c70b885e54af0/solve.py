import gurobipy as gp
from gurobipy import GRB
shelves = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
products = list(range(1, 21))
capacity = {1: 500, 2: 700, 3: 600, 4: 800, 5: 550, 6: 900, 7: 650, 8: 750, 9: 820, 10: 570}
value = {1: 50, 2: 70, 3: 30, 4: 60, 5: 80, 6: 90, 7: 40, 8: 100, 9: 55, 10: 75, 11: 65, 12: 95, 13: 45, 14: 85, 15: 70, 16: 110, 17: 50, 18: 60, 19: 120, 20: 100}
weight = {1: 10, 2: 20, 3: 5, 4: 15, 5: 25, 6: 30, 7: 12, 8: 35, 9: 10, 10: 20, 11: 18, 12: 28, 13: 8, 14: 22, 15: 25, 16: 40, 17: 14, 18: 16, 19: 50, 20: 30}
if set(shelves) != set(capacity.keys()):
    raise ValueError('Shelf capacity data missing for some shelves.')
if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
    raise ValueError('Product value/weight data missing for some products.')
m = gp.Model('BigMart_Shelf_Allocation')
x = m.addVars(shelves, products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i] for i in shelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')