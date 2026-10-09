import gurobipy as gp
from gurobipy import GRB
shelves = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
products = list(range(1, 21))
C = {1: 750, 2: 820, 3: 570, 4: 800, 5: 550, 6: 900, 7: 650, 8: 800, 9: 850, 10: 900}
v = {1: 55, 2: 75, 3: 65, 4: 60, 5: 80, 6: 90, 7: 40, 8: 100, 9: 55, 10: 75, 11: 110, 12: 50, 13: 60, 14: 120, 15: 70, 16: 110, 17: 50, 18: 60, 19: 120, 20: 100}
w = {1: 10, 2: 20, 3: 5, 4: 15, 5: 25, 6: 35, 7: 45, 8: 55, 9: 65, 10: 20, 11: 18, 12: 28, 13: 8, 14: 28, 15: 25, 16: 40, 17: 55, 18: 70, 19: 85, 20: 100}
if set(C.keys()) != set(shelves):
    raise ValueError('Shelf capacity data missing or mismatched.')
if set(v.keys()) != set(products) or set(w.keys()) != set(products):
    raise ValueError('Product value/weight data missing or mismatched.')
m = gp.Model('BigMart_Shelf_Allocation')
x_vars = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((v[j] * x_vars[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((w[j] * x_vars[i, j] for j in products)) <= C[i] for i in shelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')