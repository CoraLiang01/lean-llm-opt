import gurobipy as gp
from gurobipy import GRB
displays = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
products = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
capacity = {1: 5.0, 2: 7.0, 3: 6.0, 4: 8.0, 5: 5.5, 6: 9.0, 7: 6.5, 8: 7.5, 9: 8.2, 10: 5.7}
value = {1: 200, 2: 1500, 3: 100, 4: 800, 5: 250, 6: 600, 7: 150, 8: 80, 9: 50, 10: 300, 11: 400, 12: 120, 13: 60, 14: 40, 15: 30, 16: 25, 17: 100, 18: 500, 19: 90, 20: 180}
weight = {1: 1.0, 2: 5.0, 3: 0.5, 4: 2.0, 5: 0.3, 6: 1.5, 7: 1.0, 8: 0.8, 9: 0.2, 10: 3.0, 11: 4.0, 12: 0.5, 13: 0.3, 14: 0.4, 15: 0.05, 16: 0.02, 17: 0.6, 18: 4.0, 19: 0.2, 20: 0.5}
if set(capacity.keys()) != set(displays):
    raise ValueError('Display capacity data missing or extra entries.')
if set(value.keys()) != set(products) or set(weight.keys()) != set(products):
    raise ValueError('Product value/weight data missing or extra entries.')
m = gp.Model('Retail_Display_Allocation')
x = m.addVars(displays, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in displays for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i] for i in displays), name='')
m.addConstr(gp.quicksum((x[i, 1] for i in displays)) >= 5, name='min_smartphone')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')