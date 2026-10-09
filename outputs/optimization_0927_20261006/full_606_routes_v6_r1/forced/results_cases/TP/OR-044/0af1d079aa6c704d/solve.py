import gurobipy as gp
from gurobipy import GRB
sections = [1, 2, 3, 4, 5, 6, 7, 8]
capacities = {1: 100, 2: 150, 3: 120, 4: 130, 5: 90, 6: 110, 7: 160, 8: 140}
products = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
values = {1: 10, 2: 15, 3: 8, 4: 12, 5: 20, 6: 25, 7: 5, 8: 30, 9: 18, 10: 22}
weights = {1: 2, 2: 3, 3: 1, 4: 2, 5: 4, 6: 5, 7: 1, 8: 6, 9: 3, 10: 4}
if set(sections) != set(capacities.keys()):
    raise ValueError('Section IDs and capacities keys mismatch')
if set(products) != set(values.keys()) or set(products) != set(weights.keys()):
    raise ValueError('Product IDs and value/weight keys mismatch')
m = gp.Model('Supermarket_Section_Stocking')
x_vars = m.addVars(sections, products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((values[j] * x_vars[i, j] for i in sections for j in products)), GRB.MAXIMIZE)
for i in sections:
    m.addConstr(gp.quicksum((weights[j] * x_vars[i, j] for j in products)) <= capacities[i], name=f'cap_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')