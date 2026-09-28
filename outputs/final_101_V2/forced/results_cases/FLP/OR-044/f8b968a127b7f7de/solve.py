import gurobipy as gp
from gurobipy import GRB
sections = [1, 2, 3, 4, 5, 6, 7, 8]
section_caps = {1: 100, 2: 150, 3: 120, 4: 130, 5: 90, 6: 110, 7: 160, 8: 140}
products = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
product_vals = {1: 10, 2: 15, 3: 8, 4: 12, 5: 20, 6: 25, 7: 5, 8: 30, 9: 18, 10: 22}
product_wts = {1: 2, 2: 3, 3: 1, 4: 2, 5: 4, 6: 5, 7: 1, 8: 6, 9: 3, 10: 4}
if set(sections) != set(section_caps.keys()):
    raise ValueError('Section capacity data missing for some sections.')
if set(products) != set(product_vals.keys()) or set(products) != set(product_wts.keys()):
    raise ValueError('Product value/weight data missing for some products.')
m = gp.Model('Supermarket_Section_Stocking')
x = m.addVars(sections, products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_vals[j] * x[i, j] for i in sections for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((product_wts[j] * x[i, j] for j in products)) <= section_caps[i] for i in sections), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')