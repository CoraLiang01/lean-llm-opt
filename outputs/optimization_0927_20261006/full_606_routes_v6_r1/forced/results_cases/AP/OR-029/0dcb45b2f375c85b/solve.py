import gurobipy as gp
from gurobipy import GRB
products = ['FAUX FUR JEWEL SWEATER', 'FAUX LEATHER BOMBER JACKET', 'FAUX LEATHER BOXY FIT JACKET', 'FAUX LEATHER JACKET', 'FAUX LEATHER OVERSIZED JACKET LIMITED EDITION', 'FAUX LEATHER PUFFER JACKET', 'FAUX SHEARLING LINED SUEDE BOOTS', 'FAUX SHEARLING PLAID JACKET', 'FAUX SUEDE BOMBER JACKET', 'FAUX SUEDE JACKET', 'FAUX SUEDE OVERSHIRT', 'FAUX SUEDE PATCH JACKET']
revenue = {'FAUX FUR JEWEL SWEATER': 35.9, 'FAUX LEATHER BOMBER JACKET': 69.9, 'FAUX LEATHER BOXY FIT JACKET': 99.9, 'FAUX LEATHER JACKET': 99.9, 'FAUX LEATHER OVERSIZED JACKET LIMITED EDITION': 159.0, 'FAUX LEATHER PUFFER JACKET': 69.99, 'FAUX SHEARLING LINED SUEDE BOOTS': 99.9, 'FAUX SHEARLING PLAID JACKET': 89.9, 'FAUX SUEDE BOMBER JACKET': 69.9, 'FAUX SUEDE JACKET': 89.9, 'FAUX SUEDE OVERSHIRT': 69.9, 'FAUX SUEDE PATCH JACKET': 89.9}
initial_inventory = {'FAUX FUR JEWEL SWEATER': 20970, 'FAUX LEATHER BOMBER JACKET': 71970, 'FAUX LEATHER BOXY FIT JACKET': 32730, 'FAUX LEATHER JACKET': 71130, 'FAUX LEATHER OVERSIZED JACKET LIMITED EDITION': 34910, 'FAUX LEATHER PUFFER JACKET': 64010, 'FAUX SHEARLING LINED SUEDE BOOTS': 20760, 'FAUX SHEARLING PLAID JACKET': 12490, 'FAUX SUEDE BOMBER JACKET': 50300, 'FAUX SUEDE JACKET': 24570, 'FAUX SUEDE OVERSHIRT': 24430, 'FAUX SUEDE PATCH JACKET': 7070}
demand = {'FAUX FUR JEWEL SWEATER': 3025, 'FAUX LEATHER BOMBER JACKET': 9585, 'FAUX LEATHER BOXY FIT JACKET': 4486, 'FAUX LEATHER JACKET': 10322, 'FAUX LEATHER OVERSIZED JACKET LIMITED EDITION': 4868, 'FAUX LEATHER PUFFER JACKET': 8482, 'FAUX SHEARLING LINED SUEDE BOOTS': 2607, 'FAUX SHEARLING PLAID JACKET': 1784, 'FAUX SUEDE BOMBER JACKET': 6626, 'FAUX SUEDE JACKET': 3256, 'FAUX SUEDE OVERSHIRT': 2955, 'FAUX SUEDE PATCH JACKET': 910}
for p in products:
    if p not in revenue or p not in initial_inventory or p not in demand:
        raise ValueError(f'Missing data for product: {p}')
m = gp.Model('Faux_Product_Revenue_Maximization')
x_vars = m.addVars(products, lb=0, ub=[min(initial_inventory[p], demand[p]) for p in products], vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
for p in products:
    m.addConstr(x_vars[p] <= initial_inventory[p], name=f'ub_inv_{products.index(p) + 1}')
    m.addConstr(x_vars[p] <= demand[p], name=f'ub_dem_{products.index(p) + 1}')
    m.addConstr(x_vars[p] >= 0, name=f'lb_0_{products.index(p) + 1}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')