import gurobipy as gp
from gurobipy import GRB
indices = [56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
product_names = {56: 'FAUX FUR JEWEL SWEATER', 57: 'FAUX LEATHER BOMBER JACKET', 58: 'FAUX LEATHER BOXY FIT JACKET', 59: 'FAUX LEATHER JACKET', 60: 'FAUX LEATHER OVERSIZED JACKET LIMITED EDITION', 61: 'FAUX LEATHER PUFFER JACKET', 62: 'FAUX SHEARLING LINED SUEDE BOOTS', 63: 'FAUX SHEARLING PLAID JACKET', 64: 'FAUX SUEDE BOMBER JACKET', 65: 'FAUX SUEDE JACKET', 66: 'FAUX SUEDE OVERSHIRT', 67: 'FAUX SUEDE PATCH JACKET'}
revenue = {56: 35.9, 57: 69.9, 58: 99.9, 59: 99.9, 60: 159.0, 61: 69.99, 62: 99.9, 63: 89.9, 64: 69.9, 65: 89.9, 66: 69.9, 67: 89.9}
demand = {56: 3025, 57: 9585, 58: 4486, 59: 10322, 60: 4868, 61: 8482, 62: 2607, 63: 1784, 64: 6626, 65: 3256, 66: 2955, 67: 910}
inventory = {56: 20970, 57: 71970, 58: 32730, 59: 71130, 60: 34910, 61: 64010, 62: 20760, 63: 12490, 64: 50300, 65: 24570, 66: 24430, 67: 7070}
for i in indices:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f'Missing data for index {i}')
m = gp.Model('FAUX_Revenue_Maximization')
x_vars = m.addVars(indices, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in indices)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in indices), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in indices), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in indices:
        print(f'x_{i}: {x_vars[i].X}')
else:
    print(f'Solver status: {m.Status}')