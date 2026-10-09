import gurobipy as gp
from gurobipy import GRB
products = [{'i': 1, 'Product Name': 'FAUX FUR JEWEL SWEATER', 'r': 35.9, 'd': 3025, 's': 20970}, {'i': 2, 'Product Name': 'FAUX LEATHER BOMBER JACKET', 'r': 69.9, 'd': 9585, 's': 71970}, {'i': 3, 'Product Name': 'FAUX LEATHER BOXY FIT JACKET', 'r': 99.9, 'd': 4486, 's': 32730}, {'i': 4, 'Product Name': 'FAUX LEATHER JACKET', 'r': 99.9, 'd': 10322, 's': 71130}, {'i': 5, 'Product Name': 'FAUX LEATHER OVERSIZED JACKET LIMITED EDITION', 'r': 159.0, 'd': 4868, 's': 34910}, {'i': 6, 'Product Name': 'FAUX LEATHER PUFFER JACKET', 'r': 69.99, 'd': 8482, 's': 64010}, {'i': 7, 'Product Name': 'FAUX SHEARLING LINED SUEDE BOOTS', 'r': 99.9, 'd': 2607, 's': 20760}, {'i': 8, 'Product Name': 'FAUX SHEARLING PLAID JACKET', 'r': 89.9, 'd': 1784, 's': 12490}, {'i': 9, 'Product Name': 'FAUX SUEDE BOMBER JACKET', 'r': 69.9, 'd': 6626, 's': 50300}, {'i': 10, 'Product Name': 'FAUX SUEDE JACKET', 'r': 89.9, 'd': 3256, 's': 24570}, {'i': 11, 'Product Name': 'FAUX SUEDE OVERSHIRT', 'r': 69.9, 'd': 2955, 's': 24430}, {'i': 12, 'Product Name': 'FAUX SUEDE PATCH JACKET', 'r': 89.9, 'd': 910, 's': 7070}]
product_keys = [p['i'] for p in products]
revenue = {p['i']: p['r'] for p in products}
demand = {p['i']: p['d'] for p in products}
inventory = {p['i']: p['s'] for p in products}
if set(revenue.keys()) != set(product_keys) or set(demand.keys()) != set(product_keys) or set(inventory.keys()) != set(product_keys):
    raise ValueError('Missing data for some products.')
m = gp.Model('Faux_Product_Fulfillment')
x_vars = m.addVars(product_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in product_keys)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in product_keys), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in product_keys), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')