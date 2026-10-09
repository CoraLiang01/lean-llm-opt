import gurobipy as gp
from gurobipy import GRB
products = [{'Product Name': '27in 4K Gaming Monitor', 'Revenue': 389.99, 'Initial Inventory': 62440, 'Demand': 12474}, {'Product Name': '27in FHD Monitor', 'Revenue': 149.99, 'Initial Inventory': 75500, 'Demand': 15057}]
product_keys = [p['Product Name'] for p in products]
revenue = {p['Product Name']: p['Revenue'] for p in products}
initial_inventory = {p['Product Name']: p['Initial Inventory'] for p in products}
demand = {p['Product Name']: p['Demand'] for p in products}
m = gp.Model('27in_Product_Revenue_Max')
x_vars = m.addVars(product_keys, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[k] * x_vars[k] for k in product_keys)), GRB.MAXIMIZE)
for k in product_keys:
    m.addConstr(x_vars[k] <= initial_inventory[k], name=f'inv_{k}')
    m.addConstr(x_vars[k] <= demand[k], name=f'dem_{k}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')