import gurobipy as gp
from gurobipy import GRB
products = [{'SKU': 'ZZ2AO', 'Revenue': 24.38, 'Initial Inventory': 10.0, 'Demand': 2}, {'SKU': 'ZZD W7', 'Revenue': 30.12, 'Initial Inventory': 20.0, 'Demand': 4}, {'SKU': 'ZZM1A', 'Revenue': 19.52, 'Initial Inventory': 530.0, 'Demand': 82}, {'SKU': 'ZZNC5', 'Revenue': 10.79, 'Initial Inventory': 10.0, 'Demand': 2}, {'SKU': 'ZZX6K', 'Revenue': 111.81, 'Initial Inventory': 10.0, 'Demand': 2}]
SKUs = [p['SKU'] for p in products]
revenue = {p['SKU']: p['Revenue'] for p in products}
inventory = {p['SKU']: p['Initial Inventory'] for p in products}
demand = {p['SKU']: p['Demand'] for p in products}
for sku in SKUs:
    if sku not in revenue or sku not in inventory or sku not in demand:
        raise ValueError(f'Missing data for SKU {sku}')
m = gp.Model('ZZ_Fulfillment')
x = m.addVars(SKUs, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in SKUs)), GRB.MAXIMIZE)
for i in SKUs:
    m.addConstr(x[i] <= inventory[i], name=f'inv_{i}')
for i in SKUs:
    m.addConstr(x[i] <= demand[i], name=f'dem_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')