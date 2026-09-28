LEGACY_OBSERVATION = '```csv\nSKU,Revenue,Demand,Initial Inventory\nZZ2AO,24.38,2,10.0\nZZDW7,30.12,4,20.0\nZZM1A,19.52,82,530.0\nZZNC5,10.79,2,10.0\nZZX6K,111.81,2,10.0\n```'
LEGACY_RECORDS = [{'source': '', 'values': {'SKU': 'ZZ2AO', 'Revenue': '24.38', 'Demand': '2', 'Initial Inventory': '10.0'}}, {'source': '', 'values': {'SKU': 'ZZDW7', 'Revenue': '30.12', 'Demand': '4', 'Initial Inventory': '20.0'}}, {'source': '', 'values': {'SKU': 'ZZM1A', 'Revenue': '19.52', 'Demand': '82', 'Initial Inventory': '530.0'}}, {'source': '', 'values': {'SKU': 'ZZNC5', 'Revenue': '10.79', 'Demand': '2', 'Initial Inventory': '10.0'}}, {'source': '', 'values': {'SKU': 'ZZX6K', 'Revenue': '111.81', 'Demand': '2', 'Initial Inventory': '10.0'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
skus = []
revenue = {}
demand = {}
init_inventory = {}
for rec in records:
    vals = rec['values']
    sku = vals['SKU']
    skus.append(sku)
    try:
        revenue[sku] = float(vals['Revenue'])
        demand[sku] = int(float(vals['Demand']))
        init_inventory[sku] = int(float(vals['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Invalid data for SKU {sku}: {e}')
for sku in skus:
    if sku not in revenue or sku not in demand or sku not in init_inventory:
        raise ValueError(f'Missing data for SKU {sku}')
m = gp.Model('ZZ_Fulfillment')
x = m.addVars(skus, lb=0, ub=[min(demand[i], init_inventory[i]) for i in skus], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in skus)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= demand[i] for i in skus), name='')
m.addConstrs((x[i] <= init_inventory[i] for i in skus), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')