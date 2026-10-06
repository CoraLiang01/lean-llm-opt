LEGACY_OBSERVATION = '```csv\nSKU,Revenue,Initial Inventory,Demand\nZZ2AO,24.38,10.0,2\nZZDW7,30.12,20.0,4\nZZM1A,19.52,530.0,82\nZZNC5,10.79,10.0,2\nZZX6K,111.81,10.0,2\n```'
LEGACY_RECORDS = [{'source': '', 'values': {'SKU': 'ZZ2AO', 'Revenue': '24.38', 'Initial Inventory': '10.0', 'Demand': '2'}}, {'source': '', 'values': {'SKU': 'ZZDW7', 'Revenue': '30.12', 'Initial Inventory': '20.0', 'Demand': '4'}}, {'source': '', 'values': {'SKU': 'ZZM1A', 'Revenue': '19.52', 'Initial Inventory': '530.0', 'Demand': '82'}}, {'source': '', 'values': {'SKU': 'ZZNC5', 'Revenue': '10.79', 'Initial Inventory': '10.0', 'Demand': '2'}}, {'source': '', 'values': {'SKU': 'ZZX6K', 'Revenue': '111.81', 'Initial Inventory': '10.0', 'Demand': '2'}}]
import gurobipy as gp
from gurobipy import GRB
I = []
revenue = {}
init_inventory = {}
demand = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    sku = vals['SKU']
    I.append(sku)
    try:
        revenue[sku] = float(vals['Revenue'])
        init_inventory[sku] = float(vals['Initial Inventory'])
        demand[sku] = float(vals['Demand'])
    except KeyError as e:
        raise ValueError(f'Missing required field {e} for SKU {sku}')
    except Exception as e:
        raise ValueError(f'Error parsing record for SKU {sku}: {e}')
for sku in I:
    if sku not in revenue or sku not in init_inventory or sku not in demand:
        raise ValueError(f'Missing data for SKU {sku}')
upper_bound = {sku: min(init_inventory[sku], demand[sku]) for sku in I}
m = gp.Model('ZZ_Inventory_Allocation')
x = m.addVars(I, lb=0, ub=upper_bound, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')