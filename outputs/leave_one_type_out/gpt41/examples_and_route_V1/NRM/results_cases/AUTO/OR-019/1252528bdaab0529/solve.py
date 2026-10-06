LEGACY_OBSERVATION = 'Product Name,Revenue,Initial Inventory,Demand\n27in 4K Gaming Monitor,389.99,62440,12474\n27in FHD Monitor,149.99,75500,15057'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99', 'Initial Inventory': '62440', 'Demand': '12474'}}, {'source': '', 'values': {'Product Name': '27in FHD Monitor', 'Revenue': '149.99', 'Initial Inventory': '75500', 'Demand': '15057'}}]
import gurobipy as gp
from gurobipy import GRB
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99', 'Initial Inventory': '62440', 'Demand': '12474'}}, {'source': '', 'values': {'Product Name': '27in FHD Monitor', 'Revenue': '149.99', 'Initial Inventory': '75500', 'Demand': '15057'}}]
products = []
revenue = {}
inventory = {}
demand = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    pname = vals['Product Name']
    products.append(pname)
    try:
        revenue[pname] = float(vals['Revenue'])
        inventory[pname] = int(vals['Initial Inventory'])
        demand[pname] = int(vals['Demand'])
    except Exception as e:
        raise ValueError(f"Missing or invalid data for product '{pname}': {e}")
for pname in products:
    if pname not in revenue or pname not in inventory or pname not in demand:
        raise ValueError(f"Missing data for product '{pname}'.")
m = gp.Model('27in_Product_Fulfillment')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), GRB.MAXIMIZE)
for p in products:
    m.addConstr(x[p] <= inventory[p], name=f'inv_{p}')
    m.addConstr(x[p] <= demand[p], name=f'dem_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')