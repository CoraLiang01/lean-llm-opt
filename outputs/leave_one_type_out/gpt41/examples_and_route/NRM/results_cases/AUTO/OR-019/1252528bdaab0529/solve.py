LEGACY_OBSERVATION = 'Product Name,Revenue,Demand,Initial Inventory\n27in 4K Gaming Monitor,389.99,12474,62440\n27in FHD Monitor,149.99,15057,75500'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99', 'Demand': '12474', 'Initial Inventory': '62440'}}, {'source': '', 'values': {'Product Name': '27in FHD Monitor', 'Revenue': '149.99', 'Demand': '15057', 'Initial Inventory': '75500'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
revenue = {}
demand = {}
inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    pname = vals['Product Name']
    if '27in' in pname:
        products.append(pname)
        try:
            revenue[pname] = float(vals['Revenue'])
            demand[pname] = int(vals['Demand'])
            inventory[pname] = int(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f"Missing or invalid data for product '{pname}': {e}")
for pname in products:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f"Missing data for product '{pname}'")
m = gp.Model('27in_Product_Fulfillment')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= demand[i] for i in products), name='')
m.addConstrs((x[i] <= inventory[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')