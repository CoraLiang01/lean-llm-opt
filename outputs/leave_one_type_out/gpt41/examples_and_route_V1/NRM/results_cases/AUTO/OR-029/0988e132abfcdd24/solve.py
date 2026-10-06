LEGACY_OBSERVATION = 'Product Name,Revenue,Demand,Initial Inventory\nFAUX LEATHER BOMBER JACKET,69.9,9585,71970\nFAUX LEATHER BOXY FIT JACKET,99.9,4486,32730\nFAUX LEATHER JACKET,99.9,10322,71130\nFAUX LEATHER OVERSIZED JACKET LIMITED EDITION,159.0,4868,34910\nFAUX LEATHER PUFFER JACKET,69.99,8482,64010\nFAUX SHEARLING LINED SUEDE BOOTS,99.9,2607,20760\nFAUX SHEARLING PLAID JACKET,89.9,1784,12490\nFAUX SUEDE BOMBER JACKET,69.9,6626,50300\nFAUX SUEDE JACKET,89.9,3256,24570\nFAUX SUEDE OVERSHIRT,69.9,2955,24430\nFAUX SUEDE PATCH JACKET,89.9,910,7070'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'FAUX LEATHER BOMBER JACKET', 'Revenue': '69.9', 'Demand': '9585', 'Initial Inventory': '71970'}}, {'source': '', 'values': {'Product Name': 'FAUX LEATHER BOXY FIT JACKET', 'Revenue': '99.9', 'Demand': '4486', 'Initial Inventory': '32730'}}, {'source': '', 'values': {'Product Name': 'FAUX LEATHER JACKET', 'Revenue': '99.9', 'Demand': '10322', 'Initial Inventory': '71130'}}, {'source': '', 'values': {'Product Name': 'FAUX LEATHER OVERSIZED JACKET LIMITED EDITION', 'Revenue': '159.0', 'Demand': '4868', 'Initial Inventory': '34910'}}, {'source': '', 'values': {'Product Name': 'FAUX LEATHER PUFFER JACKET', 'Revenue': '69.99', 'Demand': '8482', 'Initial Inventory': '64010'}}, {'source': '', 'values': {'Product Name': 'FAUX SHEARLING LINED SUEDE BOOTS', 'Revenue': '99.9', 'Demand': '2607', 'Initial Inventory': '20760'}}, {'source': '', 'values': {'Product Name': 'FAUX SHEARLING PLAID JACKET', 'Revenue': '89.9', 'Demand': '1784', 'Initial Inventory': '12490'}}, {'source': '', 'values': {'Product Name': 'FAUX SUEDE BOMBER JACKET', 'Revenue': '69.9', 'Demand': '6626', 'Initial Inventory': '50300'}}, {'source': '', 'values': {'Product Name': 'FAUX SUEDE JACKET', 'Revenue': '89.9', 'Demand': '3256', 'Initial Inventory': '24570'}}, {'source': '', 'values': {'Product Name': 'FAUX SUEDE OVERSHIRT', 'Revenue': '69.9', 'Demand': '2955', 'Initial Inventory': '24430'}}, {'source': '', 'values': {'Product Name': 'FAUX SUEDE PATCH JACKET', 'Revenue': '89.9', 'Demand': '910', 'Initial Inventory': '7070'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
revenue = {}
demand = {}
inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    pname = vals['Product Name']
    products.append(pname)
    try:
        revenue[pname] = float(vals['Revenue'])
        demand[pname] = int(vals['Demand'])
        inventory[pname] = int(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f"Invalid data for product '{pname}': {e}")
for pname in products:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f"Missing data for product '{pname}'")
m = gp.Model('Faux_Product_Fulfillment')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstrs((x[p] <= demand[p] for p in products), name='')
m.addConstrs((x[p] <= inventory[p] for p in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')