LEGACY_OBSERVATION = 'Product Name,Revenue,Initial Inventory,Demand\nFAUX FUR JEWEL SWEATER,35.9,20970,3025\nFAUX LEATHER BOMBER JACKET,69.9,71970,9585\nFAUX LEATHER BOXY FIT JACKET,99.9,32730,4486\nFAUX LEATHER JACKET,99.9,71130,10322\nFAUX LEATHER OVERSIZED JACKET LIMITED EDITION,159.0,34910,4868\nFAUX LEATHER PUFFER JACKET,69.99,64010,8482\nFAUX SHEARLING LINED SUEDE BOOTS,99.9,20760,2607\nFAUX SHEARLING PLAID JACKET,89.9,12490,1784\nFAUX SUEDE BOMBER JACKET,69.9,50300,6626\nFAUX SUEDE JACKET,89.9,24570,3256\nFAUX SUEDE OVERSHIRT,69.9,24430,2955\nFAUX SUEDE PATCH JACKET,89.9,7070,910'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'FAUX FUR JEWEL SWEATER', 'Revenue': '35.9', 'Initial Inventory': '20970', 'Demand': '3025'}}, {'source': '', 'values': {'Product Name': 'FAUX LEATHER BOMBER JACKET', 'Revenue': '69.9', 'Initial Inventory': '71970', 'Demand': '9585'}}, {'source': '', 'values': {'Product Name': 'FAUX LEATHER BOXY FIT JACKET', 'Revenue': '99.9', 'Initial Inventory': '32730', 'Demand': '4486'}}, {'source': '', 'values': {'Product Name': 'FAUX LEATHER JACKET', 'Revenue': '99.9', 'Initial Inventory': '71130', 'Demand': '10322'}}, {'source': '', 'values': {'Product Name': 'FAUX LEATHER OVERSIZED JACKET LIMITED EDITION', 'Revenue': '159.0', 'Initial Inventory': '34910', 'Demand': '4868'}}, {'source': '', 'values': {'Product Name': 'FAUX LEATHER PUFFER JACKET', 'Revenue': '69.99', 'Initial Inventory': '64010', 'Demand': '8482'}}, {'source': '', 'values': {'Product Name': 'FAUX SHEARLING LINED SUEDE BOOTS', 'Revenue': '99.9', 'Initial Inventory': '20760', 'Demand': '2607'}}, {'source': '', 'values': {'Product Name': 'FAUX SHEARLING PLAID JACKET', 'Revenue': '89.9', 'Initial Inventory': '12490', 'Demand': '1784'}}, {'source': '', 'values': {'Product Name': 'FAUX SUEDE BOMBER JACKET', 'Revenue': '69.9', 'Initial Inventory': '50300', 'Demand': '6626'}}, {'source': '', 'values': {'Product Name': 'FAUX SUEDE JACKET', 'Revenue': '89.9', 'Initial Inventory': '24570', 'Demand': '3256'}}, {'source': '', 'values': {'Product Name': 'FAUX SUEDE OVERSHIRT', 'Revenue': '69.9', 'Initial Inventory': '24430', 'Demand': '2955'}}, {'source': '', 'values': {'Product Name': 'FAUX SUEDE PATCH JACKET', 'Revenue': '89.9', 'Initial Inventory': '7070', 'Demand': '910'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
revenue = {}
initial_inventory = {}
demand = {}
for rec in records:
    vals = rec['values']
    pname = vals['Product Name']
    products.append(pname)
    try:
        revenue[pname] = float(vals['Revenue'])
        initial_inventory[pname] = int(vals['Initial Inventory'])
        demand[pname] = int(vals['Demand'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {pname}: {e}')
for pname in products:
    if pname not in revenue or pname not in initial_inventory or pname not in demand:
        raise ValueError(f'Missing data for product {pname}')
m = gp.Model('Faux_Product_Fulfillment')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), GRB.MAXIMIZE)
for p in products:
    m.addConstr(x[p] <= initial_inventory[p], name=f'inv_{p}')
    m.addConstr(x[p] <= demand[p], name=f'dem_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')