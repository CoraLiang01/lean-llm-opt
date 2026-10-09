LEGACY_OBSERVATION = 'Product Name,Revenue,Initial Inventory,Demand\nBaby Food_255.28,255.28,5627060,765850'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Initial Inventory': '5627060', 'Demand': '765850'}}]
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
    except KeyError as e:
        raise ValueError(f'Missing required field {e} for product {pname}')
for pname in products:
    if pname not in revenue or pname not in initial_inventory or pname not in demand:
        raise ValueError(f'Missing data for product {pname}')
m = gp.Model('Baby_Product_Fulfillment')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
for i in products:
    m.addConstr(x[i] <= initial_inventory[i], name=f'inv_{i}')
    m.addConstr(x[i] <= demand[i], name=f'dem_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')