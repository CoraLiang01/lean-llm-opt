LEGACY_OBSERVATION = 'Product Name,Revenue,Demand,Initial Inventory\nBaby Food_255.28,255.28,3066513,22749210'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '3066513', 'Initial Inventory': '22749210'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
baby_products = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in records:
    vals = rec['values']
    pname = vals['Product Name']
    baby_products.append(pname)
    try:
        revenue[pname] = float(vals['Revenue'])
        demand[pname] = int(vals['Demand'])
        initial_inventory[pname] = int(vals['Initial Inventory'])
    except KeyError as e:
        raise ValueError(f'Missing required field {e} for product {pname}')
for pname in baby_products:
    if pname not in revenue or pname not in demand or pname not in initial_inventory:
        raise ValueError(f'Missing data for product {pname}')
m = gp.Model('Baby_Food_Fulfillment')
x = m.addVars(baby_products, lb=0, ub=[min(demand[p], initial_inventory[p]) for p in baby_products], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in baby_products)), GRB.MAXIMIZE)
for p in baby_products:
    m.addConstr(x[p] <= demand[p], name=f'demand_{p}')
    m.addConstr(x[p] <= initial_inventory[p], name=f'inventory_{p}')
    m.addConstr(x[p] >= 0, name=f'nonneg_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')