LEGACY_OBSERVATION = 'Product Name,Revenue,Demand,Initial Inventory\nBaby Food_255.28,255.28,765850,5627060'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'Baby Food_255.28', 'Revenue': '255.28', 'Demand': '765850', 'Initial Inventory': '5627060'}}]
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
    if pname.startswith('Baby'):
        baby_products.append(pname)
        try:
            revenue[pname] = float(vals['Revenue'])
            demand[pname] = int(vals['Demand'])
            initial_inventory[pname] = int(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f'Missing or invalid data for product {pname}: {e}')
for pname in baby_products:
    if pname not in revenue or pname not in demand or pname not in initial_inventory:
        raise ValueError(f'Missing data for product {pname}')
m = gp.Model('Baby_Fulfillment')
x = m.addVars(baby_products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in baby_products)), GRB.MAXIMIZE)
for i in baby_products:
    m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
    m.addConstr(x[i] <= initial_inventory[i], name=f'inventory_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')