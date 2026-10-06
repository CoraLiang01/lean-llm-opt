LEGACY_OBSERVATION = 'Product Name,Revenue,Initial Inventory,Demand\nAalopuri,20,10440.0,1483'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'Aalopuri', 'Revenue': '20', 'Initial Inventory': '10440.0', 'Demand': '1483'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
aalop_products = []
revenue = {}
initial_inventory = {}
demand = {}
for rec in records:
    vals = rec['values']
    if 'Product Name' in vals and vals['Product Name'].startswith('Aalop'):
        pname = vals['Product Name']
        aalop_products.append(pname)
        try:
            revenue[pname] = float(vals['Revenue'])
            initial_inventory[pname] = float(vals['Initial Inventory'])
            demand[pname] = float(vals['Demand'])
        except Exception as e:
            raise ValueError(f'Invalid data for {pname}: {e}')
for pname in aalop_products:
    if pname not in revenue or pname not in initial_inventory or pname not in demand:
        raise ValueError(f'Missing data for {pname}')
m = gp.Model('Aalopuri_Fulfillment')
x = m.addVars(aalop_products, lb=0, ub=[min(initial_inventory[p], demand[p]) for p in aalop_products], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in aalop_products)), GRB.MAXIMIZE)
for p in aalop_products:
    m.addConstr(x[p] <= initial_inventory[p], name=f'cap_inv_{p}')
    m.addConstr(x[p] <= demand[p], name=f'cap_dem_{p}')
    m.addConstr(x[p] >= 0, name=f'nonneg_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')