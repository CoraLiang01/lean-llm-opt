LEGACY_OBSERVATION = '{"values": {"Product Name": "27in 4K Gaming Monitor", "Revenue": "389.99", "Demand": "12474", "Initial Inventory": "62440"}}\n{"values": {"Product Name": "27in FHD Monitor", "Revenue": "149.99", "Demand": "15057", "Initial Inventory": "75500"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': '27in 4K Gaming Monitor', 'Revenue': '389.99', 'Demand': '12474', 'Initial Inventory': '62440'}}, {'source': '', 'values': {'Product Name': '27in FHD Monitor', 'Revenue': '149.99', 'Demand': '15057', 'Initial Inventory': '75500'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
revenue = {}
demand = {}
initial_inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    pname = vals['Product Name']
    products.append(pname)
    try:
        revenue[pname] = float(vals['Revenue'])
        demand[pname] = int(vals['Demand'])
        initial_inventory[pname] = int(vals['Initial Inventory'])
    except KeyError as e:
        raise ValueError(f'Missing required field {e} for product {pname}')
for pname in products:
    if pname not in revenue or pname not in demand or pname not in initial_inventory:
        raise ValueError(f'Missing data for product {pname}')
m = gp.Model('27in_Product_Fulfillment')
x = m.addVars(products, lb=0, ub={p: min(demand[p], initial_inventory[p]) for p in products}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), GRB.MAXIMIZE)
for p in products:
    m.addConstr(x[p] <= demand[p], name=f'demand_{p}')
    m.addConstr(x[p] <= initial_inventory[p], name=f'inv_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')