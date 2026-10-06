LEGACY_OBSERVATION = '{"values": {"Product Name": "27in 4K Gaming Monitor", "Revenue": "261.2933", "Demand": "12474", "Initial Inventory": "62440"}}\n{"values": {"Product Name": "27in FHD Monitor", "Revenue": "52.4965", "Demand": "15057", "Initial Inventory": "75500"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': '27in 4K Gaming Monitor', 'Revenue': '261.2933', 'Demand': '12474', 'Initial Inventory': '62440'}}, {'source': '', 'values': {'Product Name': '27in FHD Monitor', 'Revenue': '52.4965', 'Demand': '15057', 'Initial Inventory': '75500'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
revenue = {}
demand = {}
inventory = {}
for rec in records:
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
m = gp.Model('27in_Product_Revenue')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), GRB.MAXIMIZE)
for p in products:
    m.addConstr(x[p] <= demand[p], name=f'demand_{p}')
    m.addConstr(x[p] <= inventory[p], name=f'inventory_{p}')
    m.addConstr(x[p] >= 0, name=f'nonneg_{p}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')