import gurobipy as gp
from gurobipy import GRB
products = [{'Product Identifier': 'S700_1138', 'Revenue': 70.67, 'Initial Inventory': 9020, 'Demand': 1219}, {'Product Identifier': 'S700_1691', 'Revenue': 100.0, 'Initial Inventory': 8370, 'Demand': 1127}, {'Product Identifier': 'S700_1938', 'Revenue': 70.15, 'Initial Inventory': 8390, 'Demand': 1129}, {'Product Identifier': 'S700_2047', 'Revenue': 100.0, 'Initial Inventory': 8680, 'Demand': 1176}, {'Product Identifier': 'S700_2466', 'Revenue': 100.0, 'Initial Inventory': 9400, 'Demand': 1301}, {'Product Identifier': 'S700_2610', 'Revenue': 65.77, 'Initial Inventory': 9900, 'Demand': 1340}, {'Product Identifier': 'S700_2824', 'Revenue': 100.0, 'Initial Inventory': 9760, 'Demand': 1357}, {'Product Identifier': 'S700_2834', 'Revenue': 100.0, 'Initial Inventory': 8610, 'Demand': 1158}, {'Product Identifier': 'S700_3167', 'Revenue': 74.4, 'Initial Inventory': 9380, 'Demand': 1287}, {'Product Identifier': 'S700_3505', 'Revenue': 81.14, 'Initial Inventory': 9170, 'Demand': 1281}, {'Product Identifier': 'S700_3962', 'Revenue': 100.0, 'Initial Inventory': 8520, 'Demand': 1135}, {'Product Identifier': 'S700_4002', 'Revenue': 61.44, 'Initial Inventory': 10290, 'Demand': 1392}]
product_ids = [p['Product Identifier'] for p in products]
revenue = {p['Product Identifier']: p['Revenue'] for p in products}
initial_inventory = {p['Product Identifier']: p['Initial Inventory'] for p in products}
demand = {p['Product Identifier']: p['Demand'] for p in products}
for pid in product_ids:
    if pid not in revenue or pid not in initial_inventory or pid not in demand:
        raise ValueError(f'Missing data for product {pid}')
m = gp.Model('retail_revenue_max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_ids, vtype=GRB.CONTINUOUS, lb=0, name='')
m.addConstrs((x_vars[pid] <= initial_inventory[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= demand[pid] for pid in product_ids), name='')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in product_ids)), GRB.MAXIMIZE)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for pid in product_ids:
        print(f'x[{pid}]: {x_vars[pid].X}')
else:
    print(f'Solver status: {m.Status}')