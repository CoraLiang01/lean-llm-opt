LEGACY_OBSERVATION = 'Product Name,Revenue,Initial Inventory,Demand\nS700_1138,70.67,9020,1219\nS700_1691,100.0,8370,1127\nS700_1938,70.15,8390,1129\nS700_2047,100.0,8680,1176\nS700_2466,100.0,9400,1301\nS700_2610,65.77,9900,1340\nS700_2824,100.0,9760,1357\nS700_2834,100.0,8610,1158\nS700_3167,74.4,9380,1287\nS700_3505,81.14,9170,1281\nS700_3962,100.0,8520,1135\nS700_4002,61.44,10290,1392'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'S700_1138', 'Revenue': '70.67', 'Initial Inventory': '9020', 'Demand': '1219'}}, {'source': '', 'values': {'Product Name': 'S700_1691', 'Revenue': '100.0', 'Initial Inventory': '8370', 'Demand': '1127'}}, {'source': '', 'values': {'Product Name': 'S700_1938', 'Revenue': '70.15', 'Initial Inventory': '8390', 'Demand': '1129'}}, {'source': '', 'values': {'Product Name': 'S700_2047', 'Revenue': '100.0', 'Initial Inventory': '8680', 'Demand': '1176'}}, {'source': '', 'values': {'Product Name': 'S700_2466', 'Revenue': '100.0', 'Initial Inventory': '9400', 'Demand': '1301'}}, {'source': '', 'values': {'Product Name': 'S700_2610', 'Revenue': '65.77', 'Initial Inventory': '9900', 'Demand': '1340'}}, {'source': '', 'values': {'Product Name': 'S700_2824', 'Revenue': '100.0', 'Initial Inventory': '9760', 'Demand': '1357'}}, {'source': '', 'values': {'Product Name': 'S700_2834', 'Revenue': '100.0', 'Initial Inventory': '8610', 'Demand': '1158'}}, {'source': '', 'values': {'Product Name': 'S700_3167', 'Revenue': '74.4', 'Initial Inventory': '9380', 'Demand': '1287'}}, {'source': '', 'values': {'Product Name': 'S700_3505', 'Revenue': '81.14', 'Initial Inventory': '9170', 'Demand': '1281'}}, {'source': '', 'values': {'Product Name': 'S700_3962', 'Revenue': '100.0', 'Initial Inventory': '8520', 'Demand': '1135'}}, {'source': '', 'values': {'Product Name': 'S700_4002', 'Revenue': '61.44', 'Initial Inventory': '10290', 'Demand': '1392'}}]
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
    revenue[pname] = float(vals['Revenue'])
    initial_inventory[pname] = int(vals['Initial Inventory'])
    demand[pname] = int(vals['Demand'])
for pname in products:
    if pname not in revenue or pname not in initial_inventory or pname not in demand:
        raise ValueError(f'Missing data for product {pname}')
m = gp.Model('retail_revenue')
x = m.addVars(products, lb=0, ub={p: min(initial_inventory[p], demand[p]) for p in products}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstrs((x[p] <= initial_inventory[p] for p in products), name='')
m.addConstrs((x[p] <= demand[p] for p in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')