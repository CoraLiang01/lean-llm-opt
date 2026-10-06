LEGACY_OBSERVATION = 'Product Name,Revenue,Demand,Initial Inventory\nsku_I27,238,6,30\nsku_I499,287,4,20\nsku_I719,268,16,80\nsku_T18,318,14,70\nsku_T29,207,4,20\nsku_T39,258,32,160\nsku_T499,249,8,40\nsku_T9,227,2,10\nsku_3081,198,10,50\nsku_339,254,8,40\nsku_3799,246,18,90\nsku_439,258,2,10\nsku_539,268,4,20\nsku_61399,278,8,40\nsku_628,268,2,10\nsku_708,298,198,990\nsku_77,258,32,160\nsku_79,315,18,90\nsku_799,264,570,2870\nsku_8499,238,6,30\nsku_89,258,26,130\nsku_897,268,6,30\nsku_9699,288,33,170\nsku_bobo,228,33,170'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'sku_I27', 'Revenue': '238', 'Demand': '6', 'Initial Inventory': '30'}}, {'source': '', 'values': {'Product Name': 'sku_I499', 'Revenue': '287', 'Demand': '4', 'Initial Inventory': '20'}}, {'source': '', 'values': {'Product Name': 'sku_I719', 'Revenue': '268', 'Demand': '16', 'Initial Inventory': '80'}}, {'source': '', 'values': {'Product Name': 'sku_T18', 'Revenue': '318', 'Demand': '14', 'Initial Inventory': '70'}}, {'source': '', 'values': {'Product Name': 'sku_T29', 'Revenue': '207', 'Demand': '4', 'Initial Inventory': '20'}}, {'source': '', 'values': {'Product Name': 'sku_T39', 'Revenue': '258', 'Demand': '32', 'Initial Inventory': '160'}}, {'source': '', 'values': {'Product Name': 'sku_T499', 'Revenue': '249', 'Demand': '8', 'Initial Inventory': '40'}}, {'source': '', 'values': {'Product Name': 'sku_T9', 'Revenue': '227', 'Demand': '2', 'Initial Inventory': '10'}}, {'source': '', 'values': {'Product Name': 'sku_3081', 'Revenue': '198', 'Demand': '10', 'Initial Inventory': '50'}}, {'source': '', 'values': {'Product Name': 'sku_339', 'Revenue': '254', 'Demand': '8', 'Initial Inventory': '40'}}, {'source': '', 'values': {'Product Name': 'sku_3799', 'Revenue': '246', 'Demand': '18', 'Initial Inventory': '90'}}, {'source': '', 'values': {'Product Name': 'sku_439', 'Revenue': '258', 'Demand': '2', 'Initial Inventory': '10'}}, {'source': '', 'values': {'Product Name': 'sku_539', 'Revenue': '268', 'Demand': '4', 'Initial Inventory': '20'}}, {'source': '', 'values': {'Product Name': 'sku_61399', 'Revenue': '278', 'Demand': '8', 'Initial Inventory': '40'}}, {'source': '', 'values': {'Product Name': 'sku_628', 'Revenue': '268', 'Demand': '2', 'Initial Inventory': '10'}}, {'source': '', 'values': {'Product Name': 'sku_708', 'Revenue': '298', 'Demand': '198', 'Initial Inventory': '990'}}, {'source': '', 'values': {'Product Name': 'sku_77', 'Revenue': '258', 'Demand': '32', 'Initial Inventory': '160'}}, {'source': '', 'values': {'Product Name': 'sku_79', 'Revenue': '315', 'Demand': '18', 'Initial Inventory': '90'}}, {'source': '', 'values': {'Product Name': 'sku_799', 'Revenue': '264', 'Demand': '570', 'Initial Inventory': '2870'}}, {'source': '', 'values': {'Product Name': 'sku_8499', 'Revenue': '238', 'Demand': '6', 'Initial Inventory': '30'}}, {'source': '', 'values': {'Product Name': 'sku_89', 'Revenue': '258', 'Demand': '26', 'Initial Inventory': '130'}}, {'source': '', 'values': {'Product Name': 'sku_897', 'Revenue': '268', 'Demand': '6', 'Initial Inventory': '30'}}, {'source': '', 'values': {'Product Name': 'sku_9699', 'Revenue': '288', 'Demand': '33', 'Initial Inventory': '170'}}, {'source': '', 'values': {'Product Name': 'sku_bobo', 'Revenue': '228', 'Demand': '33', 'Initial Inventory': '170'}}]
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
        revenue[pname] = int(vals['Revenue'])
        demand[pname] = int(vals['Demand'])
        inventory[pname] = int(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {pname}: {e}')
for pname in products:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f'Missing data for product {pname}')
m = gp.Model('demand_fulfillment')
x = m.addVars(products, lb=0, ub={p: min(demand[p], inventory[p]) for p in products}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')