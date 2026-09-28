LEGACY_OBSERVATION = '{"Product Name": "S700_1138", "Revenue": "70.67", "Demand": "1219", "Initial Inventory": "9020"}\n{"Product Name": "S700_1691", "Revenue": "100.0", "Demand": "1127", "Initial Inventory": "8370"}\n{"Product Name": "S700_1938", "Revenue": "70.15", "Demand": "1129", "Initial Inventory": "8390"}\n{"Product Name": "S700_2047", "Revenue": "100.0", "Demand": "1176", "Initial Inventory": "8680"}\n{"Product Name": "S700_2466", "Revenue": "100.0", "Demand": "1301", "Initial Inventory": "9400"}\n{"Product Name": "S700_2610", "Revenue": "65.77", "Demand": "1340", "Initial Inventory": "9900"}\n{"Product Name": "S700_2824", "Revenue": "100.0", "Demand": "1357", "Initial Inventory": "9760"}\n{"Product Name": "S700_2834", "Revenue": "100.0", "Demand": "1158", "Initial Inventory": "8610"}\n{"Product Name": "S700_3167", "Revenue": "74.4", "Demand": "1287", "Initial Inventory": "9380"}\n{"Product Name": "S700_3505", "Revenue": "81.14", "Demand": "1281", "Initial Inventory": "9170"}\n{"Product Name": "S700_3962", "Revenue": "100.0", "Demand": "1135", "Initial Inventory": "8520"}\n{"Product Name": "S700_4002", "Revenue": "61.44", "Demand": "1392", "Initial Inventory": "10290"}'
LEGACY_RECORDS = [{'source': '', 'values': {'Product Name': 'S700_1138', 'Revenue': '70.67', 'Demand': '1219', 'Initial Inventory': '9020'}}, {'source': '', 'values': {'Product Name': 'S700_1691', 'Revenue': '100.0', 'Demand': '1127', 'Initial Inventory': '8370'}}, {'source': '', 'values': {'Product Name': 'S700_1938', 'Revenue': '70.15', 'Demand': '1129', 'Initial Inventory': '8390'}}, {'source': '', 'values': {'Product Name': 'S700_2047', 'Revenue': '100.0', 'Demand': '1176', 'Initial Inventory': '8680'}}, {'source': '', 'values': {'Product Name': 'S700_2466', 'Revenue': '100.0', 'Demand': '1301', 'Initial Inventory': '9400'}}, {'source': '', 'values': {'Product Name': 'S700_2610', 'Revenue': '65.77', 'Demand': '1340', 'Initial Inventory': '9900'}}, {'source': '', 'values': {'Product Name': 'S700_2824', 'Revenue': '100.0', 'Demand': '1357', 'Initial Inventory': '9760'}}, {'source': '', 'values': {'Product Name': 'S700_2834', 'Revenue': '100.0', 'Demand': '1158', 'Initial Inventory': '8610'}}, {'source': '', 'values': {'Product Name': 'S700_3167', 'Revenue': '74.4', 'Demand': '1287', 'Initial Inventory': '9380'}}, {'source': '', 'values': {'Product Name': 'S700_3505', 'Revenue': '81.14', 'Demand': '1281', 'Initial Inventory': '9170'}}, {'source': '', 'values': {'Product Name': 'S700_3962', 'Revenue': '100.0', 'Demand': '1135', 'Initial Inventory': '8520'}}, {'source': '', 'values': {'Product Name': 'S700_4002', 'Revenue': '61.44', 'Demand': '1392', 'Initial Inventory': '10290'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
revenue = {}
demand = {}
inventory = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    pid = vals['Product Name']
    products.append(pid)
    try:
        revenue[pid] = float(vals['Revenue'])
        demand[pid] = int(vals['Demand'])
        inventory[pid] = int(vals['Initial Inventory'])
    except Exception as e:
        raise ValueError(f'Invalid data for product {pid}: {e}')
if set(revenue) != set(products) or set(demand) != set(products) or set(inventory) != set(products):
    raise ValueError('Missing data for some products.')
m = gp.Model('Retail_Store_Inventory')
x = m.addVars(products, lb=0, ub={i: min(demand[i], inventory[i]) for i in products}, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x[i] <= demand[i] for i in products), name='')
m.addConstrs((x[i] <= inventory[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')