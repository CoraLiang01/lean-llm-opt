LEGACY_OBSERVATION = '{"values": {"Capacity": "875"}}\n\n{"values": {"ProductName": "Spinach", "Weight": "230", "Value": "64"}}\n\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "637", "Value": "75"}}\n\n{"values": {"ProductName": "Apples", "Weight": "773", "Value": "68"}}\n\n{"values": {"ProductName": "Carrots", "Weight": "653", "Value": "11"}}\n\n{"values": {"ProductName": "Basil", "Weight": "755", "Value": "91"}}\n\n{"values": {"ProductName": "Potatoes", "Weight": "670", "Value": "31"}}\n\n{"values": {"ProductName": "Green Beans", "Weight": "505", "Value": "90"}}\n\n{"values": {"ProductName": "Blueberries", "Weight": "821", "Value": "56"}}\n\n{"values": {"ProductName": "Oranges", "Weight": "83", "Value": "10"}}\n\n{"values": {"ProductName": "Watermelons", "Weight": "249", "Value": "24"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '875'}}, {'source': '', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'Value': '64'}}, {'source': '', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'Value': '75'}}, {'source': '', 'values': {'ProductName': 'Apples', 'Weight': '773', 'Value': '68'}}, {'source': '', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'Value': '11'}}, {'source': '', 'values': {'ProductName': 'Basil', 'Weight': '755', 'Value': '91'}}, {'source': '', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'Value': '31'}}, {'source': '', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'Value': '90'}}, {'source': '', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'Value': '56'}}, {'source': '', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'Value': '10'}}, {'source': '', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'Value': '24'}}]
import gurobipy as gp
from gurobipy import GRB
capacity = None
products = []
weights = {}
values = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'Capacity' in vals and vals['Capacity']:
        if capacity is not None:
            raise ValueError('Multiple capacities found in LEGACY_RECORDS')
        capacity = int(vals['Capacity'])
    elif 'ProductName' in vals and vals['ProductName']:
        pname = vals['ProductName']
        products.append(pname)
        if 'Weight' not in vals or 'Value' not in vals:
            raise ValueError(f'Missing Weight or Value for product {pname}')
        weights[pname] = int(vals['Weight'])
        values[pname] = int(vals['Value'])
if capacity is None:
    raise ValueError('No capacity found in LEGACY_RECORDS')
if set(products) != set(weights.keys()) or set(products) != set(values.keys()):
    raise ValueError('Mismatch in product, weight, or value keys')
m = gp.Model('Supermarket_Stock')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x[p] for p in products)) <= capacity, name='stock_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')