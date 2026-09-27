LEGACY_OBSERVATION = '{"values": {"Capacity": "875"}}\n\n{"values": {"ProductName": "Spinach", "Weight": "230", "Value": "64"}}\n\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "637", "Value": "75"}}\n\n{"values": {"ProductName": "Apples", "Weight": "773", "Value": "68"}}\n\n{"values": {"ProductName": "Carrots", "Weight": "653", "Value": "11"}}\n\n{"values": {"ProductName": "Basil", "Weight": "755", "Value": "91"}}\n\n{"values": {"ProductName": "Potatoes", "Weight": "670", "Value": "31"}}\n\n{"values": {"ProductName": "Green Beans", "Weight": "505", "Value": "90"}}\n\n{"values": {"ProductName": "Blueberries", "Weight": "821", "Value": "56"}}\n\n{"values": {"ProductName": "Oranges", "Weight": "83", "Value": "10"}}\n\n{"values": {"ProductName": "Watermelons", "Weight": "249", "Value": "24"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '875'}}, {'source': '', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'Value': '64'}}, {'source': '', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'Value': '75'}}, {'source': '', 'values': {'ProductName': 'Apples', 'Weight': '773', 'Value': '68'}}, {'source': '', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'Value': '11'}}, {'source': '', 'values': {'ProductName': 'Basil', 'Weight': '755', 'Value': '91'}}, {'source': '', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'Value': '31'}}, {'source': '', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'Value': '90'}}, {'source': '', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'Value': '56'}}, {'source': '', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'Value': '10'}}, {'source': '', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'Value': '24'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
capacity = None
products = []
weights = {}
values = {}
for rec in records:
    vals = rec['values']
    if 'Capacity' in vals:
        if capacity is not None:
            raise ValueError('Multiple capacities found')
        capacity = int(vals['Capacity'])
    elif 'ProductName' in vals and 'Weight' in vals and ('Value' in vals):
        pname = vals['ProductName']
        products.append(pname)
        weights[pname] = int(vals['Weight'])
        values[pname] = int(vals['Value'])
if capacity is None:
    raise ValueError('No capacity found')
if set(weights.keys()) != set(products) or set(values.keys()) != set(products):
    raise ValueError('Missing product data')
m = gp.Model('SupermarketOrder')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x[p] for p in products)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')