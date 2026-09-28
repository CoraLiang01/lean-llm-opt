LEGACY_OBSERVATION = '{"values": {"Capacity": "1035"}}\n\n{"values": {"ProductName": "Spinach", "Weight": "282", "Value": "49"}}\n\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "83", "Value": "30"}}\n\n{"values": {"ProductName": "Apples", "Weight": "251", "Value": "30"}}\n\n{"values": {"ProductName": "Carrots", "Weight": "257", "Value": "18"}}\n\n{"values": {"ProductName": "Basil", "Weight": "88", "Value": "54"}}\n\n{"values": {"ProductName": "Potatoes", "Weight": "52", "Value": "27"}}\n\n{"values": {"ProductName": "Green Beans", "Weight": "198", "Value": "91"}}\n\n{"values": {"ProductName": "Blueberries", "Weight": "203", "Value": "88"}}\n\n{"values": {"ProductName": "Oranges", "Weight": "87", "Value": "78"}}\n\n{"values": {"ProductName": "Watermelons", "Weight": "265", "Value": "22"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '1035'}}, {'source': '', 'values': {'ProductName': 'Spinach', 'Weight': '282', 'Value': '49'}}, {'source': '', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '83', 'Value': '30'}}, {'source': '', 'values': {'ProductName': 'Apples', 'Weight': '251', 'Value': '30'}}, {'source': '', 'values': {'ProductName': 'Carrots', 'Weight': '257', 'Value': '18'}}, {'source': '', 'values': {'ProductName': 'Basil', 'Weight': '88', 'Value': '54'}}, {'source': '', 'values': {'ProductName': 'Potatoes', 'Weight': '52', 'Value': '27'}}, {'source': '', 'values': {'ProductName': 'Green Beans', 'Weight': '198', 'Value': '91'}}, {'source': '', 'values': {'ProductName': 'Blueberries', 'Weight': '203', 'Value': '88'}}, {'source': '', 'values': {'ProductName': 'Oranges', 'Weight': '87', 'Value': '78'}}, {'source': '', 'values': {'ProductName': 'Watermelons', 'Weight': '265', 'Value': '22'}}]
import gurobipy as gp
from gurobipy import GRB
capacity = None
products = []
weights = {}
values = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'Capacity' in vals:
        if capacity is not None:
            raise ValueError('Multiple capacities found in LEGACY_RECORDS')
        capacity = int(vals['Capacity'])
    elif 'ProductName' in vals and 'Weight' in vals and ('Value' in vals):
        pname = vals['ProductName']
        products.append(pname)
        weights[pname] = int(vals['Weight'])
        values[pname] = int(vals['Value'])
if capacity is None:
    raise ValueError('No capacity found in LEGACY_RECORDS')
if set(weights.keys()) != set(products) or set(values.keys()) != set(products):
    raise ValueError('Missing product data in LEGACY_RECORDS')
m = gp.Model('produce_restock')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x[p] for p in products)) <= capacity, name='total_weight')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')