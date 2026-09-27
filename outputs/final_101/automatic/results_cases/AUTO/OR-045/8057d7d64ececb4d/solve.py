LEGACY_OBSERVATION = '{"values": {"Capacity": "1035"}}\n\n{"values": {"ProductName": "Spinach", "Weight": "282", "Value": "49"}}\n\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "83", "Value": "30"}}\n\n{"values": {"ProductName": "Apples", "Weight": "251", "Value": "30"}}\n\n{"values": {"ProductName": "Carrots", "Weight": "257", "Value": "18"}}\n\n{"values": {"ProductName": "Basil", "Weight": "88", "Value": "54"}}\n\n{"values": {"ProductName": "Potatoes", "Weight": "52", "Value": "27"}}\n\n{"values": {"ProductName": "Green Beans", "Weight": "198", "Value": "91"}}\n\n{"values": {"ProductName": "Blueberries", "Weight": "203", "Value": "88"}}\n\n{"values": {"ProductName": "Oranges", "Weight": "87", "Value": "78"}}\n\n{"values": {"ProductName": "Watermelons", "Weight": "265", "Value": "22"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Capacity': '1035'}}, {'source': '', 'values': {'ProductName': 'Spinach', 'Weight': '282', 'Value': '49'}}, {'source': '', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '83', 'Value': '30'}}, {'source': '', 'values': {'ProductName': 'Apples', 'Weight': '251', 'Value': '30'}}, {'source': '', 'values': {'ProductName': 'Carrots', 'Weight': '257', 'Value': '18'}}, {'source': '', 'values': {'ProductName': 'Basil', 'Weight': '88', 'Value': '54'}}, {'source': '', 'values': {'ProductName': 'Potatoes', 'Weight': '52', 'Value': '27'}}, {'source': '', 'values': {'ProductName': 'Green Beans', 'Weight': '198', 'Value': '91'}}, {'source': '', 'values': {'ProductName': 'Blueberries', 'Weight': '203', 'Value': '88'}}, {'source': '', 'values': {'ProductName': 'Oranges', 'Weight': '87', 'Value': '78'}}, {'source': '', 'values': {'ProductName': 'Watermelons', 'Weight': '265', 'Value': '22'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
capacity = None
products = []
for rec in records:
    if rec['source'] == '' and 'Capacity' in rec['values']:
        capacity = int(rec['values']['Capacity'])
    elif rec['source'] == '' and 'ProductName' in rec['values']:
        products.append({'ProductName': rec['values']['ProductName'], 'Weight': int(rec['values']['Weight']), 'Value': int(rec['values']['Value'])})
if capacity is None:
    raise ValueError('Missing total capacity in LEGACY_RECORDS')
if len(products) == 0:
    raise ValueError('No products found in LEGACY_RECORDS')
product_names = [p['ProductName'] for p in products]
weights = {p['ProductName']: p['Weight'] for p in products}
values = {p['ProductName']: p['Value'] for p in products}
if set(weights.keys()) != set(product_names) or set(values.keys()) != set(product_names):
    raise ValueError('Mismatch in product identifiers between weights and values')
m = gp.Model('ProduceOrder')
x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x[p] for p in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x[p] for p in product_names)) <= capacity, name='total_weight')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')