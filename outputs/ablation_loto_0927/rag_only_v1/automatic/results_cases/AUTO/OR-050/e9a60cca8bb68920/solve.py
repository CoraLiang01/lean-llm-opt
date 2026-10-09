LEGACY_OBSERVATION = '{"values": {"ShelfID": "1", "Capacity": "5.0"}}\n{"values": {"ShelfID": "2", "Capacity": "7.0"}}\n{"values": {"ShelfID": "3", "Capacity": "6.0"}}\n{"values": {"ShelfID": "4", "Capacity": "8.0"}}\n{"values": {"ShelfID": "5", "Capacity": "5.5"}}\n{"values": {"ShelfID": "6", "Capacity": "9.0"}}\n{"values": {"ShelfID": "7", "Capacity": "6.5"}}\n{"values": {"ShelfID": "8", "Capacity": "7.5"}}\n{"values": {"ShelfID": "9", "Capacity": "8.2"}}\n{"values": {"ShelfID": "10", "Capacity": "5.7"}}\n{"values": {"ProductName": "Smartphone", "Value": "200", "Weight": "1.0"}}\n{"values": {"ProductName": "Laptop", "Value": "1500", "Weight": "5.0"}}\n{"values": {"ProductName": "Headphones", "Value": "100", "Weight": "0.5"}}\n{"values": {"ProductName": "Camera", "Value": "800", "Weight": "2.0"}}\n{"values": {"ProductName": "Smartwatch", "Value": "250", "Weight": "0.3"}}\n{"values": {"ProductName": "Tablet", "Value": "600", "Weight": "1.5"}}\n{"values": {"ProductName": "Bluetooth Speaker", "Value": "150", "Weight": "1.0"}}\n{"values": {"ProductName": "Keyboard", "Value": "80", "Weight": "0.8"}}\n{"values": {"ProductName": "Mouse", "Value": "50", "Weight": "0.2"}}\n{"values": {"ProductName": "Monitor", "Value": "300", "Weight": "3.0"}}\n{"values": {"ProductName": "Printer", "Value": "400", "Weight": "4.0"}}\n{"values": {"ProductName": "External Hard Drive", "Value": "120", "Weight": "0.5"}}\n{"values": {"ProductName": "Router", "Value": "60", "Weight": "0.3"}}\n{"values": {"ProductName": "Power Bank", "Value": "40", "Weight": "0.4"}}\n{"values": {"ProductName": "Memory Card", "Value": "30", "Weight": "0.05"}}\n{"values": {"ProductName": "USB Flash Drive", "Value": "25", "Weight": "0.02"}}\n{"values": {"ProductName": "Smart Home Hub", "Value": "100", "Weight": "0.6"}}\n{"values": {"ProductName": "Gaming Console", "Value": "500", "Weight": "4.0"}}\n{"values": {"ProductName": "Fitness Tracker", "Value": "90", "Weight": "0.2"}}\n{"values": {"ProductName": "E-Reader", "Value": "180", "Weight": "0.5"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'ShelfID': '1', 'Capacity': '5.0'}}, {'source': '', 'values': {'ShelfID': '2', 'Capacity': '7.0'}}, {'source': '', 'values': {'ShelfID': '3', 'Capacity': '6.0'}}, {'source': '', 'values': {'ShelfID': '4', 'Capacity': '8.0'}}, {'source': '', 'values': {'ShelfID': '5', 'Capacity': '5.5'}}, {'source': '', 'values': {'ShelfID': '6', 'Capacity': '9.0'}}, {'source': '', 'values': {'ShelfID': '7', 'Capacity': '6.5'}}, {'source': '', 'values': {'ShelfID': '8', 'Capacity': '7.5'}}, {'source': '', 'values': {'ShelfID': '9', 'Capacity': '8.2'}}, {'source': '', 'values': {'ShelfID': '10', 'Capacity': '5.7'}}, {'source': '', 'values': {'ProductName': 'Smartphone', 'Value': '200', 'Weight': '1.0'}}, {'source': '', 'values': {'ProductName': 'Laptop', 'Value': '1500', 'Weight': '5.0'}}, {'source': '', 'values': {'ProductName': 'Headphones', 'Value': '100', 'Weight': '0.5'}}, {'source': '', 'values': {'ProductName': 'Camera', 'Value': '800', 'Weight': '2.0'}}, {'source': '', 'values': {'ProductName': 'Smartwatch', 'Value': '250', 'Weight': '0.3'}}, {'source': '', 'values': {'ProductName': 'Tablet', 'Value': '600', 'Weight': '1.5'}}, {'source': '', 'values': {'ProductName': 'Bluetooth Speaker', 'Value': '150', 'Weight': '1.0'}}, {'source': '', 'values': {'ProductName': 'Keyboard', 'Value': '80', 'Weight': '0.8'}}, {'source': '', 'values': {'ProductName': 'Mouse', 'Value': '50', 'Weight': '0.2'}}, {'source': '', 'values': {'ProductName': 'Monitor', 'Value': '300', 'Weight': '3.0'}}, {'source': '', 'values': {'ProductName': 'Printer', 'Value': '400', 'Weight': '4.0'}}, {'source': '', 'values': {'ProductName': 'External Hard Drive', 'Value': '120', 'Weight': '0.5'}}, {'source': '', 'values': {'ProductName': 'Router', 'Value': '60', 'Weight': '0.3'}}, {'source': '', 'values': {'ProductName': 'Power Bank', 'Value': '40', 'Weight': '0.4'}}, {'source': '', 'values': {'ProductName': 'Memory Card', 'Value': '30', 'Weight': '0.05'}}, {'source': '', 'values': {'ProductName': 'USB Flash Drive', 'Value': '25', 'Weight': '0.02'}}, {'source': '', 'values': {'ProductName': 'Smart Home Hub', 'Value': '100', 'Weight': '0.6'}}, {'source': '', 'values': {'ProductName': 'Gaming Console', 'Value': '500', 'Weight': '4.0'}}, {'source': '', 'values': {'ProductName': 'Fitness Tracker', 'Value': '90', 'Weight': '0.2'}}, {'source': '', 'values': {'ProductName': 'E-Reader', 'Value': '180', 'Weight': '0.5'}}]
from gurobipy import Model, GRB
shelf_caps = {}
product_list = []
product_vals = {}
product_wts = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'ShelfID' in v and 'Capacity' in v:
        shelf_caps[int(v['ShelfID'])] = float(v['Capacity'])
    elif 'ProductName' in v and 'Value' in v and ('Weight' in v):
        pname = v['ProductName']
        product_list.append(pname)
        product_vals[pname] = float(v['Value'])
        product_wts[pname] = float(v['Weight'])
shelves = sorted(shelf_caps.keys())
products = product_list
if len(products) != 20 or len(shelves) != 10:
    raise ValueError('Expected 10 shelves and 20 products from LEGACY_RECORDS.')
prod_idx = {p: j + 1 for (j, p) in enumerate(products)}
idx_prod = {j + 1: p for (j, p) in enumerate(products)}
m = Model()
m.Params.MIPGap = 0.0001
x = m.addVars(shelves, products, vtype=GRB.INTEGER, lb=0, name='')
for i in shelves:
    if i not in shelf_caps:
        raise ValueError(f'Missing capacity for shelf {i}')
for p in products:
    if p not in product_vals or p not in product_wts:
        raise ValueError(f'Missing value/weight for product {p}')
m.setObjective(sum((product_vals[p] * x[i, p] for i in shelves for p in products)), GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(sum((product_wts[p] * x[i, p] for p in products)) <= shelf_caps[i], name=f'cap_{i}')
first_product = products[0]
if first_product != 'Smartphone':
    raise ValueError('First product is not Smartphone as expected.')
m.addConstr(sum((x[i, first_product] for i in shelves)) >= 5, name='min_smartphone')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in shelves:
        for p in products:
            v = x[i, p]
            print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')