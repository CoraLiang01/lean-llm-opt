LEGACY_OBSERVATION = '{"values": {"ShelfID": "1", "Capacity": "5.0"}}\n{"values": {"ShelfID": "2", "Capacity": "7.0"}}\n{"values": {"ShelfID": "3", "Capacity": "6.0"}}\n{"values": {"ShelfID": "4", "Capacity": "8.0"}}\n{"values": {"ShelfID": "5", "Capacity": "5.5"}}\n{"values": {"ShelfID": "6", "Capacity": "9.0"}}\n{"values": {"ShelfID": "7", "Capacity": "6.5"}}\n{"values": {"ShelfID": "8", "Capacity": "7.5"}}\n{"values": {"ShelfID": "9", "Capacity": "8.2"}}\n{"values": {"ShelfID": "10", "Capacity": "5.7"}}\n{"values": {"ProductName": "Smartphone", "Value": "200", "Weight": "1.0"}}\n{"values": {"ProductName": "Laptop", "Value": "1500", "Weight": "5.0"}}\n{"values": {"ProductName": "Headphones", "Value": "100", "Weight": "0.5"}}\n{"values": {"ProductName": "Camera", "Value": "800", "Weight": "2.0"}}\n{"values": {"ProductName": "Smartwatch", "Value": "250", "Weight": "0.3"}}\n{"values": {"ProductName": "Tablet", "Value": "600", "Weight": "1.5"}}\n{"values": {"ProductName": "Bluetooth Speaker", "Value": "150", "Weight": "1.0"}}\n{"values": {"ProductName": "Keyboard", "Value": "80", "Weight": "0.8"}}\n{"values": {"ProductName": "Mouse", "Value": "50", "Weight": "0.2"}}\n{"values": {"ProductName": "Monitor", "Value": "300", "Weight": "3.0"}}\n{"values": {"ProductName": "Printer", "Value": "400", "Weight": "4.0"}}\n{"values": {"ProductName": "External Hard Drive", "Value": "120", "Weight": "0.5"}}\n{"values": {"ProductName": "Router", "Value": "60", "Weight": "0.3"}}\n{"values": {"ProductName": "Power Bank", "Value": "40", "Weight": "0.4"}}\n{"values": {"ProductName": "Memory Card", "Value": "30", "Weight": "0.05"}}\n{"values": {"ProductName": "USB Flash Drive", "Value": "25", "Weight": "0.02"}}\n{"values": {"ProductName": "Smart Home Hub", "Value": "100", "Weight": "0.6"}}\n{"values": {"ProductName": "Gaming Console", "Value": "500", "Weight": "4.0"}}\n{"values": {"ProductName": "Fitness Tracker", "Value": "90", "Weight": "0.2"}}\n{"values": {"ProductName": "E-Reader", "Value": "180", "Weight": "0.5"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'ShelfID': '1', 'Capacity': '5.0'}}, {'source': '', 'values': {'ShelfID': '2', 'Capacity': '7.0'}}, {'source': '', 'values': {'ShelfID': '3', 'Capacity': '6.0'}}, {'source': '', 'values': {'ShelfID': '4', 'Capacity': '8.0'}}, {'source': '', 'values': {'ShelfID': '5', 'Capacity': '5.5'}}, {'source': '', 'values': {'ShelfID': '6', 'Capacity': '9.0'}}, {'source': '', 'values': {'ShelfID': '7', 'Capacity': '6.5'}}, {'source': '', 'values': {'ShelfID': '8', 'Capacity': '7.5'}}, {'source': '', 'values': {'ShelfID': '9', 'Capacity': '8.2'}}, {'source': '', 'values': {'ShelfID': '10', 'Capacity': '5.7'}}, {'source': '', 'values': {'ProductName': 'Smartphone', 'Value': '200', 'Weight': '1.0'}}, {'source': '', 'values': {'ProductName': 'Laptop', 'Value': '1500', 'Weight': '5.0'}}, {'source': '', 'values': {'ProductName': 'Headphones', 'Value': '100', 'Weight': '0.5'}}, {'source': '', 'values': {'ProductName': 'Camera', 'Value': '800', 'Weight': '2.0'}}, {'source': '', 'values': {'ProductName': 'Smartwatch', 'Value': '250', 'Weight': '0.3'}}, {'source': '', 'values': {'ProductName': 'Tablet', 'Value': '600', 'Weight': '1.5'}}, {'source': '', 'values': {'ProductName': 'Bluetooth Speaker', 'Value': '150', 'Weight': '1.0'}}, {'source': '', 'values': {'ProductName': 'Keyboard', 'Value': '80', 'Weight': '0.8'}}, {'source': '', 'values': {'ProductName': 'Mouse', 'Value': '50', 'Weight': '0.2'}}, {'source': '', 'values': {'ProductName': 'Monitor', 'Value': '300', 'Weight': '3.0'}}, {'source': '', 'values': {'ProductName': 'Printer', 'Value': '400', 'Weight': '4.0'}}, {'source': '', 'values': {'ProductName': 'External Hard Drive', 'Value': '120', 'Weight': '0.5'}}, {'source': '', 'values': {'ProductName': 'Router', 'Value': '60', 'Weight': '0.3'}}, {'source': '', 'values': {'ProductName': 'Power Bank', 'Value': '40', 'Weight': '0.4'}}, {'source': '', 'values': {'ProductName': 'Memory Card', 'Value': '30', 'Weight': '0.05'}}, {'source': '', 'values': {'ProductName': 'USB Flash Drive', 'Value': '25', 'Weight': '0.02'}}, {'source': '', 'values': {'ProductName': 'Smart Home Hub', 'Value': '100', 'Weight': '0.6'}}, {'source': '', 'values': {'ProductName': 'Gaming Console', 'Value': '500', 'Weight': '4.0'}}, {'source': '', 'values': {'ProductName': 'Fitness Tracker', 'Value': '90', 'Weight': '0.2'}}, {'source': '', 'values': {'ProductName': 'E-Reader', 'Value': '180', 'Weight': '0.5'}}]
import gurobipy as gp
from gurobipy import GRB
shelf_records = [r for r in LEGACY_RECORDS if 'ShelfID' in r['values']]
product_records = [r for r in LEGACY_RECORDS if 'ProductName' in r['values']]
shelves = []
capacity = {}
for rec in shelf_records:
    sid = rec['values']['ShelfID']
    shelves.append(sid)
    capacity[sid] = float(rec['values']['Capacity'])
products = []
value = {}
weight = {}
for rec in product_records:
    pname = rec['values']['ProductName']
    products.append(pname)
    value[pname] = float(rec['values']['Value'])
    weight[pname] = float(rec['values']['Weight'])
if set(capacity.keys()) != set(shelves):
    raise ValueError('Shelf capacity data missing for some shelves')
if set(value.keys()) != set(products) or set(weight.keys()) != set(products):
    raise ValueError('Product value/weight data missing for some products')
first_product = products[0]
m = gp.Model('retail_display_allocation')
x = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i] for i in shelves), name='')
m.addConstr(gp.quicksum((x[i, first_product] for i in shelves)) >= 5, name='min_first')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')