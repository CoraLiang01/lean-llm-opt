LEGACY_OBSERVATION = '{"values": {"ShelfID": "1", "Capacity": "5.0"}}\n{"values": {"ShelfID": "2", "Capacity": "7.0"}}\n{"values": {"ShelfID": "3", "Capacity": "6.0"}}\n{"values": {"ShelfID": "4", "Capacity": "8.0"}}\n{"values": {"ShelfID": "5", "Capacity": "5.5"}}\n{"values": {"ShelfID": "6", "Capacity": "9.0"}}\n{"values": {"ShelfID": "7", "Capacity": "6.5"}}\n{"values": {"ShelfID": "8", "Capacity": "7.5"}}\n{"values": {"ShelfID": "9", "Capacity": "8.2"}}\n{"values": {"ShelfID": "10", "Capacity": "5.7"}}\n{"values": {"ProductName": "Smartphone", "Value": "200", "Weight": "1.0"}}\n{"values": {"ProductName": "Laptop", "Value": "1500", "Weight": "5.0"}}\n{"values": {"ProductName": "Headphones", "Value": "100", "Weight": "0.5"}}\n{"values": {"ProductName": "Camera", "Value": "800", "Weight": "2.0"}}\n{"values": {"ProductName": "Smartwatch", "Value": "250", "Weight": "0.3"}}\n{"values": {"ProductName": "Tablet", "Value": "600", "Weight": "1.5"}}\n{"values": {"ProductName": "Bluetooth Speaker", "Value": "150", "Weight": "1.0"}}\n{"values": {"ProductName": "Keyboard", "Value": "80", "Weight": "0.8"}}\n{"values": {"ProductName": "Mouse", "Value": "50", "Weight": "0.2"}}\n{"values": {"ProductName": "Monitor", "Value": "300", "Weight": "3.0"}}\n{"values": {"ProductName": "Printer", "Value": "400", "Weight": "4.0"}}\n{"values": {"ProductName": "External Hard Drive", "Value": "120", "Weight": "0.5"}}\n{"values": {"ProductName": "Router", "Value": "60", "Weight": "0.3"}}\n{"values": {"ProductName": "Power Bank", "Value": "40", "Weight": "0.4"}}\n{"values": {"ProductName": "Memory Card", "Value": "30", "Weight": "0.05"}}\n{"values": {"ProductName": "USB Flash Drive", "Value": "25", "Weight": "0.02"}}\n{"values": {"ProductName": "Smart Home Hub", "Value": "100", "Weight": "0.6"}}\n{"values": {"ProductName": "Gaming Console", "Value": "500", "Weight": "4.0"}}\n{"values": {"ProductName": "Fitness Tracker", "Value": "90", "Weight": "0.2"}}\n{"values": {"ProductName": "E-Reader", "Value": "180", "Weight": "0.5"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'ShelfID': '1', 'Capacity': '5.0'}}, {'source': '', 'values': {'ShelfID': '2', 'Capacity': '7.0'}}, {'source': '', 'values': {'ShelfID': '3', 'Capacity': '6.0'}}, {'source': '', 'values': {'ShelfID': '4', 'Capacity': '8.0'}}, {'source': '', 'values': {'ShelfID': '5', 'Capacity': '5.5'}}, {'source': '', 'values': {'ShelfID': '6', 'Capacity': '9.0'}}, {'source': '', 'values': {'ShelfID': '7', 'Capacity': '6.5'}}, {'source': '', 'values': {'ShelfID': '8', 'Capacity': '7.5'}}, {'source': '', 'values': {'ShelfID': '9', 'Capacity': '8.2'}}, {'source': '', 'values': {'ShelfID': '10', 'Capacity': '5.7'}}, {'source': '', 'values': {'ProductName': 'Smartphone', 'Value': '200', 'Weight': '1.0'}}, {'source': '', 'values': {'ProductName': 'Laptop', 'Value': '1500', 'Weight': '5.0'}}, {'source': '', 'values': {'ProductName': 'Headphones', 'Value': '100', 'Weight': '0.5'}}, {'source': '', 'values': {'ProductName': 'Camera', 'Value': '800', 'Weight': '2.0'}}, {'source': '', 'values': {'ProductName': 'Smartwatch', 'Value': '250', 'Weight': '0.3'}}, {'source': '', 'values': {'ProductName': 'Tablet', 'Value': '600', 'Weight': '1.5'}}, {'source': '', 'values': {'ProductName': 'Bluetooth Speaker', 'Value': '150', 'Weight': '1.0'}}, {'source': '', 'values': {'ProductName': 'Keyboard', 'Value': '80', 'Weight': '0.8'}}, {'source': '', 'values': {'ProductName': 'Mouse', 'Value': '50', 'Weight': '0.2'}}, {'source': '', 'values': {'ProductName': 'Monitor', 'Value': '300', 'Weight': '3.0'}}, {'source': '', 'values': {'ProductName': 'Printer', 'Value': '400', 'Weight': '4.0'}}, {'source': '', 'values': {'ProductName': 'External Hard Drive', 'Value': '120', 'Weight': '0.5'}}, {'source': '', 'values': {'ProductName': 'Router', 'Value': '60', 'Weight': '0.3'}}, {'source': '', 'values': {'ProductName': 'Power Bank', 'Value': '40', 'Weight': '0.4'}}, {'source': '', 'values': {'ProductName': 'Memory Card', 'Value': '30', 'Weight': '0.05'}}, {'source': '', 'values': {'ProductName': 'USB Flash Drive', 'Value': '25', 'Weight': '0.02'}}, {'source': '', 'values': {'ProductName': 'Smart Home Hub', 'Value': '100', 'Weight': '0.6'}}, {'source': '', 'values': {'ProductName': 'Gaming Console', 'Value': '500', 'Weight': '4.0'}}, {'source': '', 'values': {'ProductName': 'Fitness Tracker', 'Value': '90', 'Weight': '0.2'}}, {'source': '', 'values': {'ProductName': 'E-Reader', 'Value': '180', 'Weight': '0.5'}}]
import gurobipy as gp
from gurobipy import GRB
shelves = []
capacities = {}
products = []
values = {}
weights = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'ShelfID' in v and 'Capacity' in v:
        shelf = v['ShelfID']
        shelves.append(shelf)
        capacities[shelf] = float(v['Capacity'])
    if 'ProductName' in v and 'Value' in v and ('Weight' in v):
        prod = v['ProductName']
        products.append(prod)
        values[prod] = float(v['Value'])
        weights[prod] = float(v['Weight'])
if len(shelves) == 0 or len(products) == 0:
    raise ValueError('Missing shelves or products in LEGACY_RECORDS')
for shelf in shelves:
    if shelf not in capacities:
        raise ValueError(f'Missing capacity for shelf {shelf}')
for prod in products:
    if prod not in values or prod not in weights:
        raise ValueError(f'Missing value or weight for product {prod}')
m = gp.Model('Shelf_Product_Allocation')
x = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[prod] * x[shelf, prod] for shelf in shelves for prod in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[prod] * x[shelf, prod] for prod in products)) <= capacities[shelf] for shelf in shelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')