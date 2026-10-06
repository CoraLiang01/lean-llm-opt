LEGACY_OBSERVATION = 'capacity.csv\n\nShelfID,archive_revision_number,Capacity\n1,2,5.0\n2,9,7.0\n3,9,6.0\n4,7,8.0\n5,3,5.5\n6,3,9.0\n7,1,6.5\n8,2,7.5\n9,3,8.2\n10,4,5.7\n\nproducts.csv\n\narchive_revision_number,ProductName,Value,record_keeper_group,Weight\n8,Smartphone,200,Team A,1.0\n8,Laptop,1500,Team C,5.0\n8,Headphones,100,Team B,0.5\n1,Camera,800,Team C,2.0\n9,Smartwatch,250,Team B,0.3\n4,Tablet,600,Team C,1.5\n9,Bluetooth Speaker,150,Team C,1.0\n4,Keyboard,80,Team C,0.8\n3,Mouse,50,Team C,0.2\n1,Monitor,300,Team B,3.0\n7,Printer,400,Team B,4.0\n2,External Hard Drive,120,Team C,0.5\n5,Router,60,Team A,0.3\n6,Power Bank,40,Team B,0.4\n1,Memory Card,30,Team B,0.05\n2,USB Flash Drive,25,Team A,0.02\n8,Smart Home Hub,100,Team B,0.6\n9,Gaming Console,500,Team B,4.0\n6,Fitness Tracker,90,Team B,0.2\n1,E-Reader,180,Team C,0.5'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'ShelfID': '1', 'archive_revision_number': '2', 'Capacity': '5.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '2', 'archive_revision_number': '9', 'Capacity': '7.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '3', 'archive_revision_number': '9', 'Capacity': '6.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '4', 'archive_revision_number': '7', 'Capacity': '8.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '5', 'archive_revision_number': '3', 'Capacity': '5.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '6', 'archive_revision_number': '3', 'Capacity': '9.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '7', 'archive_revision_number': '1', 'Capacity': '6.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '8', 'archive_revision_number': '2', 'Capacity': '7.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '9', 'archive_revision_number': '3', 'Capacity': '8.2'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '10', 'archive_revision_number': '4', 'Capacity': '5.7'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '8', 'ProductName': 'Smartphone', 'Value': '200', 'record_keeper_group': 'Team A', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '8', 'ProductName': 'Laptop', 'Value': '1500', 'record_keeper_group': 'Team C', 'Weight': '5.0'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '8', 'ProductName': 'Headphones', 'Value': '100', 'record_keeper_group': 'Team B', 'Weight': '0.5'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '1', 'ProductName': 'Camera', 'Value': '800', 'record_keeper_group': 'Team C', 'Weight': '2.0'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '9', 'ProductName': 'Smartwatch', 'Value': '250', 'record_keeper_group': 'Team B', 'Weight': '0.3'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '4', 'ProductName': 'Tablet', 'Value': '600', 'record_keeper_group': 'Team C', 'Weight': '1.5'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '9', 'ProductName': 'Bluetooth Speaker', 'Value': '150', 'record_keeper_group': 'Team C', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '4', 'ProductName': 'Keyboard', 'Value': '80', 'record_keeper_group': 'Team C', 'Weight': '0.8'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '3', 'ProductName': 'Mouse', 'Value': '50', 'record_keeper_group': 'Team C', 'Weight': '0.2'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '1', 'ProductName': 'Monitor', 'Value': '300', 'record_keeper_group': 'Team B', 'Weight': '3.0'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '7', 'ProductName': 'Printer', 'Value': '400', 'record_keeper_group': 'Team B', 'Weight': '4.0'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '2', 'ProductName': 'External Hard Drive', 'Value': '120', 'record_keeper_group': 'Team C', 'Weight': '0.5'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '5', 'ProductName': 'Router', 'Value': '60', 'record_keeper_group': 'Team A', 'Weight': '0.3'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '6', 'ProductName': 'Power Bank', 'Value': '40', 'record_keeper_group': 'Team B', 'Weight': '0.4'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '1', 'ProductName': 'Memory Card', 'Value': '30', 'record_keeper_group': 'Team B', 'Weight': '0.05'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '2', 'ProductName': 'USB Flash Drive', 'Value': '25', 'record_keeper_group': 'Team A', 'Weight': '0.02'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '8', 'ProductName': 'Smart Home Hub', 'Value': '100', 'record_keeper_group': 'Team B', 'Weight': '0.6'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '9', 'ProductName': 'Gaming Console', 'Value': '500', 'record_keeper_group': 'Team B', 'Weight': '4.0'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '6', 'ProductName': 'Fitness Tracker', 'Value': '90', 'record_keeper_group': 'Team B', 'Weight': '0.2'}}, {'source': 'products.csv', 'values': {'archive_revision_number': '1', 'ProductName': 'E-Reader', 'Value': '180', 'record_keeper_group': 'Team C', 'Weight': '0.5'}}]
import gurobipy as gp
from gurobipy import GRB
shelves = []
capacity = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        shelf = rec['values']['ShelfID']
        shelves.append(shelf)
        capacity[shelf] = float(rec['values']['Capacity'])
products = []
value = {}
weight = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        prod = rec['values']['ProductName']
        products.append(prod)
        value[prod] = float(rec['values']['Value'])
        weight[prod] = float(rec['values']['Weight'])
if len(shelves) == 0 or len(products) == 0:
    raise RuntimeError('Missing shelves or products data.')
for s in shelves:
    if s not in capacity:
        raise RuntimeError(f'Missing capacity for shelf {s}.')
for p in products:
    if p not in value or p not in weight:
        raise RuntimeError(f'Missing value or weight for product {p}.')
m = gp.Model('Shelf_Product_Allocation')
x = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x[s, p] for s in shelves for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[p] * x[s, p] for p in products)) <= capacity[s] for s in shelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')