LEGACY_OBSERVATION = 'capacity.csv\nShelfID,ArchiveRevisionCount,Capacity,ArchivePageCount\n1,16,5,23\n2,17,7,20\n3,18,6,28\n4,12,8,18\n5,7,5.5,19\n6,1,9,29\n7,27,6.5,9\n8,20,7.5,14\n9,22,8.2,26\n10,28,5.7,12\n\nproducts.csv\nArchivePageCount,ArchiveRevisionCount,ArchiveFolder,ProductName,Value,Weight\n4,15,Folder_C,Smartphone,200,1\n7,15,Folder_B,Laptop,1500,5\n25,5,Folder_C,Headphones,100,0.5\n25,8,Folder_B,Camera,800,2\n30,13,Folder_A,Smartwatch,250,0.3\n16,28,Folder_C,Tablet,600,1.5\n24,23,Folder_B,Bluetooth Speaker,150,1\n10,4,Folder_B,Keyboard,80,0.8\n23,22,Folder_B,Mouse,50,0.2\n9,16,Folder_C,Monitor,300,3\n11,29,Folder_B,Printer,400,4\n22,12,Folder_B,External Hard Drive,120,0.5\n28,19,Folder_A,Router,60,0.3\n5,14,Folder_C,Power Bank,40,0.4\n22,5,Folder_B,Memory Card,30,0.05\n7,29,Folder_C,USB Flash Drive,25,0.02\n29,20,Folder_A,Smart Home Hub,100,0.6\n8,30,Folder_A,Gaming Console,500,4\n19,16,Folder_A,Fitness Tracker,90,0.2\n21,18,Folder_A,E-Reader,180,0.5'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'ShelfID': '1', 'ArchiveRevisionCount': '16', 'Capacity': '5', 'ArchivePageCount': '23'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '2', 'ArchiveRevisionCount': '17', 'Capacity': '7', 'ArchivePageCount': '20'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '3', 'ArchiveRevisionCount': '18', 'Capacity': '6', 'ArchivePageCount': '28'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '4', 'ArchiveRevisionCount': '12', 'Capacity': '8', 'ArchivePageCount': '18'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '5', 'ArchiveRevisionCount': '7', 'Capacity': '5.5', 'ArchivePageCount': '19'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '6', 'ArchiveRevisionCount': '1', 'Capacity': '9', 'ArchivePageCount': '29'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '7', 'ArchiveRevisionCount': '27', 'Capacity': '6.5', 'ArchivePageCount': '9'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '8', 'ArchiveRevisionCount': '20', 'Capacity': '7.5', 'ArchivePageCount': '14'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '9', 'ArchiveRevisionCount': '22', 'Capacity': '8.2', 'ArchivePageCount': '26'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '10', 'ArchiveRevisionCount': '28', 'Capacity': '5.7', 'ArchivePageCount': '12'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '4', 'ArchiveRevisionCount': '15', 'ArchiveFolder': 'Folder_C', 'ProductName': 'Smartphone', 'Value': '200', 'Weight': '1'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '7', 'ArchiveRevisionCount': '15', 'ArchiveFolder': 'Folder_B', 'ProductName': 'Laptop', 'Value': '1500', 'Weight': '5'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '25', 'ArchiveRevisionCount': '5', 'ArchiveFolder': 'Folder_C', 'ProductName': 'Headphones', 'Value': '100', 'Weight': '0.5'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '25', 'ArchiveRevisionCount': '8', 'ArchiveFolder': 'Folder_B', 'ProductName': 'Camera', 'Value': '800', 'Weight': '2'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '30', 'ArchiveRevisionCount': '13', 'ArchiveFolder': 'Folder_A', 'ProductName': 'Smartwatch', 'Value': '250', 'Weight': '0.3'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '16', 'ArchiveRevisionCount': '28', 'ArchiveFolder': 'Folder_C', 'ProductName': 'Tablet', 'Value': '600', 'Weight': '1.5'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '24', 'ArchiveRevisionCount': '23', 'ArchiveFolder': 'Folder_B', 'ProductName': 'Bluetooth Speaker', 'Value': '150', 'Weight': '1'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '10', 'ArchiveRevisionCount': '4', 'ArchiveFolder': 'Folder_B', 'ProductName': 'Keyboard', 'Value': '80', 'Weight': '0.8'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '23', 'ArchiveRevisionCount': '22', 'ArchiveFolder': 'Folder_B', 'ProductName': 'Mouse', 'Value': '50', 'Weight': '0.2'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '9', 'ArchiveRevisionCount': '16', 'ArchiveFolder': 'Folder_C', 'ProductName': 'Monitor', 'Value': '300', 'Weight': '3'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '11', 'ArchiveRevisionCount': '29', 'ArchiveFolder': 'Folder_B', 'ProductName': 'Printer', 'Value': '400', 'Weight': '4'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '22', 'ArchiveRevisionCount': '12', 'ArchiveFolder': 'Folder_B', 'ProductName': 'External Hard Drive', 'Value': '120', 'Weight': '0.5'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '28', 'ArchiveRevisionCount': '19', 'ArchiveFolder': 'Folder_A', 'ProductName': 'Router', 'Value': '60', 'Weight': '0.3'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '5', 'ArchiveRevisionCount': '14', 'ArchiveFolder': 'Folder_C', 'ProductName': 'Power Bank', 'Value': '40', 'Weight': '0.4'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '22', 'ArchiveRevisionCount': '5', 'ArchiveFolder': 'Folder_B', 'ProductName': 'Memory Card', 'Value': '30', 'Weight': '0.05'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '7', 'ArchiveRevisionCount': '29', 'ArchiveFolder': 'Folder_C', 'ProductName': 'USB Flash Drive', 'Value': '25', 'Weight': '0.02'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '29', 'ArchiveRevisionCount': '20', 'ArchiveFolder': 'Folder_A', 'ProductName': 'Smart Home Hub', 'Value': '100', 'Weight': '0.6'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '8', 'ArchiveRevisionCount': '30', 'ArchiveFolder': 'Folder_A', 'ProductName': 'Gaming Console', 'Value': '500', 'Weight': '4'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '19', 'ArchiveRevisionCount': '16', 'ArchiveFolder': 'Folder_A', 'ProductName': 'Fitness Tracker', 'Value': '90', 'Weight': '0.2'}}, {'source': 'products.csv', 'values': {'ArchivePageCount': '21', 'ArchiveRevisionCount': '18', 'ArchiveFolder': 'Folder_A', 'ProductName': 'E-Reader', 'Value': '180', 'Weight': '0.5'}}]
import gurobipy as gp
from gurobipy import GRB
shelves = []
capacity = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        shelf_id = rec['values']['ShelfID']
        shelves.append(shelf_id)
        capacity[shelf_id] = float(rec['values']['Capacity'])
products = []
value = {}
weight = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        pname = rec['values']['ProductName']
        products.append(pname)
        value[pname] = float(rec['values']['Value'])
        weight[pname] = float(rec['values']['Weight'])
if len(shelves) == 0 or len(products) == 0:
    raise ValueError('Missing shelves or products data.')
if len(capacity) != len(shelves):
    raise ValueError('Capacity data missing for some shelves.')
if len(value) != len(products) or len(weight) != len(products):
    raise ValueError('Value or weight data missing for some products.')
m = gp.Model('retail_display_allocation')
x = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x[s, p] for s in shelves for p in products)), GRB.MAXIMIZE)
for s in shelves:
    m.addConstr(gp.quicksum((weight[p] * x[s, p] for p in products)) <= capacity[s], name=f'cap_{s}')
first_product = products[0]
m.addConstr(gp.quicksum((x[s, first_product] for s in shelves)) >= 5, name='min_smartphone')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')