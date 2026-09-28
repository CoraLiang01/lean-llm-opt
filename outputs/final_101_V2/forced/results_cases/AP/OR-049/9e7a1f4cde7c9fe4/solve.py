import gurobipy as gp
from gurobipy import GRB
shelves = [{'ShelfID': '1', 'Capacity': 5.0}, {'ShelfID': '2', 'Capacity': 7.0}, {'ShelfID': '3', 'Capacity': 6.0}, {'ShelfID': '4', 'Capacity': 8.0}, {'ShelfID': '5', 'Capacity': 5.5}, {'ShelfID': '6', 'Capacity': 9.0}, {'ShelfID': '7', 'Capacity': 6.5}, {'ShelfID': '8', 'Capacity': 7.5}, {'ShelfID': '9', 'Capacity': 8.2}, {'ShelfID': '10', 'Capacity': 5.7}]
products = [{'ProductName': 'Smartphone', 'Value': 200, 'Weight': 1.0}, {'ProductName': 'Laptop', 'Value': 1500, 'Weight': 5.0}, {'ProductName': 'Headphones', 'Value': 100, 'Weight': 0.5}, {'ProductName': 'Camera', 'Value': 800, 'Weight': 2.0}, {'ProductName': 'Smartwatch', 'Value': 250, 'Weight': 0.3}, {'ProductName': 'Tablet', 'Value': 600, 'Weight': 1.5}, {'ProductName': 'Bluetooth Speaker', 'Value': 150, 'Weight': 1.0}, {'ProductName': 'Keyboard', 'Value': 80, 'Weight': 0.8}, {'ProductName': 'Mouse', 'Value': 50, 'Weight': 0.2}, {'ProductName': 'Monitor', 'Value': 300, 'Weight': 3.0}, {'ProductName': 'Printer', 'Value': 400, 'Weight': 4.0}, {'ProductName': 'External Hard Drive', 'Value': 120, 'Weight': 0.5}, {'ProductName': 'Router', 'Value': 60, 'Weight': 0.3}, {'ProductName': 'Power Bank', 'Value': 40, 'Weight': 0.4}, {'ProductName': 'Memory Card', 'Value': 30, 'Weight': 0.05}, {'ProductName': 'USB Flash Drive', 'Value': 25, 'Weight': 0.02}, {'ProductName': 'Smart Home Hub', 'Value': 100, 'Weight': 0.6}, {'ProductName': 'Gaming Console', 'Value': 500, 'Weight': 4.0}, {'ProductName': 'Fitness Tracker', 'Value': 90, 'Weight': 0.2}, {'ProductName': 'E-Reader', 'Value': 180, 'Weight': 0.5}]
shelf_ids = [s['ShelfID'] for s in shelves]
product_names = [p['ProductName'] for p in products]
C = {s['ShelfID']: s['Capacity'] for s in shelves}
v = {p['ProductName']: p['Value'] for p in products}
w = {p['ProductName']: p['Weight'] for p in products}
if len(shelf_ids) != 10:
    raise ValueError('Expected 10 shelves, got %d' % len(shelf_ids))
if len(product_names) != 20:
    raise ValueError('Expected 20 products, got %d' % len(product_names))
if set(C.keys()) != set(shelf_ids):
    raise ValueError('Shelf capacity keys do not match shelf IDs')
if set(v.keys()) != set(product_names) or set(w.keys()) != set(product_names):
    raise ValueError('Product value/weight keys do not match product names')
m = gp.Model('Shelf_Product_Allocation')
x = m.addVars(shelf_ids, product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((v[j] * x[i, j] for i in shelf_ids for j in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((w[j] * x[i, j] for j in product_names)) <= C[i] for i in shelf_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')