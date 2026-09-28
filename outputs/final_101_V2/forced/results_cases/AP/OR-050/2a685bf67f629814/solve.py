import gurobipy as gp
from gurobipy import GRB
displays = [{'ShelfID': '1', 'Capacity': 5.0}, {'ShelfID': '2', 'Capacity': 7.0}, {'ShelfID': '3', 'Capacity': 6.0}, {'ShelfID': '4', 'Capacity': 8.0}, {'ShelfID': '5', 'Capacity': 5.5}, {'ShelfID': '6', 'Capacity': 9.0}, {'ShelfID': '7', 'Capacity': 6.5}, {'ShelfID': '8', 'Capacity': 7.5}, {'ShelfID': '9', 'Capacity': 8.2}, {'ShelfID': '10', 'Capacity': 5.7}]
products = [{'ProductName': 'Smartphone', 'Value': 200, 'Weight': 1.0}, {'ProductName': 'Laptop', 'Value': 1500, 'Weight': 5.0}, {'ProductName': 'Headphones', 'Value': 100, 'Weight': 0.5}, {'ProductName': 'Camera', 'Value': 800, 'Weight': 2.0}, {'ProductName': 'Smartwatch', 'Value': 250, 'Weight': 0.3}, {'ProductName': 'Tablet', 'Value': 600, 'Weight': 1.5}, {'ProductName': 'Bluetooth Speaker', 'Value': 150, 'Weight': 1.0}, {'ProductName': 'Keyboard', 'Value': 80, 'Weight': 0.8}, {'ProductName': 'Mouse', 'Value': 50, 'Weight': 0.2}, {'ProductName': 'Monitor', 'Value': 300, 'Weight': 3.0}, {'ProductName': 'Printer', 'Value': 400, 'Weight': 4.0}, {'ProductName': 'External Hard Drive', 'Value': 120, 'Weight': 0.5}, {'ProductName': 'Router', 'Value': 60, 'Weight': 0.3}, {'ProductName': 'Power Bank', 'Value': 40, 'Weight': 0.4}, {'ProductName': 'Memory Card', 'Value': 30, 'Weight': 0.05}, {'ProductName': 'USB Flash Drive', 'Value': 25, 'Weight': 0.02}, {'ProductName': 'Smart Home Hub', 'Value': 100, 'Weight': 0.6}, {'ProductName': 'Gaming Console', 'Value': 500, 'Weight': 4.0}, {'ProductName': 'Fitness Tracker', 'Value': 90, 'Weight': 0.2}, {'ProductName': 'E-Reader', 'Value': 180, 'Weight': 0.5}]
display_ids = [d['ShelfID'] for d in displays]
product_names = [p['ProductName'] for p in products]
C = {d['ShelfID']: d['Capacity'] for d in displays}
v = {p['ProductName']: p['Value'] for p in products}
w = {p['ProductName']: p['Weight'] for p in products}
if len(display_ids) != 10:
    raise ValueError('Expected 10 displays, got %d' % len(display_ids))
if len(product_names) != 20:
    raise ValueError('Expected 20 products, got %d' % len(product_names))
for d in display_ids:
    if d not in C:
        raise ValueError(f'Missing capacity for display {d}')
for p in product_names:
    if p not in v or p not in w:
        raise ValueError(f'Missing value or weight for product {p}')
m = gp.Model('Retail_Display_Allocation')
x = m.addVars(display_ids, product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((v[p] * x[d, p] for d in display_ids for p in product_names)), GRB.MAXIMIZE)
for d in display_ids:
    m.addConstr(gp.quicksum((w[p] * x[d, p] for p in product_names)) <= C[d], name=f'cap_{d}')
m.addConstr(gp.quicksum((x[d, product_names[0]] for d in display_ids)) >= 5, name='min_smartphone')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')