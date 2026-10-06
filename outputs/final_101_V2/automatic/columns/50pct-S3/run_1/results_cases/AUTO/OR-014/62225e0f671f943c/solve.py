LEGACY_OBSERVATION = 'capacity.csv\nShelfID,previous_period_capacity,Capacity\n1,4.91750,5.0\n2,7.69790,7.0\n3,5.03040,6.0\n4,8.08720,8.0\n5,4.67775,5.5\n6,9.99450,9.0\n7,5.55945,6.5\n8,8.8200,7.5\n9,9.40294,8.2\n10,6.66729,5.7\n\nproducts.csv\nprevious_period_unit_value,ProductName,Value,previous_period_stock_status,Weight\n236,Smartphone,200,Overstock,1.0\n1289,Laptop,1500,Balanced,5.0\n93,Headphones,100,Stockout,0.5\n788,Camera,800,Overstock,2.0\n279,Smartwatch,250,Overstock,0.3\n589,Tablet,600,Overstock,1.5\n169,Bluetooth Speaker,150,Stockout,1.0\n67,Keyboard,80,Stockout,0.8\n55,Mouse,50,Stockout,0.2\n329,Monitor,300,Overstock,3.0\n462,Printer,400,Overstock,4.0\n141,External Hard Drive,120,Balanced,0.5\n58,Router,60,Balanced,0.3\n38,Power Bank,40,Stockout,0.4\n32,Memory Card,30,Overstock,0.05\n24,USB Flash Drive,25,Stockout,0.02\n92,Smart Home Hub,100,Stockout,0.6\n510,Gaming Console,500,Stockout,4.0\n80,Fitness Tracker,90,Overstock,0.2\n165,E-Reader,180,Overstock,0.5'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'ShelfID': '1', 'previous_period_capacity': '4.91750', 'Capacity': '5.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '2', 'previous_period_capacity': '7.69790', 'Capacity': '7.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '3', 'previous_period_capacity': '5.03040', 'Capacity': '6.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '4', 'previous_period_capacity': '8.08720', 'Capacity': '8.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '5', 'previous_period_capacity': '4.67775', 'Capacity': '5.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '6', 'previous_period_capacity': '9.99450', 'Capacity': '9.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '7', 'previous_period_capacity': '5.55945', 'Capacity': '6.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '8', 'previous_period_capacity': '8.8200', 'Capacity': '7.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '9', 'previous_period_capacity': '9.40294', 'Capacity': '8.2'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '10', 'previous_period_capacity': '6.66729', 'Capacity': '5.7'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '236', 'ProductName': 'Smartphone', 'Value': '200', 'previous_period_stock_status': 'Overstock', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '1289', 'ProductName': 'Laptop', 'Value': '1500', 'previous_period_stock_status': 'Balanced', 'Weight': '5.0'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '93', 'ProductName': 'Headphones', 'Value': '100', 'previous_period_stock_status': 'Stockout', 'Weight': '0.5'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '788', 'ProductName': 'Camera', 'Value': '800', 'previous_period_stock_status': 'Overstock', 'Weight': '2.0'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '279', 'ProductName': 'Smartwatch', 'Value': '250', 'previous_period_stock_status': 'Overstock', 'Weight': '0.3'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '589', 'ProductName': 'Tablet', 'Value': '600', 'previous_period_stock_status': 'Overstock', 'Weight': '1.5'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '169', 'ProductName': 'Bluetooth Speaker', 'Value': '150', 'previous_period_stock_status': 'Stockout', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '67', 'ProductName': 'Keyboard', 'Value': '80', 'previous_period_stock_status': 'Stockout', 'Weight': '0.8'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '55', 'ProductName': 'Mouse', 'Value': '50', 'previous_period_stock_status': 'Stockout', 'Weight': '0.2'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '329', 'ProductName': 'Monitor', 'Value': '300', 'previous_period_stock_status': 'Overstock', 'Weight': '3.0'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '462', 'ProductName': 'Printer', 'Value': '400', 'previous_period_stock_status': 'Overstock', 'Weight': '4.0'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '141', 'ProductName': 'External Hard Drive', 'Value': '120', 'previous_period_stock_status': 'Balanced', 'Weight': '0.5'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '58', 'ProductName': 'Router', 'Value': '60', 'previous_period_stock_status': 'Balanced', 'Weight': '0.3'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '38', 'ProductName': 'Power Bank', 'Value': '40', 'previous_period_stock_status': 'Stockout', 'Weight': '0.4'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '32', 'ProductName': 'Memory Card', 'Value': '30', 'previous_period_stock_status': 'Overstock', 'Weight': '0.05'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '24', 'ProductName': 'USB Flash Drive', 'Value': '25', 'previous_period_stock_status': 'Stockout', 'Weight': '0.02'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '92', 'ProductName': 'Smart Home Hub', 'Value': '100', 'previous_period_stock_status': 'Stockout', 'Weight': '0.6'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '510', 'ProductName': 'Gaming Console', 'Value': '500', 'previous_period_stock_status': 'Stockout', 'Weight': '4.0'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '80', 'ProductName': 'Fitness Tracker', 'Value': '90', 'previous_period_stock_status': 'Overstock', 'Weight': '0.2'}}, {'source': 'products.csv', 'values': {'previous_period_unit_value': '165', 'ProductName': 'E-Reader', 'Value': '180', 'previous_period_stock_status': 'Overstock', 'Weight': '0.5'}}]
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
    raise ValueError('Missing shelves or products in LEGACY_RECORDS')
for s in shelves:
    if s not in capacity:
        raise ValueError(f'Missing capacity for shelf {s}')
for p in products:
    if p not in value or p not in weight:
        raise ValueError(f'Missing value or weight for product {p}')
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