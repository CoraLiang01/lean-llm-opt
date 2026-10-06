LEGACY_OBSERVATION = 'capacity.csv\nShelfID,Capacity\n1,5.0\n2,7.0\n3,6.0\n4,8.0\n5,5.5\n6,9.0\n7,6.5\n8,7.5\n9,8.2\n10,5.7\n\nproducts.csv\nProductName,Value,Weight\nSmartphone,200,1.0\nLaptop,1500,5.0\nHeadphones,100,0.5\nCamera,800,2.0\nSmartwatch,250,0.3\nTablet,600,1.5\nBluetooth Speaker,150,1.0\nKeyboard,80,0.8\nMouse,50,0.2\nMonitor,300,3.0\nPrinter,400,4.0\nExternal Hard Drive,120,0.5\nRouter,60,0.3\nPower Bank,40,0.4\nMemory Card,30,0.05\nUSB Flash Drive,25,0.02\nSmart Home Hub,100,0.6\nGaming Console,500,4.0\nFitness Tracker,90,0.2\nE-Reader,180,0.5'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'ShelfID': '1', 'Capacity': '5.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '2', 'Capacity': '7.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '3', 'Capacity': '6.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '4', 'Capacity': '8.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '5', 'Capacity': '5.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '6', 'Capacity': '9.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '7', 'Capacity': '6.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '8', 'Capacity': '7.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '9', 'Capacity': '8.2'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '10', 'Capacity': '5.7'}}, {'source': 'products.csv', 'values': {'ProductName': 'Smartphone', 'Value': '200', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Laptop', 'Value': '1500', 'Weight': '5.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Headphones', 'Value': '100', 'Weight': '0.5'}}, {'source': 'products.csv', 'values': {'ProductName': 'Camera', 'Value': '800', 'Weight': '2.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Smartwatch', 'Value': '250', 'Weight': '0.3'}}, {'source': 'products.csv', 'values': {'ProductName': 'Tablet', 'Value': '600', 'Weight': '1.5'}}, {'source': 'products.csv', 'values': {'ProductName': 'Bluetooth Speaker', 'Value': '150', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Keyboard', 'Value': '80', 'Weight': '0.8'}}, {'source': 'products.csv', 'values': {'ProductName': 'Mouse', 'Value': '50', 'Weight': '0.2'}}, {'source': 'products.csv', 'values': {'ProductName': 'Monitor', 'Value': '300', 'Weight': '3.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Printer', 'Value': '400', 'Weight': '4.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'External Hard Drive', 'Value': '120', 'Weight': '0.5'}}, {'source': 'products.csv', 'values': {'ProductName': 'Router', 'Value': '60', 'Weight': '0.3'}}, {'source': 'products.csv', 'values': {'ProductName': 'Power Bank', 'Value': '40', 'Weight': '0.4'}}, {'source': 'products.csv', 'values': {'ProductName': 'Memory Card', 'Value': '30', 'Weight': '0.05'}}, {'source': 'products.csv', 'values': {'ProductName': 'USB Flash Drive', 'Value': '25', 'Weight': '0.02'}}, {'source': 'products.csv', 'values': {'ProductName': 'Smart Home Hub', 'Value': '100', 'Weight': '0.6'}}, {'source': 'products.csv', 'values': {'ProductName': 'Gaming Console', 'Value': '500', 'Weight': '4.0'}}, {'source': 'products.csv', 'values': {'ProductName': 'Fitness Tracker', 'Value': '90', 'Weight': '0.2'}}, {'source': 'products.csv', 'values': {'ProductName': 'E-Reader', 'Value': '180', 'Weight': '0.5'}}]
import gurobipy as gp
from gurobipy import GRB
capacity_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
displays = [rec['values']['ShelfID'] for rec in capacity_records]
capacities = {rec['values']['ShelfID']: float(rec['values']['Capacity']) for rec in capacity_records}
products = [rec['values']['ProductName'] for rec in product_records]
values = {rec['values']['ProductName']: float(rec['values']['Value']) for rec in product_records}
weights = {rec['values']['ProductName']: float(rec['values']['Weight']) for rec in product_records}
m = gp.Model('retail_display_allocation')
x = m.addVars(displays, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x[d, p] for d in displays for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[p] * x[d, p] for p in products)) <= capacities[d] for d in displays), name='')
first_product = products[0]
m.addConstr(gp.quicksum((x[d, first_product] for d in displays)) >= 5, name='min_smartphone')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')