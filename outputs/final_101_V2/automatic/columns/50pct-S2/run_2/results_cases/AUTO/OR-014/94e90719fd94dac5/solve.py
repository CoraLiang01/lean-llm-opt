LEGACY_OBSERVATION = 'capacity.csv\nShelfID,store_weekly_visitor_count,Capacity\n1,1820,5.0\n2,1820,7.0\n3,850,6.0\n4,850,8.0\n5,1460,5.5\n6,850,9.0\n7,1820,6.5\n8,850,7.5\n9,1460,8.2\n10,1820,5.7\n\nproducts.csv\nproduct_catalog_page_views,ProductName,Value,supplier_service_tier,Weight\n1380,Smartphone,200,Priority,1.0\n180,Laptop,1500,Priority,5.0\n340,Headphones,100,Premium,0.5\n180,Camera,800,Standard,2.0\n340,Smartwatch,250,Premium,0.3\n1040,Tablet,600,Standard,1.5\n1040,Bluetooth Speaker,150,Priority,1.0\n560,Keyboard,80,Priority,0.8\n560,Mouse,50,Priority,0.2\n340,Monitor,300,Priority,3.0\n340,Printer,400,Priority,4.0\n560,External Hard Drive,120,Priority,0.5\n180,Router,60,Premium,0.3\n1380,Power Bank,40,Priority,0.4\n560,Memory Card,30,Premium,0.05\n560,USB Flash Drive,25,Priority,0.02\n180,Smart Home Hub,100,Premium,0.6\n180,Gaming Console,500,Standard,4.0\n790,Fitness Tracker,90,Priority,0.2\n180,E-Reader,180,Priority,0.5'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'ShelfID': '1', 'store_weekly_visitor_count': '1820', 'Capacity': '5.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '2', 'store_weekly_visitor_count': '1820', 'Capacity': '7.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '3', 'store_weekly_visitor_count': '850', 'Capacity': '6.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '4', 'store_weekly_visitor_count': '850', 'Capacity': '8.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '5', 'store_weekly_visitor_count': '1460', 'Capacity': '5.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '6', 'store_weekly_visitor_count': '850', 'Capacity': '9.0'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '7', 'store_weekly_visitor_count': '1820', 'Capacity': '6.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '8', 'store_weekly_visitor_count': '850', 'Capacity': '7.5'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '9', 'store_weekly_visitor_count': '1460', 'Capacity': '8.2'}}, {'source': 'capacity.csv', 'values': {'ShelfID': '10', 'store_weekly_visitor_count': '1820', 'Capacity': '5.7'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '1380', 'ProductName': 'Smartphone', 'Value': '200', 'supplier_service_tier': 'Priority', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '180', 'ProductName': 'Laptop', 'Value': '1500', 'supplier_service_tier': 'Priority', 'Weight': '5.0'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '340', 'ProductName': 'Headphones', 'Value': '100', 'supplier_service_tier': 'Premium', 'Weight': '0.5'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '180', 'ProductName': 'Camera', 'Value': '800', 'supplier_service_tier': 'Standard', 'Weight': '2.0'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '340', 'ProductName': 'Smartwatch', 'Value': '250', 'supplier_service_tier': 'Premium', 'Weight': '0.3'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '1040', 'ProductName': 'Tablet', 'Value': '600', 'supplier_service_tier': 'Standard', 'Weight': '1.5'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '1040', 'ProductName': 'Bluetooth Speaker', 'Value': '150', 'supplier_service_tier': 'Priority', 'Weight': '1.0'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '560', 'ProductName': 'Keyboard', 'Value': '80', 'supplier_service_tier': 'Priority', 'Weight': '0.8'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '560', 'ProductName': 'Mouse', 'Value': '50', 'supplier_service_tier': 'Priority', 'Weight': '0.2'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '340', 'ProductName': 'Monitor', 'Value': '300', 'supplier_service_tier': 'Priority', 'Weight': '3.0'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '340', 'ProductName': 'Printer', 'Value': '400', 'supplier_service_tier': 'Priority', 'Weight': '4.0'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '560', 'ProductName': 'External Hard Drive', 'Value': '120', 'supplier_service_tier': 'Priority', 'Weight': '0.5'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '180', 'ProductName': 'Router', 'Value': '60', 'supplier_service_tier': 'Premium', 'Weight': '0.3'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '1380', 'ProductName': 'Power Bank', 'Value': '40', 'supplier_service_tier': 'Priority', 'Weight': '0.4'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '560', 'ProductName': 'Memory Card', 'Value': '30', 'supplier_service_tier': 'Premium', 'Weight': '0.05'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '560', 'ProductName': 'USB Flash Drive', 'Value': '25', 'supplier_service_tier': 'Priority', 'Weight': '0.02'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '180', 'ProductName': 'Smart Home Hub', 'Value': '100', 'supplier_service_tier': 'Premium', 'Weight': '0.6'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '180', 'ProductName': 'Gaming Console', 'Value': '500', 'supplier_service_tier': 'Standard', 'Weight': '4.0'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '790', 'ProductName': 'Fitness Tracker', 'Value': '90', 'supplier_service_tier': 'Priority', 'Weight': '0.2'}}, {'source': 'products.csv', 'values': {'product_catalog_page_views': '180', 'ProductName': 'E-Reader', 'Value': '180', 'supplier_service_tier': 'Priority', 'Weight': '0.5'}}]
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
    raise RuntimeError('Missing shelves or products data.')
for s in shelves:
    if s not in capacity:
        raise RuntimeError(f'Missing capacity for shelf {s}')
for p in products:
    if p not in value or p not in weight:
        raise RuntimeError(f'Missing value or weight for product {p}')
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