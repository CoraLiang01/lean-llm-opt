LEGACY_OBSERVATION = '{"values": {"ProductName": "Spinach", "Weight": "230", "supplier_quality_rating": "3.2", "Value": "64", "catalog_display_group": "Featured"}}\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "637", "supplier_quality_rating": "4.1", "Value": "75", "catalog_display_group": "Featured"}}\n{"values": {"ProductName": "Apples", "Weight": "773", "supplier_quality_rating": "4.7", "Value": "68", "catalog_display_group": "Seasonal"}}\n{"values": {"ProductName": "Carrots", "Weight": "653", "supplier_quality_rating": "4.7", "Value": "11", "catalog_display_group": "Featured"}}\n{"values": {"ProductName": "Basil", "Weight": "755", "supplier_quality_rating": "4.1", "Value": "91", "catalog_display_group": "Seasonal"}}\n{"values": {"ProductName": "Potatoes", "Weight": "670", "supplier_quality_rating": "3.5", "Value": "31", "catalog_display_group": "Everyday"}}\n{"values": {"ProductName": "Green Beans", "Weight": "505", "supplier_quality_rating": "4.1", "Value": "90", "catalog_display_group": "Featured"}}\n{"values": {"ProductName": "Blueberries", "Weight": "821", "supplier_quality_rating": "3.5", "Value": "56", "catalog_display_group": "Seasonal"}}\n{"values": {"ProductName": "Oranges", "Weight": "83", "supplier_quality_rating": "3.2", "Value": "10", "catalog_display_group": "Seasonal"}}\n{"values": {"ProductName": "Watermelons", "Weight": "249", "supplier_quality_rating": "4.4", "Value": "24", "catalog_display_group": "Everyday"}}\n{"values": {"refrigeration_service_visits": "8", "Capacity": "875"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'supplier_quality_rating': '3.2', 'Value': '64', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'supplier_quality_rating': '4.1', 'Value': '75', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Apples', 'Weight': '773', 'supplier_quality_rating': '4.7', 'Value': '68', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'supplier_quality_rating': '4.7', 'Value': '11', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Basil', 'Weight': '755', 'supplier_quality_rating': '4.1', 'Value': '91', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'supplier_quality_rating': '3.5', 'Value': '31', 'catalog_display_group': 'Everyday'}}, {'source': '', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'supplier_quality_rating': '4.1', 'Value': '90', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'supplier_quality_rating': '3.5', 'Value': '56', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'supplier_quality_rating': '3.2', 'Value': '10', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'supplier_quality_rating': '4.4', 'Value': '24', 'catalog_display_group': 'Everyday'}}, {'source': '', 'values': {'refrigeration_service_visits': '8', 'Capacity': '875'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
product_records = [r for r in records if 'ProductName' in r['values']]
capacity_records = [r for r in records if 'Capacity' in r['values']]
if len(product_records) != 10:
    raise ValueError(f'Expected 10 product records, got {len(product_records)}')
if len(capacity_records) != 1:
    raise ValueError(f'Expected 1 capacity record, got {len(capacity_records)}')
products = [r['values']['ProductName'] for r in product_records]
value = {r['values']['ProductName']: int(r['values']['Value']) for r in product_records}
weight = {r['values']['ProductName']: int(r['values']['Weight']) for r in product_records}
capacity = int(capacity_records[0]['values']['Capacity'])
m = gp.Model('Supermarket_Stock_Optimization')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x[p] for p in products)) <= capacity, name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')