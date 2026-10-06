LEGACY_OBSERVATION = '{"values": {"refrigeration_service_visits": "8", "Capacity": "875"}}\n{"values": {"ProductName": "Spinach", "Weight": "230", "supplier_quality_rating": "3.2", "Value": "64", "catalog_display_group": "Featured"}}\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "637", "supplier_quality_rating": "4.1", "Value": "75", "catalog_display_group": "Featured"}}\n{"values": {"ProductName": "Apples", "Weight": "773", "supplier_quality_rating": "4.7", "Value": "68", "catalog_display_group": "Seasonal"}}\n{"values": {"ProductName": "Carrots", "Weight": "653", "supplier_quality_rating": "4.7", "Value": "11", "catalog_display_group": "Featured"}}\n{"values": {"ProductName": "Basil", "Weight": "755", "supplier_quality_rating": "4.1", "Value": "91", "catalog_display_group": "Seasonal"}}\n{"values": {"ProductName": "Potatoes", "Weight": "670", "supplier_quality_rating": "3.5", "Value": "31", "catalog_display_group": "Everyday"}}\n{"values": {"ProductName": "Green Beans", "Weight": "505", "supplier_quality_rating": "4.1", "Value": "90", "catalog_display_group": "Featured"}}\n{"values": {"ProductName": "Blueberries", "Weight": "821", "supplier_quality_rating": "3.5", "Value": "56", "catalog_display_group": "Seasonal"}}\n{"values": {"ProductName": "Oranges", "Weight": "83", "supplier_quality_rating": "3.2", "Value": "10", "catalog_display_group": "Seasonal"}}\n{"values": {"ProductName": "Watermelons", "Weight": "249", "supplier_quality_rating": "4.4", "Value": "24", "catalog_display_group": "Everyday"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'refrigeration_service_visits': '8', 'Capacity': '875'}}, {'source': '', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'supplier_quality_rating': '3.2', 'Value': '64', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'supplier_quality_rating': '4.1', 'Value': '75', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Apples', 'Weight': '773', 'supplier_quality_rating': '4.7', 'Value': '68', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'supplier_quality_rating': '4.7', 'Value': '11', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Basil', 'Weight': '755', 'supplier_quality_rating': '4.1', 'Value': '91', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'supplier_quality_rating': '3.5', 'Value': '31', 'catalog_display_group': 'Everyday'}}, {'source': '', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'supplier_quality_rating': '4.1', 'Value': '90', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'supplier_quality_rating': '3.5', 'Value': '56', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'supplier_quality_rating': '3.2', 'Value': '10', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'supplier_quality_rating': '4.4', 'Value': '24', 'catalog_display_group': 'Everyday'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
values = {}
weights = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'ProductName' in v:
        pname = v['ProductName']
        products.append(pname)
        if 'Value' not in v or 'Weight' not in v:
            raise ValueError(f'Missing Value or Weight for product {pname}')
        values[pname] = int(v['Value'])
        weights[pname] = int(v['Weight'])
capacity = None
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'Capacity' in v:
        capacity = int(v['Capacity'])
if capacity is None:
    raise ValueError('Missing Capacity in LEGACY_RECORDS')
if len(products) != len(values) or len(products) != len(weights):
    raise ValueError('Mismatch in product, value, or weight data')
m = gp.Model('SupermarketStock')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x[p] for p in products)) <= capacity, name='stock_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')