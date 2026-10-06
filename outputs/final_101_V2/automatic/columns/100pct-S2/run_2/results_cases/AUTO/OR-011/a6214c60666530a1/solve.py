LEGACY_OBSERVATION = '{"values": {"refrigeration_service_visits": "8", "Capacity": "875"}}\n\n{"values": {"ProductName": "Spinach", "Weight": "230", "supplier_catalog_revision_count": "3", "supplier_quality_rating": "3.2", "Value": "64", "catalog_display_group": "Featured"}}\n\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "637", "supplier_catalog_revision_count": "3", "supplier_quality_rating": "4.1", "Value": "75", "catalog_display_group": "Featured"}}\n\n{"values": {"ProductName": "Apples", "Weight": "773", "supplier_catalog_revision_count": "1", "supplier_quality_rating": "4.7", "Value": "68", "catalog_display_group": "Seasonal"}}\n\n{"values": {"ProductName": "Carrots", "Weight": "653", "supplier_catalog_revision_count": "3", "supplier_quality_rating": "4.7", "Value": "11", "catalog_display_group": "Featured"}}\n\n{"values": {"ProductName": "Basil", "Weight": "755", "supplier_catalog_revision_count": "6", "supplier_quality_rating": "4.1", "Value": "91", "catalog_display_group": "Seasonal"}}\n\n{"values": {"ProductName": "Potatoes", "Weight": "670", "supplier_catalog_revision_count": "6", "supplier_quality_rating": "3.5", "Value": "31", "catalog_display_group": "Everyday"}}\n\n{"values": {"ProductName": "Green Beans", "Weight": "505", "supplier_catalog_revision_count": "1", "supplier_quality_rating": "4.1", "Value": "90", "catalog_display_group": "Featured"}}\n\n{"values": {"ProductName": "Blueberries", "Weight": "821", "supplier_catalog_revision_count": "6", "supplier_quality_rating": "3.5", "Value": "56", "catalog_display_group": "Seasonal"}}\n\n{"values": {"ProductName": "Oranges", "Weight": "83", "supplier_catalog_revision_count": "4", "supplier_quality_rating": "3.2", "Value": "10", "catalog_display_group": "Seasonal"}}\n\n{"values": {"ProductName": "Watermelons", "Weight": "249", "supplier_catalog_revision_count": "3", "supplier_quality_rating": "4.4", "Value": "24", "catalog_display_group": "Everyday"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'refrigeration_service_visits': '8', 'Capacity': '875'}}, {'source': '', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '3.2', 'Value': '64', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '4.1', 'Value': '75', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Apples', 'Weight': '773', 'supplier_catalog_revision_count': '1', 'supplier_quality_rating': '4.7', 'Value': '68', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '4.7', 'Value': '11', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Basil', 'Weight': '755', 'supplier_catalog_revision_count': '6', 'supplier_quality_rating': '4.1', 'Value': '91', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'supplier_catalog_revision_count': '6', 'supplier_quality_rating': '3.5', 'Value': '31', 'catalog_display_group': 'Everyday'}}, {'source': '', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'supplier_catalog_revision_count': '1', 'supplier_quality_rating': '4.1', 'Value': '90', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'supplier_catalog_revision_count': '6', 'supplier_quality_rating': '3.5', 'Value': '56', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'supplier_catalog_revision_count': '4', 'supplier_quality_rating': '3.2', 'Value': '10', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '4.4', 'Value': '24', 'catalog_display_group': 'Everyday'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
products = []
capacity = None
for rec in records:
    vals = rec['values']
    if 'ProductName' in vals and 'Value' in vals and ('Weight' in vals):
        products.append({'ProductName': vals['ProductName'], 'Value': int(vals['Value']), 'Weight': int(vals['Weight'])})
    if 'Capacity' in vals:
        capacity = int(vals['Capacity'])
if capacity is None:
    raise ValueError('No capacity found in LEGACY_RECORDS')
if len(products) == 0:
    raise ValueError('No products found in LEGACY_RECORDS')
product_names = [p['ProductName'] for p in products]
values = {p['ProductName']: p['Value'] for p in products}
weights = {p['ProductName']: p['Weight'] for p in products}
if set(values.keys()) != set(product_names) or set(weights.keys()) != set(product_names):
    raise ValueError('Mismatch in product identifiers and coefficients')
m = gp.Model('Supermarket_Stock')
x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x[i] for i in product_names)) <= capacity, name='stock_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')