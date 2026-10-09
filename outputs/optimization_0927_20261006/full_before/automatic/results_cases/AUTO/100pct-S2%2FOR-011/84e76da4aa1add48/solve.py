LEGACY_OBSERVATION = '{"values": {"refrigeration_service_visits": "8", "Capacity": "875"}}\n\n{"values": {"ProductName": "Spinach", "Weight": "230", "supplier_catalog_revision_count": "3", "supplier_quality_rating": "3.2", "Value": "64", "catalog_display_group": "Featured"}}\n\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "637", "supplier_catalog_revision_count": "3", "supplier_quality_rating": "4.1", "Value": "75", "catalog_display_group": "Featured"}}\n\n{"values": {"ProductName": "Apples", "Weight": "773", "supplier_catalog_revision_count": "1", "supplier_quality_rating": "4.7", "Value": "68", "catalog_display_group": "Seasonal"}}\n\n{"values": {"ProductName": "Carrots", "Weight": "653", "supplier_catalog_revision_count": "3", "supplier_quality_rating": "4.7", "Value": "11", "catalog_display_group": "Featured"}}\n\n{"values": {"ProductName": "Basil", "Weight": "755", "supplier_catalog_revision_count": "6", "supplier_quality_rating": "4.1", "Value": "91", "catalog_display_group": "Seasonal"}}\n\n{"values": {"ProductName": "Potatoes", "Weight": "670", "supplier_catalog_revision_count": "6", "supplier_quality_rating": "3.5", "Value": "31", "catalog_display_group": "Everyday"}}\n\n{"values": {"ProductName": "Green Beans", "Weight": "505", "supplier_catalog_revision_count": "1", "supplier_quality_rating": "4.1", "Value": "90", "catalog_display_group": "Featured"}}\n\n{"values": {"ProductName": "Blueberries", "Weight": "821", "supplier_catalog_revision_count": "6", "supplier_quality_rating": "3.5", "Value": "56", "catalog_display_group": "Seasonal"}}\n\n{"values": {"ProductName": "Oranges", "Weight": "83", "supplier_catalog_revision_count": "4", "supplier_quality_rating": "3.2", "Value": "10", "catalog_display_group": "Seasonal"}}\n\n{"values": {"ProductName": "Watermelons", "Weight": "249", "supplier_catalog_revision_count": "3", "supplier_quality_rating": "4.4", "Value": "24", "catalog_display_group": "Everyday"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'refrigeration_service_visits': '8', 'Capacity': '875'}}, {'source': '', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '3.2', 'Value': '64', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '4.1', 'Value': '75', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Apples', 'Weight': '773', 'supplier_catalog_revision_count': '1', 'supplier_quality_rating': '4.7', 'Value': '68', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '4.7', 'Value': '11', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Basil', 'Weight': '755', 'supplier_catalog_revision_count': '6', 'supplier_quality_rating': '4.1', 'Value': '91', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'supplier_catalog_revision_count': '6', 'supplier_quality_rating': '3.5', 'Value': '31', 'catalog_display_group': 'Everyday'}}, {'source': '', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'supplier_catalog_revision_count': '1', 'supplier_quality_rating': '4.1', 'Value': '90', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'supplier_catalog_revision_count': '6', 'supplier_quality_rating': '3.5', 'Value': '56', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'supplier_catalog_revision_count': '4', 'supplier_quality_rating': '3.2', 'Value': '10', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '4.4', 'Value': '24', 'catalog_display_group': 'Everyday'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
capacity = None
for rec in records:
    vals = rec['values']
    if 'Capacity' in vals:
        capacity = int(vals['Capacity'])
        break
if capacity is None:
    raise ValueError('Missing Capacity in LEGACY_RECORDS')
products = []
weights = {}
values = {}
for rec in records:
    vals = rec['values']
    if 'ProductName' in vals and 'Weight' in vals and ('Value' in vals):
        pname = vals['ProductName']
        products.append(pname)
        weights[pname] = int(vals['Weight'])
        values[pname] = int(vals['Value'])
if not len(products) == len(weights) == len(values):
    raise ValueError('Mismatch in product, weight, or value data')
m = gp.Model('SupermarketReplenishment')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x[p] for p in products)) <= capacity, name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')