LEGACY_OBSERVATION = '{"values": {"refrigeration_service_visits": "8", "Capacity": "875"}}\n{"values": {"ProductName": "Spinach", "Weight": "230", "supplier_catalog_revision_count": "3", "supplier_quality_rating": "3.2", "Value": "64", "catalog_display_group": "Featured"}}\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "637", "supplier_catalog_revision_count": "3", "supplier_quality_rating": "4.1", "Value": "75", "catalog_display_group": "Featured"}}\n{"values": {"ProductName": "Apples", "Weight": "773", "supplier_catalog_revision_count": "1", "supplier_quality_rating": "4.7", "Value": "68", "catalog_display_group": "Seasonal"}}\n{"values": {"ProductName": "Carrots", "Weight": "653", "supplier_catalog_revision_count": "3", "supplier_quality_rating": "4.7", "Value": "11", "catalog_display_group": "Featured"}}\n{"values": {"ProductName": "Basil", "Weight": "755", "supplier_catalog_revision_count": "6", "supplier_quality_rating": "4.1", "Value": "91", "catalog_display_group": "Seasonal"}}\n{"values": {"ProductName": "Potatoes", "Weight": "670", "supplier_catalog_revision_count": "6", "supplier_quality_rating": "3.5", "Value": "31", "catalog_display_group": "Everyday"}}\n{"values": {"ProductName": "Green Beans", "Weight": "505", "supplier_catalog_revision_count": "1", "supplier_quality_rating": "4.1", "Value": "90", "catalog_display_group": "Featured"}}\n{"values": {"ProductName": "Blueberries", "Weight": "821", "supplier_catalog_revision_count": "6", "supplier_quality_rating": "3.5", "Value": "56", "catalog_display_group": "Seasonal"}}\n{"values": {"ProductName": "Oranges", "Weight": "83", "supplier_catalog_revision_count": "4", "supplier_quality_rating": "3.2", "Value": "10", "catalog_display_group": "Seasonal"}}\n{"values": {"ProductName": "Watermelons", "Weight": "249", "supplier_catalog_revision_count": "3", "supplier_quality_rating": "4.4", "Value": "24", "catalog_display_group": "Everyday"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'refrigeration_service_visits': '8', 'Capacity': '875'}}, {'source': '', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '3.2', 'Value': '64', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '4.1', 'Value': '75', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Apples', 'Weight': '773', 'supplier_catalog_revision_count': '1', 'supplier_quality_rating': '4.7', 'Value': '68', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '4.7', 'Value': '11', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Basil', 'Weight': '755', 'supplier_catalog_revision_count': '6', 'supplier_quality_rating': '4.1', 'Value': '91', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'supplier_catalog_revision_count': '6', 'supplier_quality_rating': '3.5', 'Value': '31', 'catalog_display_group': 'Everyday'}}, {'source': '', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'supplier_catalog_revision_count': '1', 'supplier_quality_rating': '4.1', 'Value': '90', 'catalog_display_group': 'Featured'}}, {'source': '', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'supplier_catalog_revision_count': '6', 'supplier_quality_rating': '3.5', 'Value': '56', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'supplier_catalog_revision_count': '4', 'supplier_quality_rating': '3.2', 'Value': '10', 'catalog_display_group': 'Seasonal'}}, {'source': '', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'supplier_catalog_revision_count': '3', 'supplier_quality_rating': '4.4', 'Value': '24', 'catalog_display_group': 'Everyday'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
value = {}
weight = {}
capacity = None
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'ProductName' in vals:
        pname = vals['ProductName']
        products.append(pname)
        if 'Value' not in vals or 'Weight' not in vals:
            raise ValueError(f'Missing Value or Weight for product {pname}')
        value[pname] = int(vals['Value'])
        weight[pname] = int(vals['Weight'])
    if 'Capacity' in vals and vals['Capacity']:
        if capacity is not None and int(vals['Capacity']) != capacity:
            raise ValueError('Multiple conflicting capacities found')
        capacity = int(vals['Capacity'])
if capacity is None:
    raise ValueError('No capacity found in LEGACY_RECORDS')
m = gp.Model('Supermarket_Stock_Optimization')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x[p] for p in products)) <= capacity, name='stock_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')