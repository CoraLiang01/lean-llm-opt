LEGACY_OBSERVATION = 'products.csv\nProductName,Weight,supplier_quality_rating,Value,catalog_display_group\nSpinach,230,3.2,64,Featured\nShiitake Mushrooms,637,4.1,75,Featured\nApples,773,4.7,68,Seasonal\nCarrots,653,4.7,11,Featured\nBasil,755,4.1,91,Seasonal\nPotatoes,670,3.5,31,Everyday\nGreen Beans,505,4.1,90,Featured\nBlueberries,821,3.5,56,Seasonal\nOranges,83,3.2,10,Seasonal\nWatermelons,249,4.4,24,Everyday\n\ncapacity.csv\nrefrigeration_service_visits,Capacity\n8,875'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'supplier_quality_rating': '3.2', 'Value': '64', 'catalog_display_group': 'Featured'}}, {'source': 'products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'supplier_quality_rating': '4.1', 'Value': '75', 'catalog_display_group': 'Featured'}}, {'source': 'products.csv', 'values': {'ProductName': 'Apples', 'Weight': '773', 'supplier_quality_rating': '4.7', 'Value': '68', 'catalog_display_group': 'Seasonal'}}, {'source': 'products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'supplier_quality_rating': '4.7', 'Value': '11', 'catalog_display_group': 'Featured'}}, {'source': 'products.csv', 'values': {'ProductName': 'Basil', 'Weight': '755', 'supplier_quality_rating': '4.1', 'Value': '91', 'catalog_display_group': 'Seasonal'}}, {'source': 'products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'supplier_quality_rating': '3.5', 'Value': '31', 'catalog_display_group': 'Everyday'}}, {'source': 'products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'supplier_quality_rating': '4.1', 'Value': '90', 'catalog_display_group': 'Featured'}}, {'source': 'products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'supplier_quality_rating': '3.5', 'Value': '56', 'catalog_display_group': 'Seasonal'}}, {'source': 'products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'supplier_quality_rating': '3.2', 'Value': '10', 'catalog_display_group': 'Seasonal'}}, {'source': 'products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'supplier_quality_rating': '4.4', 'Value': '24', 'catalog_display_group': 'Everyday'}}, {'source': 'capacity.csv', 'values': {'refrigeration_service_visits': '8', 'Capacity': '875'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
weights = {}
values = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        pname = rec['values']['ProductName']
        products.append(pname)
        weights[pname] = int(rec['values']['Weight'])
        values[pname] = int(rec['values']['Value'])
capacity = None
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        if 'Capacity' in rec['values']:
            capacity = int(rec['values']['Capacity'])
if capacity is None:
    raise ValueError('Missing capacity value in LEGACY_RECORDS.')
if len(products) != len(weights) or len(products) != len(values):
    raise ValueError('Mismatch in product, weight, or value data.')
m = gp.Model('Supermarket_Stock_Optimization')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x[p] for p in products)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')