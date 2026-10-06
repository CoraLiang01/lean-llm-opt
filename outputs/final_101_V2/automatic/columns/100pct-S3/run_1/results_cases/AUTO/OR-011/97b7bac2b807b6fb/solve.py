LEGACY_OBSERVATION = 'capacity.csv\nprevious_period_capacity,Capacity\n991,875\n\nproducts.csv\nProductName,Weight,previous_period_resource_requirement,previous_period_unit_value,Value,previous_period_stock_status\nSpinach,230,261,58,64,Balanced\nShiitake Mushrooms,637,523,81,75,Balanced\nApples,773,895,79,68,Stockout\nCarrots,653,659,9,11,Stockout\nBasil,755,730,77,91,Overstock\nPotatoes,670,709,26,31,Stockout\nGreen Beans,505,498,87,90,Balanced\nBlueberries,821,736,66,56,Stockout\nOranges,83,91,12,10,Balanced\nWatermelons,249,209,22,24,Overstock'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'previous_period_capacity': '991', 'Capacity': '875'}}, {'source': 'products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'previous_period_resource_requirement': '261', 'previous_period_unit_value': '58', 'Value': '64', 'previous_period_stock_status': 'Balanced'}}, {'source': 'products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'previous_period_resource_requirement': '523', 'previous_period_unit_value': '81', 'Value': '75', 'previous_period_stock_status': 'Balanced'}}, {'source': 'products.csv', 'values': {'ProductName': 'Apples', 'Weight': '773', 'previous_period_resource_requirement': '895', 'previous_period_unit_value': '79', 'Value': '68', 'previous_period_stock_status': 'Stockout'}}, {'source': 'products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'previous_period_resource_requirement': '659', 'previous_period_unit_value': '9', 'Value': '11', 'previous_period_stock_status': 'Stockout'}}, {'source': 'products.csv', 'values': {'ProductName': 'Basil', 'Weight': '755', 'previous_period_resource_requirement': '730', 'previous_period_unit_value': '77', 'Value': '91', 'previous_period_stock_status': 'Overstock'}}, {'source': 'products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'previous_period_resource_requirement': '709', 'previous_period_unit_value': '26', 'Value': '31', 'previous_period_stock_status': 'Stockout'}}, {'source': 'products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'previous_period_resource_requirement': '498', 'previous_period_unit_value': '87', 'Value': '90', 'previous_period_stock_status': 'Balanced'}}, {'source': 'products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'previous_period_resource_requirement': '736', 'previous_period_unit_value': '66', 'Value': '56', 'previous_period_stock_status': 'Stockout'}}, {'source': 'products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'previous_period_resource_requirement': '91', 'previous_period_unit_value': '12', 'Value': '10', 'previous_period_stock_status': 'Balanced'}}, {'source': 'products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'previous_period_resource_requirement': '209', 'previous_period_unit_value': '22', 'Value': '24', 'previous_period_stock_status': 'Overstock'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
values = {}
weights = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        pname = rec['values']['ProductName']
        products.append(pname)
        values[pname] = int(rec['values']['Value'])
        weights[pname] = int(rec['values']['Weight'])
capacity = None
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        capacity = int(rec['values']['Capacity'])
if capacity is None:
    raise ValueError('Missing capacity data')
if len(products) != len(values) or len(products) != len(weights):
    raise ValueError('Missing product data')
m = gp.Model('Supermarket_Stock')
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