LEGACY_OBSERVATION = 'capacity.csv\nprevious_period_capacity,Capacity\n991,875\n\nproducts.csv\nProductName,Weight,previous_period_unit_value,Value,previous_period_stock_status\nSpinach,230,58,64,Balanced\nShiitake Mushrooms,637,81,75,Balanced\nApples,773,79,68,Stockout\nCarrots,653,9,11,Stockout\nBasil,755,77,91,Overstock\nPotatoes,670,26,31,Stockout\nGreen Beans,505,87,90,Balanced\nBlueberries,821,66,56,Stockout\nOranges,83,12,10,Balanced\nWatermelons,249,22,24,Overstock'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'previous_period_capacity': '991', 'Capacity': '875'}}, {'source': 'products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'previous_period_unit_value': '58', 'Value': '64', 'previous_period_stock_status': 'Balanced'}}, {'source': 'products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'previous_period_unit_value': '81', 'Value': '75', 'previous_period_stock_status': 'Balanced'}}, {'source': 'products.csv', 'values': {'ProductName': 'Apples', 'Weight': '773', 'previous_period_unit_value': '79', 'Value': '68', 'previous_period_stock_status': 'Stockout'}}, {'source': 'products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'previous_period_unit_value': '9', 'Value': '11', 'previous_period_stock_status': 'Stockout'}}, {'source': 'products.csv', 'values': {'ProductName': 'Basil', 'Weight': '755', 'previous_period_unit_value': '77', 'Value': '91', 'previous_period_stock_status': 'Overstock'}}, {'source': 'products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'previous_period_unit_value': '26', 'Value': '31', 'previous_period_stock_status': 'Stockout'}}, {'source': 'products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'previous_period_unit_value': '87', 'Value': '90', 'previous_period_stock_status': 'Balanced'}}, {'source': 'products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'previous_period_unit_value': '66', 'Value': '56', 'previous_period_stock_status': 'Stockout'}}, {'source': 'products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'previous_period_unit_value': '12', 'Value': '10', 'previous_period_stock_status': 'Balanced'}}, {'source': 'products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'previous_period_unit_value': '22', 'Value': '24', 'previous_period_stock_status': 'Overstock'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
weights = {}
values = {}
capacity = None
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        pname = rec['values']['ProductName']
        products.append(pname)
        weights[pname] = int(rec['values']['Weight'])
        values[pname] = int(rec['values']['Value'])
    elif rec['source'] == 'capacity.csv':
        if 'Capacity' in rec['values']:
            capacity = int(rec['values']['Capacity'])
if capacity is None:
    raise ValueError('Missing capacity value in LEGACY_RECORDS')
if set(products) != set(weights.keys()) or set(products) != set(values.keys()):
    raise ValueError('Missing product data in LEGACY_RECORDS')
m = gp.Model('Supermarket_Stock')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x[i] for i in products)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')