LEGACY_OBSERVATION = 'capacity.csv\nprevious_period_capacity,Capacity\n991,875\n\nproducts.csv\nProductName,Weight,previous_period_resource_requirement,previous_period_unit_value,Value,previous_period_stock_status\nSpinach,230,261,58,64,Balanced\nShiitake Mushrooms,637,523,81,75,Balanced\nApples,773,895,79,68,Stockout\nCarrots,653,659,9,11,Stockout\nBasil,755,730,77,91,Overstock\nPotatoes,670,709,26,31,Stockout\nGreen Beans,505,498,87,90,Balanced\nBlueberries,821,736,66,56,Stockout\nOranges,83,91,12,10,Balanced\nWatermelons,249,209,22,24,Overstock'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'previous_period_capacity': '991', 'Capacity': '875'}}, {'source': 'products.csv', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'previous_period_resource_requirement': '261', 'previous_period_unit_value': '58', 'Value': '64', 'previous_period_stock_status': 'Balanced'}}, {'source': 'products.csv', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'previous_period_resource_requirement': '523', 'previous_period_unit_value': '81', 'Value': '75', 'previous_period_stock_status': 'Balanced'}}, {'source': 'products.csv', 'values': {'ProductName': 'Apples', 'Weight': '773', 'previous_period_resource_requirement': '895', 'previous_period_unit_value': '79', 'Value': '68', 'previous_period_stock_status': 'Stockout'}}, {'source': 'products.csv', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'previous_period_resource_requirement': '659', 'previous_period_unit_value': '9', 'Value': '11', 'previous_period_stock_status': 'Stockout'}}, {'source': 'products.csv', 'values': {'ProductName': 'Basil', 'Weight': '755', 'previous_period_resource_requirement': '730', 'previous_period_unit_value': '77', 'Value': '91', 'previous_period_stock_status': 'Overstock'}}, {'source': 'products.csv', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'previous_period_resource_requirement': '709', 'previous_period_unit_value': '26', 'Value': '31', 'previous_period_stock_status': 'Stockout'}}, {'source': 'products.csv', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'previous_period_resource_requirement': '498', 'previous_period_unit_value': '87', 'Value': '90', 'previous_period_stock_status': 'Balanced'}}, {'source': 'products.csv', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'previous_period_resource_requirement': '736', 'previous_period_unit_value': '66', 'Value': '56', 'previous_period_stock_status': 'Stockout'}}, {'source': 'products.csv', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'previous_period_resource_requirement': '91', 'previous_period_unit_value': '12', 'Value': '10', 'previous_period_stock_status': 'Balanced'}}, {'source': 'products.csv', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'previous_period_resource_requirement': '209', 'previous_period_unit_value': '22', 'Value': '24', 'previous_period_stock_status': 'Overstock'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
value = {}
weight = {}
capacity = None
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        pname = rec['values']['ProductName']
        products.append(pname)
        value[pname] = int(rec['values']['Value'])
        weight[pname] = int(rec['values']['Weight'])
    elif rec['source'] == 'capacity.csv':
        if 'Capacity' in rec['values']:
            capacity = int(rec['values']['Capacity'])
if set(value) != set(products) or set(weight) != set(products):
    raise ValueError('Missing value or weight data for some products.')
if capacity is None:
    raise ValueError('Missing capacity data.')
m = gp.Model('Supermarket_Stock')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in products)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')