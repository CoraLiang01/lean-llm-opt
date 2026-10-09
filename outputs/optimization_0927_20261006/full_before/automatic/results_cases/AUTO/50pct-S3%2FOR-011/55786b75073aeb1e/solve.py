LEGACY_OBSERVATION = '{"values": {"previous_period_capacity": "991", "Capacity": "875"}}\n\n{"values": {"ProductName": "Spinach", "Weight": "230", "previous_period_unit_value": "58", "Value": "64", "previous_period_stock_status": "Balanced"}}\n\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "637", "previous_period_unit_value": "81", "Value": "75", "previous_period_stock_status": "Balanced"}}\n\n{"values": {"ProductName": "Apples", "Weight": "773", "previous_period_unit_value": "79", "Value": "68", "previous_period_stock_status": "Stockout"}}\n\n{"values": {"ProductName": "Carrots", "Weight": "653", "previous_period_unit_value": "9", "Value": "11", "previous_period_stock_status": "Stockout"}}\n\n{"values": {"ProductName": "Basil", "Weight": "755", "previous_period_unit_value": "77", "Value": "91", "previous_period_stock_status": "Overstock"}}\n\n{"values": {"ProductName": "Potatoes", "Weight": "670", "previous_period_unit_value": "26", "Value": "31", "previous_period_stock_status": "Stockout"}}\n\n{"values": {"ProductName": "Green Beans", "Weight": "505", "previous_period_unit_value": "87", "Value": "90", "previous_period_stock_status": "Balanced"}}\n\n{"values": {"ProductName": "Blueberries", "Weight": "821", "previous_period_unit_value": "66", "Value": "56", "previous_period_stock_status": "Stockout"}}\n\n{"values": {"ProductName": "Oranges", "Weight": "83", "previous_period_unit_value": "12", "Value": "10", "previous_period_stock_status": "Balanced"}}\n\n{"values": {"ProductName": "Watermelons", "Weight": "249", "previous_period_unit_value": "22", "Value": "24", "previous_period_stock_status": "Overstock"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'previous_period_capacity': '991', 'Capacity': '875'}}, {'source': '', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'previous_period_unit_value': '58', 'Value': '64', 'previous_period_stock_status': 'Balanced'}}, {'source': '', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'previous_period_unit_value': '81', 'Value': '75', 'previous_period_stock_status': 'Balanced'}}, {'source': '', 'values': {'ProductName': 'Apples', 'Weight': '773', 'previous_period_unit_value': '79', 'Value': '68', 'previous_period_stock_status': 'Stockout'}}, {'source': '', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'previous_period_unit_value': '9', 'Value': '11', 'previous_period_stock_status': 'Stockout'}}, {'source': '', 'values': {'ProductName': 'Basil', 'Weight': '755', 'previous_period_unit_value': '77', 'Value': '91', 'previous_period_stock_status': 'Overstock'}}, {'source': '', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'previous_period_unit_value': '26', 'Value': '31', 'previous_period_stock_status': 'Stockout'}}, {'source': '', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'previous_period_unit_value': '87', 'Value': '90', 'previous_period_stock_status': 'Balanced'}}, {'source': '', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'previous_period_unit_value': '66', 'Value': '56', 'previous_period_stock_status': 'Stockout'}}, {'source': '', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'previous_period_unit_value': '12', 'Value': '10', 'previous_period_stock_status': 'Balanced'}}, {'source': '', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'previous_period_unit_value': '22', 'Value': '24', 'previous_period_stock_status': 'Overstock'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
capacity = None
for rec in records:
    vals = rec['values']
    if 'Capacity' in vals and vals['Capacity']:
        capacity = int(vals['Capacity'])
        break
if capacity is None:
    raise ValueError('Missing overall capacity.')
products = []
weights = {}
values = {}
for rec in records:
    vals = rec['values']
    if 'ProductName' in vals and vals['ProductName']:
        pname = vals['ProductName']
        products.append(pname)
        if 'Weight' not in vals or 'Value' not in vals:
            raise ValueError(f'Missing data for product {pname}')
        weights[pname] = int(vals['Weight'])
        values[pname] = int(vals['Value'])
if len(products) == 0 or len(weights) != len(products) or len(values) != len(products):
    raise ValueError('Product data incomplete or inconsistent.')
m = gp.Model('Supermarket_Stock_Optimization')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[p] * x[p] for p in products)) <= capacity, name='stock_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')