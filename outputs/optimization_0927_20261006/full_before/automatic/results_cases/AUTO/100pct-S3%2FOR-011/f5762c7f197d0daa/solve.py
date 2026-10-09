LEGACY_OBSERVATION = '{"values": {"previous_period_capacity": "991", "Capacity": "875"}}\n{"values": {"ProductName": "Spinach", "Weight": "230", "previous_period_resource_requirement": "261", "previous_period_unit_value": "58", "Value": "64", "previous_period_stock_status": "Balanced"}}\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "637", "previous_period_resource_requirement": "523", "previous_period_unit_value": "81", "Value": "75", "previous_period_stock_status": "Balanced"}}\n{"values": {"ProductName": "Apples", "Weight": "773", "previous_period_resource_requirement": "895", "previous_period_unit_value": "79", "Value": "68", "previous_period_stock_status": "Stockout"}}\n{"values": {"ProductName": "Carrots", "Weight": "653", "previous_period_resource_requirement": "659", "previous_period_unit_value": "9", "Value": "11", "previous_period_stock_status": "Stockout"}}\n{"values": {"ProductName": "Basil", "Weight": "755", "previous_period_resource_requirement": "730", "previous_period_unit_value": "77", "Value": "91", "previous_period_stock_status": "Overstock"}}\n{"values": {"ProductName": "Potatoes", "Weight": "670", "previous_period_resource_requirement": "709", "previous_period_unit_value": "26", "Value": "31", "previous_period_stock_status": "Stockout"}}\n{"values": {"ProductName": "Green Beans", "Weight": "505", "previous_period_resource_requirement": "498", "previous_period_unit_value": "87", "Value": "90", "previous_period_stock_status": "Balanced"}}\n{"values": {"ProductName": "Blueberries", "Weight": "821", "previous_period_resource_requirement": "736", "previous_period_unit_value": "66", "Value": "56", "previous_period_stock_status": "Stockout"}}\n{"values": {"ProductName": "Oranges", "Weight": "83", "previous_period_resource_requirement": "91", "previous_period_unit_value": "12", "Value": "10", "previous_period_stock_status": "Balanced"}}\n{"values": {"ProductName": "Watermelons", "Weight": "249", "previous_period_resource_requirement": "209", "previous_period_unit_value": "22", "Value": "24", "previous_period_stock_status": "Overstock"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'previous_period_capacity': '991', 'Capacity': '875'}}, {'source': '', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'previous_period_resource_requirement': '261', 'previous_period_unit_value': '58', 'Value': '64', 'previous_period_stock_status': 'Balanced'}}, {'source': '', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'previous_period_resource_requirement': '523', 'previous_period_unit_value': '81', 'Value': '75', 'previous_period_stock_status': 'Balanced'}}, {'source': '', 'values': {'ProductName': 'Apples', 'Weight': '773', 'previous_period_resource_requirement': '895', 'previous_period_unit_value': '79', 'Value': '68', 'previous_period_stock_status': 'Stockout'}}, {'source': '', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'previous_period_resource_requirement': '659', 'previous_period_unit_value': '9', 'Value': '11', 'previous_period_stock_status': 'Stockout'}}, {'source': '', 'values': {'ProductName': 'Basil', 'Weight': '755', 'previous_period_resource_requirement': '730', 'previous_period_unit_value': '77', 'Value': '91', 'previous_period_stock_status': 'Overstock'}}, {'source': '', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'previous_period_resource_requirement': '709', 'previous_period_unit_value': '26', 'Value': '31', 'previous_period_stock_status': 'Stockout'}}, {'source': '', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'previous_period_resource_requirement': '498', 'previous_period_unit_value': '87', 'Value': '90', 'previous_period_stock_status': 'Balanced'}}, {'source': '', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'previous_period_resource_requirement': '736', 'previous_period_unit_value': '66', 'Value': '56', 'previous_period_stock_status': 'Stockout'}}, {'source': '', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'previous_period_resource_requirement': '91', 'previous_period_unit_value': '12', 'Value': '10', 'previous_period_stock_status': 'Balanced'}}, {'source': '', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'previous_period_resource_requirement': '209', 'previous_period_unit_value': '22', 'Value': '24', 'previous_period_stock_status': 'Overstock'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
value = {}
weight = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'ProductName' in vals:
        pname = vals['ProductName']
        products.append(pname)
        value[pname] = int(vals['Value'])
        weight[pname] = int(vals['Weight'])
capacity = None
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'Capacity' in vals:
        capacity = int(vals['Capacity'])
if capacity is None:
    raise ValueError('Missing total capacity in LEGACY_RECORDS.')
for pname in products:
    if pname not in value or pname not in weight:
        raise ValueError(f'Missing value or weight for product {pname}.')
m = gp.Model('SupermarketStock')
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