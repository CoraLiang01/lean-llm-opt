LEGACY_OBSERVATION = '{"values": {"previous_period_capacity": "205", "resource_capacity": "180"}}\n{"values": {"previous_period_stock_status": "Stockout", "item_name": "Baguette", "previous_period_unit_value": "1044", "item_value": "888", "resource_requirement": "4"}}\n{"values": {"previous_period_stock_status": "Overstock", "item_name": "Croissant", "previous_period_unit_value": "142", "item_value": "134", "resource_requirement": "2"}}\n{"values": {"previous_period_stock_status": "Balanced", "item_name": "Sourdough", "previous_period_unit_value": "141", "item_value": "129", "resource_requirement": "4"}}\n{"values": {"previous_period_stock_status": "Overstock", "item_name": "Rye Bread", "previous_period_unit_value": "311", "item_value": "370", "resource_requirement": "3"}}\n{"values": {"previous_period_stock_status": "Overstock", "item_name": "Focaccia", "previous_period_unit_value": "770", "item_value": "765", "resource_requirement": "1"}}\n{"values": {"previous_period_stock_status": "Overstock", "item_name": "Ciabatta", "previous_period_unit_value": "129", "item_value": "154", "resource_requirement": "2"}}\n{"values": {"previous_period_stock_status": "Balanced", "item_name": "Pita", "previous_period_unit_value": "914", "item_value": "837", "resource_requirement": "1"}}\n{"values": {"previous_period_stock_status": "Stockout", "item_name": "Bagel", "previous_period_unit_value": "668", "item_value": "584", "resource_requirement": "3"}}\n{"values": {"previous_period_stock_status": "Stockout", "item_name": "English Muffin", "previous_period_unit_value": "314", "item_value": "365", "resource_requirement": "3"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'previous_period_capacity': '205', 'resource_capacity': '180'}}, {'source': '', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'Baguette', 'previous_period_unit_value': '1044', 'item_value': '888', 'resource_requirement': '4'}}, {'source': '', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Croissant', 'previous_period_unit_value': '142', 'item_value': '134', 'resource_requirement': '2'}}, {'source': '', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': 'Sourdough', 'previous_period_unit_value': '141', 'item_value': '129', 'resource_requirement': '4'}}, {'source': '', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Rye Bread', 'previous_period_unit_value': '311', 'item_value': '370', 'resource_requirement': '3'}}, {'source': '', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Focaccia', 'previous_period_unit_value': '770', 'item_value': '765', 'resource_requirement': '1'}}, {'source': '', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Ciabatta', 'previous_period_unit_value': '129', 'item_value': '154', 'resource_requirement': '2'}}, {'source': '', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': 'Pita', 'previous_period_unit_value': '914', 'item_value': '837', 'resource_requirement': '1'}}, {'source': '', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'Bagel', 'previous_period_unit_value': '668', 'item_value': '584', 'resource_requirement': '3'}}, {'source': '', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'English Muffin', 'previous_period_unit_value': '314', 'item_value': '365', 'resource_requirement': '3'}}]
import gurobipy as gp
from gurobipy import GRB
resource_capacity = None
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'resource_capacity' in vals:
        resource_capacity = int(vals['resource_capacity'])
        break
if resource_capacity is None:
    raise ValueError('Missing resource_capacity in LEGACY_RECORDS')
items = []
profit = {}
space = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'item_name' in vals and 'item_value' in vals and ('resource_requirement' in vals):
        name = vals['item_name']
        items.append(name)
        profit[name] = int(vals['item_value'])
        space[name] = int(vals['resource_requirement'])
if not items or len(items) != 9:
    raise ValueError('Expected 9 bread types in LEGACY_RECORDS')
for name in items:
    if name not in profit or name not in space:
        raise ValueError(f'Missing profit or space for {name}')
m = gp.Model('Bakery_Bread_Order')
x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[i] * x[i] for i in items)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((space[i] * x[i] for i in items)) <= resource_capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')