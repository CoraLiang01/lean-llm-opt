LEGACY_OBSERVATION = 'products.csv\nprevious_period_stock_status,item_name,previous_period_unit_value,item_value,resource_requirement\nStockout,Baguette,1044,888,4\nOverstock,Croissant,142,134,2\nBalanced,Sourdough,141,129,4\nOverstock,Rye Bread,311,370,3\nStockout,Brioche,753,921,2\nOverstock,Focaccia,770,765,1\nOverstock,Ciabatta,129,154,2\nBalanced,Pita,914,837,1\nStockout,Bagel,668,584,3\nStockout,English Muffin,314,365,3\n\ncapacity.csv\nprevious_period_capacity,resource_capacity\n205,180'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'Baguette', 'previous_period_unit_value': '1044', 'item_value': '888', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Croissant', 'previous_period_unit_value': '142', 'item_value': '134', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': 'Sourdough', 'previous_period_unit_value': '141', 'item_value': '129', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Rye Bread', 'previous_period_unit_value': '311', 'item_value': '370', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'Brioche', 'previous_period_unit_value': '753', 'item_value': '921', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Focaccia', 'previous_period_unit_value': '770', 'item_value': '765', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Ciabatta', 'previous_period_unit_value': '129', 'item_value': '154', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': 'Pita', 'previous_period_unit_value': '914', 'item_value': '837', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'Bagel', 'previous_period_unit_value': '668', 'item_value': '584', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'English Muffin', 'previous_period_unit_value': '314', 'item_value': '365', 'resource_requirement': '3'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '205', 'resource_capacity': '180'}}]
import gurobipy as gp
from gurobipy import GRB
products = [rec['values'] for rec in LEGACY_RECORDS if rec['source'] == 'products.csv']
capacities = [rec['values'] for rec in LEGACY_RECORDS if rec['source'] == 'capacity.csv']
items = [p['item_name'] for p in products]
item_value = {}
resource_requirement = {}
for p in products:
    try:
        item_value[p['item_name']] = int(p['item_value'])
        resource_requirement[p['item_name']] = int(p['resource_requirement'])
    except Exception as e:
        raise ValueError(f"Missing or invalid data for item {p['item_name']}: {e}")
if not capacities:
    raise ValueError('No capacity data found.')
try:
    resource_capacity = sum((int(c['resource_capacity']) for c in capacities if 'resource_capacity' in c))
except Exception as e:
    raise ValueError(f'Missing or invalid resource_capacity: {e}')
for i in items:
    if i not in item_value or i not in resource_requirement:
        raise ValueError(f'Missing coefficients for item {i}')
m = gp.Model('Bakery_Bread_Stock')
x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((item_value[i] * x[i] for i in items)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((resource_requirement[i] * x[i] for i in items)) <= resource_capacity, name='storage_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')