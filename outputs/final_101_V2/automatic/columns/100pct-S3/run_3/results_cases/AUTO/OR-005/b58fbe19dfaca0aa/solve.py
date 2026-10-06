LEGACY_OBSERVATION = 'products.csv\nprevious_period_stock_status,item_name,previous_period_unit_value,item_value,previous_period_resource_requirement,resource_requirement\nStockout,Baguette,1044,888,5,4\nOverstock,Croissant,142,134,3,2\nBalanced,Sourdough,141,129,5,4\nOverstock,Rye Bread,311,370,4,3\nStockout,Brioche,753,921,3,2\nOverstock,Focaccia,770,765,2,1\nOverstock,Ciabatta,129,154,3,2\nBalanced,Pita,914,837,2,1\nStockout,Bagel,668,584,2,3\nStockout,English Muffin,314,365,4,3\n\ncapacity.csv\nprevious_period_capacity,resource_capacity\n205,180'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'Baguette', 'previous_period_unit_value': '1044', 'item_value': '888', 'previous_period_resource_requirement': '5', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Croissant', 'previous_period_unit_value': '142', 'item_value': '134', 'previous_period_resource_requirement': '3', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': 'Sourdough', 'previous_period_unit_value': '141', 'item_value': '129', 'previous_period_resource_requirement': '5', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Rye Bread', 'previous_period_unit_value': '311', 'item_value': '370', 'previous_period_resource_requirement': '4', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'Brioche', 'previous_period_unit_value': '753', 'item_value': '921', 'previous_period_resource_requirement': '3', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Focaccia', 'previous_period_unit_value': '770', 'item_value': '765', 'previous_period_resource_requirement': '2', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Ciabatta', 'previous_period_unit_value': '129', 'item_value': '154', 'previous_period_resource_requirement': '3', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': 'Pita', 'previous_period_unit_value': '914', 'item_value': '837', 'previous_period_resource_requirement': '2', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'Bagel', 'previous_period_unit_value': '668', 'item_value': '584', 'previous_period_resource_requirement': '2', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'English Muffin', 'previous_period_unit_value': '314', 'item_value': '365', 'previous_period_resource_requirement': '4', 'resource_requirement': '3'}}, {'source': 'capacity.csv', 'values': {'previous_period_capacity': '205', 'resource_capacity': '180'}}]
import gurobipy as gp
from gurobipy import GRB
products = [rec['values'] for rec in LEGACY_RECORDS if rec['source'] == 'products.csv']
capacity_recs = [rec['values'] for rec in LEGACY_RECORDS if rec['source'] == 'capacity.csv']
items = [p['item_name'] for p in products]
item_value = {}
resource_requirement = {}
for p in products:
    try:
        item_value[p['item_name']] = int(p['item_value'])
        resource_requirement[p['item_name']] = int(p['resource_requirement'])
    except Exception as e:
        raise ValueError(f"Missing or invalid data for item {p['item_name']}: {e}")
resource_capacities = []
for c in capacity_recs:
    try:
        resource_capacities.append(int(c['resource_capacity']))
    except Exception as e:
        raise ValueError(f'Missing or invalid resource_capacity: {e}')
if not resource_capacities:
    raise ValueError('No resource capacities found in LEGACY_RECORDS')
resource_capacity = sum(resource_capacities)
for i in items:
    if i not in item_value or i not in resource_requirement:
        raise ValueError(f'Missing coefficients for item {i}')
m = gp.Model('Bakery_Order')
x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((item_value[i] * x[i] for i in items)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((resource_requirement[i] * x[i] for i in items)) <= resource_capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')