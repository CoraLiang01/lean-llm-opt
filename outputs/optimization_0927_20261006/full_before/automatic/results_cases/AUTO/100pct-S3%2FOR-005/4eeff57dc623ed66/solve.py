LEGACY_OBSERVATION = 'capacity.csv\nprevious_period_capacity,resource_capacity\n205,180\n\nproducts.csv\nprevious_period_stock_status,item_name,previous_period_unit_value,item_value,previous_period_resource_requirement,resource_requirement\nStockout,Baguette,1044,888,5,4\nOverstock,Croissant,142,134,3,2\nBalanced,Sourdough,141,129,5,4\nOverstock,Rye Bread,311,370,4,3\nStockout,Brioche,753,921,3,2\nOverstock,Focaccia,770,765,2,1\nOverstock,Ciabatta,129,154,3,2\nBalanced,Pita,914,837,2,1\nStockout,Bagel,668,584,2,3\nStockout,English Muffin,314,365,4,3'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'previous_period_capacity': '205', 'resource_capacity': '180'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'Baguette', 'previous_period_unit_value': '1044', 'item_value': '888', 'previous_period_resource_requirement': '5', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Croissant', 'previous_period_unit_value': '142', 'item_value': '134', 'previous_period_resource_requirement': '3', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': 'Sourdough', 'previous_period_unit_value': '141', 'item_value': '129', 'previous_period_resource_requirement': '5', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Rye Bread', 'previous_period_unit_value': '311', 'item_value': '370', 'previous_period_resource_requirement': '4', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'Brioche', 'previous_period_unit_value': '753', 'item_value': '921', 'previous_period_resource_requirement': '3', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Focaccia', 'previous_period_unit_value': '770', 'item_value': '765', 'previous_period_resource_requirement': '2', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Overstock', 'item_name': 'Ciabatta', 'previous_period_unit_value': '129', 'item_value': '154', 'previous_period_resource_requirement': '3', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Balanced', 'item_name': 'Pita', 'previous_period_unit_value': '914', 'item_value': '837', 'previous_period_resource_requirement': '2', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'Bagel', 'previous_period_unit_value': '668', 'item_value': '584', 'previous_period_resource_requirement': '2', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'previous_period_stock_status': 'Stockout', 'item_name': 'English Muffin', 'previous_period_unit_value': '314', 'item_value': '365', 'previous_period_resource_requirement': '4', 'resource_requirement': '3'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
profits = {}
requirements = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        name = rec['values']['item_name']
        products.append(name)
        try:
            profits[name] = int(rec['values']['item_value'])
        except Exception:
            raise ValueError(f'Missing or invalid item_value for {name}')
        try:
            requirements[name] = int(rec['values']['resource_requirement'])
        except Exception:
            raise ValueError(f'Missing or invalid resource_requirement for {name}')
capacities = []
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        try:
            capacities.append(int(rec['values']['resource_capacity']))
        except Exception:
            raise ValueError('Missing or invalid resource_capacity in capacity.csv')
if not capacities:
    raise ValueError('No resource_capacity found in LEGACY_RECORDS')
total_capacity = sum(capacities)
if set(profits.keys()) != set(products) or set(requirements.keys()) != set(products):
    raise ValueError('Mismatch in products, profits, or requirements data')
m = gp.Model('Bakery_Order')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profits[i] * x[i] for i in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((requirements[i] * x[i] for i in products)) <= total_capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')