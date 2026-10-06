LEGACY_OBSERVATION = 'products.csv\nsupplier_service_tier,item_name,ingredient_supplier_rating,item_value,resource_requirement\nStandard,Baguette,4.7,888,4\nPremium,Croissant,3.2,134,2\nStandard,Sourdough,3.2,129,4\nPriority,Rye Bread,3.5,370,3\nStandard,Brioche,4.7,921,2\nPremium,Focaccia,3.5,765,1\nPriority,Ciabatta,3.2,154,2\nPremium,Pita,3.8,837,1\nPremium,Bagel,4.7,584,3\nStandard,English Muffin,4.4,365,3\n\ncapacity.csv\ndaily_cleaning_cost,resource_capacity\n204.6,180'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'Baguette', 'ingredient_supplier_rating': '4.7', 'item_value': '888', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Croissant', 'ingredient_supplier_rating': '3.2', 'item_value': '134', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'Sourdough', 'ingredient_supplier_rating': '3.2', 'item_value': '129', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Priority', 'item_name': 'Rye Bread', 'ingredient_supplier_rating': '3.5', 'item_value': '370', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'Brioche', 'ingredient_supplier_rating': '4.7', 'item_value': '921', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Focaccia', 'ingredient_supplier_rating': '3.5', 'item_value': '765', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Priority', 'item_name': 'Ciabatta', 'ingredient_supplier_rating': '3.2', 'item_value': '154', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Pita', 'ingredient_supplier_rating': '3.8', 'item_value': '837', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Bagel', 'ingredient_supplier_rating': '4.7', 'item_value': '584', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'English Muffin', 'ingredient_supplier_rating': '4.4', 'item_value': '365', 'resource_requirement': '3'}}, {'source': 'capacity.csv', 'values': {'daily_cleaning_cost': '204.6', 'resource_capacity': '180'}}]
import gurobipy as gp
from gurobipy import GRB
products = [rec['values'] for rec in LEGACY_RECORDS if rec['source'] == 'products.csv']
capacity_recs = [rec['values'] for rec in LEGACY_RECORDS if rec['source'] == 'capacity.csv']
items = [p['item_name'] for p in products]
profit = {}
space = {}
for p in products:
    try:
        profit[p['item_name']] = int(p['item_value'])
        space[p['item_name']] = int(p['resource_requirement'])
    except Exception as e:
        raise ValueError(f"Missing or invalid data for item {p['item_name']}: {e}")
if not capacity_recs or 'resource_capacity' not in capacity_recs[0]:
    raise ValueError('Missing resource_capacity in capacity.csv')
capacity = int(capacity_recs[0]['resource_capacity'])
if set(profit.keys()) != set(items) or set(space.keys()) != set(items):
    raise ValueError('Profit or space data missing for some items.')
m = gp.Model('Bakery_Order')
x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[i] * x[i] for i in items)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((space[i] * x[i] for i in items)) <= capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')