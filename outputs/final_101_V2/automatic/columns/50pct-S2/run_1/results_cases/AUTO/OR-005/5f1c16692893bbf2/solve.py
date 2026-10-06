LEGACY_OBSERVATION = 'products.csv\nsupplier_service_tier,item_name,ingredient_supplier_rating,item_value,resource_requirement\nStandard,Baguette,4.7,888,4\nPremium,Croissant,3.2,134,2\nStandard,Sourdough,3.2,129,4\nPriority,Rye Bread,3.5,370,3\nStandard,Brioche,4.7,921,2\nPremium,Focaccia,3.5,765,1\nPriority,Ciabatta,3.2,154,2\nPremium,Pita,3.8,837,1\nPremium,Bagel,4.7,584,3\nStandard,English Muffin,4.4,365,3\n\ncapacity.csv\ndaily_cleaning_cost,resource_capacity\n204.6,180'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'Baguette', 'ingredient_supplier_rating': '4.7', 'item_value': '888', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Croissant', 'ingredient_supplier_rating': '3.2', 'item_value': '134', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'Sourdough', 'ingredient_supplier_rating': '3.2', 'item_value': '129', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Priority', 'item_name': 'Rye Bread', 'ingredient_supplier_rating': '3.5', 'item_value': '370', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'Brioche', 'ingredient_supplier_rating': '4.7', 'item_value': '921', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Focaccia', 'ingredient_supplier_rating': '3.5', 'item_value': '765', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Priority', 'item_name': 'Ciabatta', 'ingredient_supplier_rating': '3.2', 'item_value': '154', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Pita', 'ingredient_supplier_rating': '3.8', 'item_value': '837', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Bagel', 'ingredient_supplier_rating': '4.7', 'item_value': '584', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'English Muffin', 'ingredient_supplier_rating': '4.4', 'item_value': '365', 'resource_requirement': '3'}}, {'source': 'capacity.csv', 'values': {'daily_cleaning_cost': '204.6', 'resource_capacity': '180'}}]
import gurobipy as gp
from gurobipy import GRB
products = [rec for rec in LEGACY_RECORDS if rec['source'] == 'products.csv']
capacity_rec = next((rec for rec in LEGACY_RECORDS if rec['source'] == 'capacity.csv'))
item_names = []
item_value = {}
resource_requirement = {}
for prod in products:
    name = prod['values']['item_name']
    item_names.append(name)
    item_value[name] = int(prod['values']['item_value'])
    resource_requirement[name] = int(prod['values']['resource_requirement'])
resource_capacity = int(capacity_rec['values']['resource_capacity'])
if set(item_names) != set(item_value.keys()) or set(item_names) != set(resource_requirement.keys()):
    raise ValueError('Mismatch in item identifiers between data sources.')
m = gp.Model('Bakery_Stocking')
x = m.addVars(item_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((item_value[i] * x[i] for i in item_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((resource_requirement[i] * x[i] for i in item_names)) <= resource_capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')