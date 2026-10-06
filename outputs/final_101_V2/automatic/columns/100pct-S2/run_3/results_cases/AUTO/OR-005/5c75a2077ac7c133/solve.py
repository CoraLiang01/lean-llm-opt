LEGACY_OBSERVATION = 'capacity.csv\ndaily_cleaning_cost,resource_capacity\n204.6,180\n\nproducts.csv\nsupplier_service_tier,item_name,ingredient_supplier_rating,item_value,average_baking_minutes,resource_requirement\nStandard,Baguette,4.7,888,30,4\nPremium,Croissant,3.2,134,30,2\nStandard,Sourdough,3.2,129,45,4\nPriority,Rye Bread,3.5,370,30,3\nStandard,Brioche,4.7,921,12,2\nPremium,Focaccia,3.5,765,24,1\nPriority,Ciabatta,3.2,154,45,2\nPremium,Pita,3.8,837,18,1\nPremium,Bagel,4.7,584,30,3\nStandard,English Muffin,4.4,365,12,3'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'daily_cleaning_cost': '204.6', 'resource_capacity': '180'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'Baguette', 'ingredient_supplier_rating': '4.7', 'item_value': '888', 'average_baking_minutes': '30', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Croissant', 'ingredient_supplier_rating': '3.2', 'item_value': '134', 'average_baking_minutes': '30', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'Sourdough', 'ingredient_supplier_rating': '3.2', 'item_value': '129', 'average_baking_minutes': '45', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Priority', 'item_name': 'Rye Bread', 'ingredient_supplier_rating': '3.5', 'item_value': '370', 'average_baking_minutes': '30', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'Brioche', 'ingredient_supplier_rating': '4.7', 'item_value': '921', 'average_baking_minutes': '12', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Focaccia', 'ingredient_supplier_rating': '3.5', 'item_value': '765', 'average_baking_minutes': '24', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Priority', 'item_name': 'Ciabatta', 'ingredient_supplier_rating': '3.2', 'item_value': '154', 'average_baking_minutes': '45', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Pita', 'ingredient_supplier_rating': '3.8', 'item_value': '837', 'average_baking_minutes': '18', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Bagel', 'ingredient_supplier_rating': '4.7', 'item_value': '584', 'average_baking_minutes': '30', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'English Muffin', 'ingredient_supplier_rating': '4.4', 'item_value': '365', 'average_baking_minutes': '12', 'resource_requirement': '3'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
item_value = {}
resource_requirement = {}
item_names = []
resource_capacity = None
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        vals = rec['values']
        name = vals['item_name']
        item_names.append(name)
        products.append(name)
        try:
            item_value[name] = float(vals['item_value'])
        except Exception:
            raise ValueError(f'Missing or invalid item_value for {name}')
        try:
            resource_requirement[name] = float(vals['resource_requirement'])
        except Exception:
            raise ValueError(f'Missing or invalid resource_requirement for {name}')
    elif rec['source'] == 'capacity.csv':
        vals = rec['values']
        if 'resource_capacity' in vals:
            try:
                resource_capacity = float(vals['resource_capacity'])
            except Exception:
                raise ValueError('Missing or invalid resource_capacity in capacity.csv')
if resource_capacity is None:
    raise ValueError('No resource_capacity found in capacity.csv')
for name in item_names:
    if name not in item_value or name not in resource_requirement:
        raise ValueError(f'Missing data for product {name}')
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