LEGACY_OBSERVATION = '{"values": {"daily_cleaning_cost": "204.6", "resource_capacity": "180"}}\n\n{"values": {"supplier_service_tier": "Standard", "item_name": "Baguette", "ingredient_supplier_rating": "4.7", "item_value": "888", "resource_requirement": "4"}}\n\n{"values": {"supplier_service_tier": "Premium", "item_name": "Croissant", "ingredient_supplier_rating": "3.2", "item_value": "134", "resource_requirement": "2"}}\n\n{"values": {"supplier_service_tier": "Standard", "item_name": "Sourdough", "ingredient_supplier_rating": "3.2", "item_value": "129", "resource_requirement": "4"}}\n\n{"values": {"supplier_service_tier": "Priority", "item_name": "Rye Bread", "ingredient_supplier_rating": "3.5", "item_value": "370", "resource_requirement": "3"}}\n\n{"values": {"supplier_service_tier": "Standard", "item_name": "Brioche", "ingredient_supplier_rating": "4.7", "item_value": "921", "resource_requirement": "2"}}\n\n{"values": {"supplier_service_tier": "Premium", "item_name": "Focaccia", "ingredient_supplier_rating": "3.5", "item_value": "765", "resource_requirement": "1"}}\n\n{"values": {"supplier_service_tier": "Priority", "item_name": "Ciabatta", "ingredient_supplier_rating": "3.2", "item_value": "154", "resource_requirement": "2"}}\n\n{"values": {"supplier_service_tier": "Premium", "item_name": "Pita", "ingredient_supplier_rating": "3.8", "item_value": "837", "resource_requirement": "1"}}\n\n{"values": {"supplier_service_tier": "Premium", "item_name": "Bagel", "ingredient_supplier_rating": "4.7", "item_value": "584", "resource_requirement": "3"}}\n\n{"values": {"supplier_service_tier": "Standard", "item_name": "English Muffin", "ingredient_supplier_rating": "4.4", "item_value": "365", "resource_requirement": "3"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'daily_cleaning_cost': '204.6', 'resource_capacity': '180'}}, {'source': '', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'Baguette', 'ingredient_supplier_rating': '4.7', 'item_value': '888', 'resource_requirement': '4'}}, {'source': '', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Croissant', 'ingredient_supplier_rating': '3.2', 'item_value': '134', 'resource_requirement': '2'}}, {'source': '', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'Sourdough', 'ingredient_supplier_rating': '3.2', 'item_value': '129', 'resource_requirement': '4'}}, {'source': '', 'values': {'supplier_service_tier': 'Priority', 'item_name': 'Rye Bread', 'ingredient_supplier_rating': '3.5', 'item_value': '370', 'resource_requirement': '3'}}, {'source': '', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'Brioche', 'ingredient_supplier_rating': '4.7', 'item_value': '921', 'resource_requirement': '2'}}, {'source': '', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Focaccia', 'ingredient_supplier_rating': '3.5', 'item_value': '765', 'resource_requirement': '1'}}, {'source': '', 'values': {'supplier_service_tier': 'Priority', 'item_name': 'Ciabatta', 'ingredient_supplier_rating': '3.2', 'item_value': '154', 'resource_requirement': '2'}}, {'source': '', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Pita', 'ingredient_supplier_rating': '3.8', 'item_value': '837', 'resource_requirement': '1'}}, {'source': '', 'values': {'supplier_service_tier': 'Premium', 'item_name': 'Bagel', 'ingredient_supplier_rating': '4.7', 'item_value': '584', 'resource_requirement': '3'}}, {'source': '', 'values': {'supplier_service_tier': 'Standard', 'item_name': 'English Muffin', 'ingredient_supplier_rating': '4.4', 'item_value': '365', 'resource_requirement': '3'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
resource_capacity = None
for rec in records:
    vals = rec.get('values', {})
    if 'resource_capacity' in vals:
        resource_capacity = float(vals['resource_capacity'])
        break
if resource_capacity is None:
    raise ValueError('Missing resource_capacity in LEGACY_RECORDS')
bread_types = ['Baguette', 'Croissant', 'Sourdough', 'Rye Bread', 'Brioche', 'Focaccia', 'Ciabatta', 'Pita', 'Bagel', 'English Muffin']
item_value = {}
resource_requirement = {}
for bread in bread_types:
    found = False
    for rec in records:
        vals = rec.get('values', {})
        if vals.get('item_name', None) == bread:
            item_value[bread] = float(vals['item_value'])
            resource_requirement[bread] = float(vals['resource_requirement'])
            found = True
            break
    if not found:
        raise ValueError(f'Missing data for bread type: {bread}')
m = gp.Model('Bakery_Bread_Stocking')
x = m.addVars(bread_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((item_value[b] * x[b] for b in bread_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((resource_requirement[b] * x[b] for b in bread_types)) <= resource_capacity, name='storage_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')