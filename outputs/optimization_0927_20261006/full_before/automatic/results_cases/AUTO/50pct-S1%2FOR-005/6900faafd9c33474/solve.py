LEGACY_OBSERVATION = '{"values": {"archive_revision_number": "3", "resource_capacity": "180"}}\n{"values": {"record_keeper_group": "Team C", "item_name": "Baguette", "archive_revision_number": "5", "item_value": "888", "resource_requirement": "4"}}\n{"values": {"record_keeper_group": "Team C", "item_name": "Croissant", "archive_revision_number": "6", "item_value": "134", "resource_requirement": "2"}}\n{"values": {"record_keeper_group": "Team A", "item_name": "Sourdough", "archive_revision_number": "5", "item_value": "129", "resource_requirement": "4"}}\n{"values": {"record_keeper_group": "Team C", "item_name": "Rye Bread", "archive_revision_number": "1", "item_value": "370", "resource_requirement": "3"}}\n{"values": {"record_keeper_group": "Team A", "item_name": "Brioche", "archive_revision_number": "8", "item_value": "921", "resource_requirement": "2"}}\n{"values": {"record_keeper_group": "Team C", "item_name": "Focaccia", "archive_revision_number": "9", "item_value": "765", "resource_requirement": "1"}}\n{"values": {"record_keeper_group": "Team B", "item_name": "Ciabatta", "archive_revision_number": "1", "item_value": "154", "resource_requirement": "2"}}\n{"values": {"record_keeper_group": "Team B", "item_name": "Pita", "archive_revision_number": "2", "item_value": "837", "resource_requirement": "1"}}\n{"values": {"record_keeper_group": "Team C", "item_name": "Bagel", "archive_revision_number": "2", "item_value": "584", "resource_requirement": "3"}}\n{"values": {"record_keeper_group": "Team A", "item_name": "English Muffin", "archive_revision_number": "1", "item_value": "365", "resource_requirement": "3"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'archive_revision_number': '3', 'resource_capacity': '180'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Baguette', 'archive_revision_number': '5', 'item_value': '888', 'resource_requirement': '4'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Croissant', 'archive_revision_number': '6', 'item_value': '134', 'resource_requirement': '2'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'item_name': 'Sourdough', 'archive_revision_number': '5', 'item_value': '129', 'resource_requirement': '4'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Rye Bread', 'archive_revision_number': '1', 'item_value': '370', 'resource_requirement': '3'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'item_name': 'Brioche', 'archive_revision_number': '8', 'item_value': '921', 'resource_requirement': '2'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Focaccia', 'archive_revision_number': '9', 'item_value': '765', 'resource_requirement': '1'}}, {'source': '', 'values': {'record_keeper_group': 'Team B', 'item_name': 'Ciabatta', 'archive_revision_number': '1', 'item_value': '154', 'resource_requirement': '2'}}, {'source': '', 'values': {'record_keeper_group': 'Team B', 'item_name': 'Pita', 'archive_revision_number': '2', 'item_value': '837', 'resource_requirement': '1'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Bagel', 'archive_revision_number': '2', 'item_value': '584', 'resource_requirement': '3'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'item_name': 'English Muffin', 'archive_revision_number': '1', 'item_value': '365', 'resource_requirement': '3'}}]
import gurobipy as gp
from gurobipy import GRB
resource_capacity = None
items = []
item_value = {}
resource_requirement = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'resource_capacity' in vals:
        if resource_capacity is not None:
            raise ValueError('Multiple resource_capacity entries found.')
        resource_capacity = int(vals['resource_capacity'])
    if 'item_name' in vals:
        name = vals['item_name']
        items.append(name)
        if 'item_value' not in vals or 'resource_requirement' not in vals:
            raise ValueError(f'Missing data for item {name}')
        item_value[name] = int(vals['item_value'])
        resource_requirement[name] = int(vals['resource_requirement'])
if resource_capacity is None:
    raise ValueError('No resource_capacity found in LEGACY_RECORDS.')
if set(items) != set(item_value.keys()) or set(items) != set(resource_requirement.keys()):
    raise ValueError('Mismatch in item identifiers and coefficients.')
m = gp.Model('Bakery_Bread_Order')
x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((item_value[i] * x[i] for i in items)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((resource_requirement[i] * x[i] for i in items)) <= resource_capacity, name='storage_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')