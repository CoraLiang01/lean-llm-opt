LEGACY_OBSERVATION = '{"values": {"archive_revision_number": "3", "resource_capacity": "180"}}\n\n{"values": {"record_keeper_group": "Team C", "item_name": "Baguette", "archive_revision_number": "5", "item_value": "888", "archived_attachment_count": "4", "resource_requirement": "4"}}\n\n{"values": {"record_keeper_group": "Team C", "item_name": "Croissant", "archive_revision_number": "6", "item_value": "134", "archived_attachment_count": "6", "resource_requirement": "2"}}\n\n{"values": {"record_keeper_group": "Team A", "item_name": "Sourdough", "archive_revision_number": "5", "item_value": "129", "archived_attachment_count": "1", "resource_requirement": "4"}}\n\n{"values": {"record_keeper_group": "Team C", "item_name": "Rye Bread", "archive_revision_number": "1", "item_value": "370", "archived_attachment_count": "1", "resource_requirement": "3"}}\n\n{"values": {"record_keeper_group": "Team A", "item_name": "Brioche", "archive_revision_number": "8", "item_value": "921", "archived_attachment_count": "2", "resource_requirement": "2"}}\n\n{"values": {"record_keeper_group": "Team C", "item_name": "Focaccia", "archive_revision_number": "9", "item_value": "765", "archived_attachment_count": "6", "resource_requirement": "1"}}\n\n{"values": {"record_keeper_group": "Team B", "item_name": "Ciabatta", "archive_revision_number": "1", "item_value": "154", "archived_attachment_count": "3", "resource_requirement": "2"}}\n\n{"values": {"record_keeper_group": "Team B", "item_name": "Pita", "archive_revision_number": "2", "item_value": "837", "archived_attachment_count": "6", "resource_requirement": "1"}}\n\n{"values": {"record_keeper_group": "Team C", "item_name": "Bagel", "archive_revision_number": "2", "item_value": "584", "archived_attachment_count": "1", "resource_requirement": "3"}}\n\n{"values": {"record_keeper_group": "Team A", "item_name": "English Muffin", "archive_revision_number": "1", "item_value": "365", "archived_attachment_count": "6", "resource_requirement": "3"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'archive_revision_number': '3', 'resource_capacity': '180'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Baguette', 'archive_revision_number': '5', 'item_value': '888', 'archived_attachment_count': '4', 'resource_requirement': '4'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Croissant', 'archive_revision_number': '6', 'item_value': '134', 'archived_attachment_count': '6', 'resource_requirement': '2'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'item_name': 'Sourdough', 'archive_revision_number': '5', 'item_value': '129', 'archived_attachment_count': '1', 'resource_requirement': '4'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Rye Bread', 'archive_revision_number': '1', 'item_value': '370', 'archived_attachment_count': '1', 'resource_requirement': '3'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'item_name': 'Brioche', 'archive_revision_number': '8', 'item_value': '921', 'archived_attachment_count': '2', 'resource_requirement': '2'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Focaccia', 'archive_revision_number': '9', 'item_value': '765', 'archived_attachment_count': '6', 'resource_requirement': '1'}}, {'source': '', 'values': {'record_keeper_group': 'Team B', 'item_name': 'Ciabatta', 'archive_revision_number': '1', 'item_value': '154', 'archived_attachment_count': '3', 'resource_requirement': '2'}}, {'source': '', 'values': {'record_keeper_group': 'Team B', 'item_name': 'Pita', 'archive_revision_number': '2', 'item_value': '837', 'archived_attachment_count': '6', 'resource_requirement': '1'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Bagel', 'archive_revision_number': '2', 'item_value': '584', 'archived_attachment_count': '1', 'resource_requirement': '3'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'item_name': 'English Muffin', 'archive_revision_number': '1', 'item_value': '365', 'archived_attachment_count': '6', 'resource_requirement': '3'}}]
import gurobipy as gp
from gurobipy import GRB
resource_capacity = None
for rec in LEGACY_RECORDS:
    vals = rec.get('values', {})
    if 'resource_capacity' in vals:
        resource_capacity = int(vals['resource_capacity'])
        break
if resource_capacity is None:
    raise ValueError('Missing resource_capacity in LEGACY_RECORDS')
items = []
profit = {}
space = {}
for rec in LEGACY_RECORDS:
    vals = rec.get('values', {})
    if 'item_name' in vals and 'item_value' in vals and ('resource_requirement' in vals):
        name = vals['item_name']
        items.append(name)
        profit[name] = int(vals['item_value'])
        space[name] = int(vals['resource_requirement'])
for name in items:
    if name not in profit or name not in space:
        raise ValueError(f'Missing data for item {name}')
m = gp.Model('BakeryBreadOrder')
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