LEGACY_OBSERVATION = 'products.csv\nrecord_keeper_group,item_name,archive_revision_number,item_value,archived_attachment_count,resource_requirement\nTeam C,Baguette,5,888,4,4\nTeam C,Croissant,6,134,6,2\nTeam A,Sourdough,5,129,1,4\nTeam C,Rye Bread,1,370,1,3\nTeam A,Brioche,8,921,2,2\nTeam C,Focaccia,9,765,6,1\nTeam B,Ciabatta,1,154,3,2\nTeam B,Pita,2,837,6,1\nTeam C,Bagel,2,584,1,3\nTeam A,English Muffin,1,365,6,3\n\ncapacity.csv\narchive_revision_number,resource_capacity\n3,180'
LEGACY_RECORDS = [{'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Baguette', 'archive_revision_number': '5', 'item_value': '888', 'archived_attachment_count': '4', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Croissant', 'archive_revision_number': '6', 'item_value': '134', 'archived_attachment_count': '6', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'item_name': 'Sourdough', 'archive_revision_number': '5', 'item_value': '129', 'archived_attachment_count': '1', 'resource_requirement': '4'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Rye Bread', 'archive_revision_number': '1', 'item_value': '370', 'archived_attachment_count': '1', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'item_name': 'Brioche', 'archive_revision_number': '8', 'item_value': '921', 'archived_attachment_count': '2', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Focaccia', 'archive_revision_number': '9', 'item_value': '765', 'archived_attachment_count': '6', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'item_name': 'Ciabatta', 'archive_revision_number': '1', 'item_value': '154', 'archived_attachment_count': '3', 'resource_requirement': '2'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'item_name': 'Pita', 'archive_revision_number': '2', 'item_value': '837', 'archived_attachment_count': '6', 'resource_requirement': '1'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'item_name': 'Bagel', 'archive_revision_number': '2', 'item_value': '584', 'archived_attachment_count': '1', 'resource_requirement': '3'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'item_name': 'English Muffin', 'archive_revision_number': '1', 'item_value': '365', 'archived_attachment_count': '6', 'resource_requirement': '3'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '3', 'resource_capacity': '180'}}]
import gurobipy as gp
from gurobipy import GRB
products = [rec['values'] for rec in LEGACY_RECORDS if rec['source'] == 'products.csv']
capacities = [rec['values'] for rec in LEGACY_RECORDS if rec['source'] == 'capacity.csv']
item_names = [p['item_name'] for p in products]
profit = {}
resource_req = {}
for p in products:
    try:
        profit[p['item_name']] = int(p['item_value'])
        resource_req[p['item_name']] = int(p['resource_requirement'])
    except Exception as e:
        raise ValueError(f"Missing or invalid data for item {p['item_name']}: {e}")
if len(capacities) != 1 or 'resource_capacity' not in capacities[0]:
    raise ValueError('Missing or ambiguous resource capacity in LEGACY_RECORDS')
resource_capacity = int(capacities[0]['resource_capacity'])
for name in item_names:
    if name not in profit or name not in resource_req:
        raise ValueError(f'Missing profit or resource requirement for item {name}')
m = gp.Model('Bakery_Order_Optimization')
x = m.addVars(item_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[i] * x[i] for i in item_names)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((resource_req[i] * x[i] for i in item_names)) <= resource_capacity, name='storage')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')