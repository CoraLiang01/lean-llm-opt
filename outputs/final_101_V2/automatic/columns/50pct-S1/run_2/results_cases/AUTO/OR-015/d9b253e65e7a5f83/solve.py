LEGACY_OBSERVATION = 'capacity.csv\narchive_revision_number,resource_id,resource_capacity\n5,1,500\n8,2,700\n1,3,600\n2,4,800\n2,5,550\n7,6,900\n4,7,650\n1,8,750\n2,9,820\n2,10,570\n\nproducts.csv\nrecord_keeper_group,item_name,item_value,resource_requirement,archive_revision_number\nTeam B,1,50,10,4\nTeam A,2,70,20,3\nTeam C,3,30,5,8\nTeam B,4,60,15,5\nTeam B,5,80,25,2\nTeam C,6,90,30,1\nTeam B,7,40,12,5\nTeam B,8,100,35,8\nTeam C,9,55,10,2\nTeam C,10,75,20,6\nTeam C,11,65,18,8\nTeam B,12,95,28,3\nTeam A,13,45,8,8\nTeam A,14,85,22,4\nTeam B,15,70,25,3\nTeam A,16,110,40,3\nTeam A,17,50,14,2\nTeam B,18,60,16,5\nTeam C,19,120,50,8\nTeam B,20,100,30,8'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'archive_revision_number': '5', 'resource_id': '1', 'resource_capacity': '500'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '8', 'resource_id': '2', 'resource_capacity': '700'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '1', 'resource_id': '3', 'resource_capacity': '600'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '2', 'resource_id': '4', 'resource_capacity': '800'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '2', 'resource_id': '5', 'resource_capacity': '550'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '7', 'resource_id': '6', 'resource_capacity': '900'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '4', 'resource_id': '7', 'resource_capacity': '650'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '1', 'resource_id': '8', 'resource_capacity': '750'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '2', 'resource_id': '9', 'resource_capacity': '820'}}, {'source': 'capacity.csv', 'values': {'archive_revision_number': '2', 'resource_id': '10', 'resource_capacity': '570'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'item_name': '1', 'item_value': '50', 'resource_requirement': '10', 'archive_revision_number': '4'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'item_name': '2', 'item_value': '70', 'resource_requirement': '20', 'archive_revision_number': '3'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'item_name': '3', 'item_value': '30', 'resource_requirement': '5', 'archive_revision_number': '8'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'item_name': '4', 'item_value': '60', 'resource_requirement': '15', 'archive_revision_number': '5'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'item_name': '5', 'item_value': '80', 'resource_requirement': '25', 'archive_revision_number': '2'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'item_name': '6', 'item_value': '90', 'resource_requirement': '30', 'archive_revision_number': '1'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'item_name': '7', 'item_value': '40', 'resource_requirement': '12', 'archive_revision_number': '5'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'item_name': '8', 'item_value': '100', 'resource_requirement': '35', 'archive_revision_number': '8'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'item_name': '9', 'item_value': '55', 'resource_requirement': '10', 'archive_revision_number': '2'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'item_name': '10', 'item_value': '75', 'resource_requirement': '20', 'archive_revision_number': '6'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'item_name': '11', 'item_value': '65', 'resource_requirement': '18', 'archive_revision_number': '8'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'item_name': '12', 'item_value': '95', 'resource_requirement': '28', 'archive_revision_number': '3'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'item_name': '13', 'item_value': '45', 'resource_requirement': '8', 'archive_revision_number': '8'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'item_name': '14', 'item_value': '85', 'resource_requirement': '22', 'archive_revision_number': '4'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'item_name': '15', 'item_value': '70', 'resource_requirement': '25', 'archive_revision_number': '3'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'item_name': '16', 'item_value': '110', 'resource_requirement': '40', 'archive_revision_number': '3'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'item_name': '17', 'item_value': '50', 'resource_requirement': '14', 'archive_revision_number': '2'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'item_name': '18', 'item_value': '60', 'resource_requirement': '16', 'archive_revision_number': '5'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'item_name': '19', 'item_value': '120', 'resource_requirement': '50', 'archive_revision_number': '8'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'item_name': '20', 'item_value': '100', 'resource_requirement': '30', 'archive_revision_number': '8'}}]
import gurobipy as gp
from gurobipy import GRB
shelves = []
capacity = {}
products = []
value = {}
requirement = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        rid = str(rec['values']['resource_id'])
        shelves.append(rid)
        capacity[rid] = int(rec['values']['resource_capacity'])
    elif rec['source'] == 'products.csv':
        pid = str(rec['values']['item_name'])
        products.append(pid)
        value[pid] = int(rec['values']['item_value'])
        requirement[pid] = int(rec['values']['resource_requirement'])
shelves = list(dict.fromkeys(shelves))
products = list(dict.fromkeys(products))
for rid in shelves:
    if rid not in capacity:
        raise ValueError(f'Missing capacity for shelf {rid}')
for pid in products:
    if pid not in value or pid not in requirement:
        raise ValueError(f'Missing value or requirement for product {pid}')
m = gp.Model('BigMart_Shelf_Allocation')
x = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[pid] * x[rid, pid] for rid in shelves for pid in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((requirement[pid] * x[rid, pid] for pid in products)) <= capacity[rid] for rid in shelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')