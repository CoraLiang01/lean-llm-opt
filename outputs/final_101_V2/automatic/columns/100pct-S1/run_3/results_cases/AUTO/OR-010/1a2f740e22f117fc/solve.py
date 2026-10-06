LEGACY_OBSERVATION = 'capacity.csv\nSectionID,archived_attachment_count,archive_revision_number,Capacity\n1,1,3,100\n2,1,7,150\n3,3,3,120\n4,2,4,130\n5,3,6,90\n6,3,4,110\n7,1,3,160\n8,3,8,140\n\nproducts.csv\nrecord_keeper_group,ProductName,Value,archive_revision_number,Weight,archived_attachment_count\nTeam B,1,10,1,2,4\nTeam A,2,15,4,3,3\nTeam C,3,8,2,1,2\nTeam A,4,12,9,2,2\nTeam A,5,20,3,4,2\nTeam C,6,25,7,5,2\nTeam A,7,5,2,1,3\nTeam C,8,30,3,6,3\nTeam C,9,18,7,3,1\nTeam A,10,22,5,4,2'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'SectionID': '1', 'archived_attachment_count': '1', 'archive_revision_number': '3', 'Capacity': '100'}}, {'source': 'capacity.csv', 'values': {'SectionID': '2', 'archived_attachment_count': '1', 'archive_revision_number': '7', 'Capacity': '150'}}, {'source': 'capacity.csv', 'values': {'SectionID': '3', 'archived_attachment_count': '3', 'archive_revision_number': '3', 'Capacity': '120'}}, {'source': 'capacity.csv', 'values': {'SectionID': '4', 'archived_attachment_count': '2', 'archive_revision_number': '4', 'Capacity': '130'}}, {'source': 'capacity.csv', 'values': {'SectionID': '5', 'archived_attachment_count': '3', 'archive_revision_number': '6', 'Capacity': '90'}}, {'source': 'capacity.csv', 'values': {'SectionID': '6', 'archived_attachment_count': '3', 'archive_revision_number': '4', 'Capacity': '110'}}, {'source': 'capacity.csv', 'values': {'SectionID': '7', 'archived_attachment_count': '1', 'archive_revision_number': '3', 'Capacity': '160'}}, {'source': 'capacity.csv', 'values': {'SectionID': '8', 'archived_attachment_count': '3', 'archive_revision_number': '8', 'Capacity': '140'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'ProductName': '1', 'Value': '10', 'archive_revision_number': '1', 'Weight': '2', 'archived_attachment_count': '4'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'ProductName': '2', 'Value': '15', 'archive_revision_number': '4', 'Weight': '3', 'archived_attachment_count': '3'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'ProductName': '3', 'Value': '8', 'archive_revision_number': '2', 'Weight': '1', 'archived_attachment_count': '2'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'ProductName': '4', 'Value': '12', 'archive_revision_number': '9', 'Weight': '2', 'archived_attachment_count': '2'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'ProductName': '5', 'Value': '20', 'archive_revision_number': '3', 'Weight': '4', 'archived_attachment_count': '2'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'ProductName': '6', 'Value': '25', 'archive_revision_number': '7', 'Weight': '5', 'archived_attachment_count': '2'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'ProductName': '7', 'Value': '5', 'archive_revision_number': '2', 'Weight': '1', 'archived_attachment_count': '3'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'ProductName': '8', 'Value': '30', 'archive_revision_number': '3', 'Weight': '6', 'archived_attachment_count': '3'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'ProductName': '9', 'Value': '18', 'archive_revision_number': '7', 'Weight': '3', 'archived_attachment_count': '1'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'ProductName': '10', 'Value': '22', 'archive_revision_number': '5', 'Weight': '4', 'archived_attachment_count': '2'}}]
import gurobipy as gp
from gurobipy import GRB
capacity_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
sections = [rec['values']['SectionID'] for rec in capacity_records]
capacities = {rec['values']['SectionID']: int(rec['values']['Capacity']) for rec in capacity_records}
products = [rec['values']['ProductName'] for rec in product_records]
values = {rec['values']['ProductName']: int(rec['values']['Value']) for rec in product_records}
weights = {rec['values']['ProductName']: int(rec['values']['Weight']) for rec in product_records}
if set(capacities.keys()) != set(sections):
    raise ValueError('Missing capacity data for some sections')
if set(values.keys()) != set(products) or set(weights.keys()) != set(products):
    raise ValueError('Missing value or weight data for some products')
m = gp.Model('Supermarket_Section_Allocation')
x = m.addVars(sections, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[j] * x[i, j] for i in sections for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x[i, j] for j in products)) <= capacities[i] for i in sections), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')