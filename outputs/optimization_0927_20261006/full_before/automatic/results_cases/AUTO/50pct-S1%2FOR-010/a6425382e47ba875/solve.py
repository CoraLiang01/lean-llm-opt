LEGACY_OBSERVATION = '{"values": {"SectionID": "1", "archive_revision_number": "3", "Capacity": "100"}}\n{"values": {"SectionID": "2", "archive_revision_number": "7", "Capacity": "150"}}\n{"values": {"SectionID": "3", "archive_revision_number": "3", "Capacity": "120"}}\n{"values": {"SectionID": "4", "archive_revision_number": "4", "Capacity": "130"}}\n{"values": {"SectionID": "5", "archive_revision_number": "6", "Capacity": "90"}}\n{"values": {"SectionID": "6", "archive_revision_number": "4", "Capacity": "110"}}\n{"values": {"SectionID": "7", "archive_revision_number": "3", "Capacity": "160"}}\n{"values": {"SectionID": "8", "archive_revision_number": "8", "Capacity": "140"}}\n{"values": {"record_keeper_group": "Team B", "ProductName": "1", "Value": "10", "archive_revision_number": "1", "Weight": "2"}}\n{"values": {"record_keeper_group": "Team A", "ProductName": "2", "Value": "15", "archive_revision_number": "4", "Weight": "3"}}\n{"values": {"record_keeper_group": "Team C", "ProductName": "3", "Value": "8", "archive_revision_number": "2", "Weight": "1"}}\n{"values": {"record_keeper_group": "Team A", "ProductName": "4", "Value": "12", "archive_revision_number": "9", "Weight": "2"}}\n{"values": {"record_keeper_group": "Team A", "ProductName": "5", "Value": "20", "archive_revision_number": "3", "Weight": "4"}}\n{"values": {"record_keeper_group": "Team C", "ProductName": "6", "Value": "25", "archive_revision_number": "7", "Weight": "5"}}\n{"values": {"record_keeper_group": "Team A", "ProductName": "7", "Value": "5", "archive_revision_number": "2", "Weight": "1"}}\n{"values": {"record_keeper_group": "Team C", "ProductName": "8", "Value": "30", "archive_revision_number": "3", "Weight": "6"}}\n{"values": {"record_keeper_group": "Team C", "ProductName": "9", "Value": "18", "archive_revision_number": "7", "Weight": "3"}}\n{"values": {"record_keeper_group": "Team A", "ProductName": "10", "Value": "22", "archive_revision_number": "5", "Weight": "4"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'SectionID': '1', 'archive_revision_number': '3', 'Capacity': '100'}}, {'source': '', 'values': {'SectionID': '2', 'archive_revision_number': '7', 'Capacity': '150'}}, {'source': '', 'values': {'SectionID': '3', 'archive_revision_number': '3', 'Capacity': '120'}}, {'source': '', 'values': {'SectionID': '4', 'archive_revision_number': '4', 'Capacity': '130'}}, {'source': '', 'values': {'SectionID': '5', 'archive_revision_number': '6', 'Capacity': '90'}}, {'source': '', 'values': {'SectionID': '6', 'archive_revision_number': '4', 'Capacity': '110'}}, {'source': '', 'values': {'SectionID': '7', 'archive_revision_number': '3', 'Capacity': '160'}}, {'source': '', 'values': {'SectionID': '8', 'archive_revision_number': '8', 'Capacity': '140'}}, {'source': '', 'values': {'record_keeper_group': 'Team B', 'ProductName': '1', 'Value': '10', 'archive_revision_number': '1', 'Weight': '2'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'ProductName': '2', 'Value': '15', 'archive_revision_number': '4', 'Weight': '3'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'ProductName': '3', 'Value': '8', 'archive_revision_number': '2', 'Weight': '1'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'ProductName': '4', 'Value': '12', 'archive_revision_number': '9', 'Weight': '2'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'ProductName': '5', 'Value': '20', 'archive_revision_number': '3', 'Weight': '4'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'ProductName': '6', 'Value': '25', 'archive_revision_number': '7', 'Weight': '5'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'ProductName': '7', 'Value': '5', 'archive_revision_number': '2', 'Weight': '1'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'ProductName': '8', 'Value': '30', 'archive_revision_number': '3', 'Weight': '6'}}, {'source': '', 'values': {'record_keeper_group': 'Team C', 'ProductName': '9', 'Value': '18', 'archive_revision_number': '7', 'Weight': '3'}}, {'source': '', 'values': {'record_keeper_group': 'Team A', 'ProductName': '10', 'Value': '22', 'archive_revision_number': '5', 'Weight': '4'}}]
import gurobipy as gp
from gurobipy import GRB
sections = []
capacities = {}
products = []
values = {}
weights = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'SectionID' in v and 'Capacity' in v:
        sid = str(v['SectionID'])
        sections.append(sid)
        capacities[sid] = int(v['Capacity'])
    if 'ProductName' in v and 'Value' in v and ('Weight' in v):
        pid = str(v['ProductName'])
        products.append(pid)
        values[pid] = int(v['Value'])
        weights[pid] = int(v['Weight'])
sections = sorted(set(sections), key=int)
products = sorted(set(products), key=int)
for sid in sections:
    if sid not in capacities:
        raise ValueError(f'Missing capacity for section {sid}')
for pid in products:
    if pid not in values or pid not in weights:
        raise ValueError(f'Missing value/weight for product {pid}')
m = gp.Model('Supermarket_Section_Allocation')
x = m.addVars(sections, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[pid] * x[sid, pid] for sid in sections for pid in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[pid] * x[sid, pid] for pid in products)) <= capacities[sid] for sid in sections), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')