LEGACY_OBSERVATION = '{"values": {"archive_revision_number": "1", "Capacity": "875"}}\n{"values": {"ProductName": "Spinach", "Weight": "230", "archived_attachment_count": "2", "archive_revision_number": "1", "Value": "64", "record_keeper_group": "Team A"}}\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "637", "archived_attachment_count": "6", "archive_revision_number": "1", "Value": "75", "record_keeper_group": "Team B"}}\n{"values": {"ProductName": "Apples", "Weight": "773", "archived_attachment_count": "4", "archive_revision_number": "1", "Value": "68", "record_keeper_group": "Team C"}}\n{"values": {"ProductName": "Carrots", "Weight": "653", "archived_attachment_count": "4", "archive_revision_number": "8", "Value": "11", "record_keeper_group": "Team C"}}\n{"values": {"ProductName": "Basil", "Weight": "755", "archived_attachment_count": "6", "archive_revision_number": "5", "Value": "91", "record_keeper_group": "Team A"}}\n{"values": {"ProductName": "Potatoes", "Weight": "670", "archived_attachment_count": "6", "archive_revision_number": "3", "Value": "31", "record_keeper_group": "Team B"}}\n{"values": {"ProductName": "Green Beans", "Weight": "505", "archived_attachment_count": "2", "archive_revision_number": "3", "Value": "90", "record_keeper_group": "Team C"}}\n{"values": {"ProductName": "Blueberries", "Weight": "821", "archived_attachment_count": "3", "archive_revision_number": "4", "Value": "56", "record_keeper_group": "Team B"}}\n{"values": {"ProductName": "Oranges", "Weight": "83", "archived_attachment_count": "2", "archive_revision_number": "5", "Value": "10", "record_keeper_group": "Team B"}}\n{"values": {"ProductName": "Watermelons", "Weight": "249", "archived_attachment_count": "2", "archive_revision_number": "8", "Value": "24", "record_keeper_group": "Team B"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'archive_revision_number': '1', 'Capacity': '875'}}, {'source': '', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'archived_attachment_count': '2', 'archive_revision_number': '1', 'Value': '64', 'record_keeper_group': 'Team A'}}, {'source': '', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'archived_attachment_count': '6', 'archive_revision_number': '1', 'Value': '75', 'record_keeper_group': 'Team B'}}, {'source': '', 'values': {'ProductName': 'Apples', 'Weight': '773', 'archived_attachment_count': '4', 'archive_revision_number': '1', 'Value': '68', 'record_keeper_group': 'Team C'}}, {'source': '', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'archived_attachment_count': '4', 'archive_revision_number': '8', 'Value': '11', 'record_keeper_group': 'Team C'}}, {'source': '', 'values': {'ProductName': 'Basil', 'Weight': '755', 'archived_attachment_count': '6', 'archive_revision_number': '5', 'Value': '91', 'record_keeper_group': 'Team A'}}, {'source': '', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'archived_attachment_count': '6', 'archive_revision_number': '3', 'Value': '31', 'record_keeper_group': 'Team B'}}, {'source': '', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'archived_attachment_count': '2', 'archive_revision_number': '3', 'Value': '90', 'record_keeper_group': 'Team C'}}, {'source': '', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'archived_attachment_count': '3', 'archive_revision_number': '4', 'Value': '56', 'record_keeper_group': 'Team B'}}, {'source': '', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'archived_attachment_count': '2', 'archive_revision_number': '5', 'Value': '10', 'record_keeper_group': 'Team B'}}, {'source': '', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'archived_attachment_count': '2', 'archive_revision_number': '8', 'Value': '24', 'record_keeper_group': 'Team B'}}]
import gurobipy as gp
from gurobipy import GRB
products = []
value = {}
weight = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if 'ProductName' in vals:
        pname = vals['ProductName']
        products.append(pname)
        value[pname] = int(vals['Value'])
        weight[pname] = int(vals['Weight'])
    elif 'Capacity' in vals:
        capacity = int(vals['Capacity'])
if set(value.keys()) != set(products) or set(weight.keys()) != set(products):
    raise ValueError('Missing value or weight data for some products.')
if not isinstance(capacity, int):
    raise ValueError('Missing or invalid capacity.')
m = gp.Model('SupermarketStock')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x[p] for p in products)) <= capacity, name='stock_capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')