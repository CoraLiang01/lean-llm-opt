LEGACY_OBSERVATION = '{"values": {"archive_revision_number": "1", "Capacity": "875"}}\n{"values": {"ProductName": "Spinach", "Weight": "230", "archive_revision_number": "1", "Value": "64", "record_keeper_group": "Team A"}}\n{"values": {"ProductName": "Shiitake Mushrooms", "Weight": "637", "archive_revision_number": "1", "Value": "75", "record_keeper_group": "Team B"}}\n{"values": {"ProductName": "Apples", "Weight": "773", "archive_revision_number": "1", "Value": "68", "record_keeper_group": "Team C"}}\n{"values": {"ProductName": "Carrots", "Weight": "653", "archive_revision_number": "8", "Value": "11", "record_keeper_group": "Team C"}}\n{"values": {"ProductName": "Basil", "Weight": "755", "archive_revision_number": "5", "Value": "91", "record_keeper_group": "Team A"}}\n{"values": {"ProductName": "Potatoes", "Weight": "670", "archive_revision_number": "3", "Value": "31", "record_keeper_group": "Team B"}}\n{"values": {"ProductName": "Green Beans", "Weight": "505", "archive_revision_number": "3", "Value": "90", "record_keeper_group": "Team C"}}\n{"values": {"ProductName": "Blueberries", "Weight": "821", "archive_revision_number": "4", "Value": "56", "record_keeper_group": "Team B"}}\n{"values": {"ProductName": "Oranges", "Weight": "83", "archive_revision_number": "5", "Value": "10", "record_keeper_group": "Team B"}}\n{"values": {"ProductName": "Watermelons", "Weight": "249", "archive_revision_number": "8", "Value": "24", "record_keeper_group": "Team B"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'archive_revision_number': '1', 'Capacity': '875'}}, {'source': '', 'values': {'ProductName': 'Spinach', 'Weight': '230', 'archive_revision_number': '1', 'Value': '64', 'record_keeper_group': 'Team A'}}, {'source': '', 'values': {'ProductName': 'Shiitake Mushrooms', 'Weight': '637', 'archive_revision_number': '1', 'Value': '75', 'record_keeper_group': 'Team B'}}, {'source': '', 'values': {'ProductName': 'Apples', 'Weight': '773', 'archive_revision_number': '1', 'Value': '68', 'record_keeper_group': 'Team C'}}, {'source': '', 'values': {'ProductName': 'Carrots', 'Weight': '653', 'archive_revision_number': '8', 'Value': '11', 'record_keeper_group': 'Team C'}}, {'source': '', 'values': {'ProductName': 'Basil', 'Weight': '755', 'archive_revision_number': '5', 'Value': '91', 'record_keeper_group': 'Team A'}}, {'source': '', 'values': {'ProductName': 'Potatoes', 'Weight': '670', 'archive_revision_number': '3', 'Value': '31', 'record_keeper_group': 'Team B'}}, {'source': '', 'values': {'ProductName': 'Green Beans', 'Weight': '505', 'archive_revision_number': '3', 'Value': '90', 'record_keeper_group': 'Team C'}}, {'source': '', 'values': {'ProductName': 'Blueberries', 'Weight': '821', 'archive_revision_number': '4', 'Value': '56', 'record_keeper_group': 'Team B'}}, {'source': '', 'values': {'ProductName': 'Oranges', 'Weight': '83', 'archive_revision_number': '5', 'Value': '10', 'record_keeper_group': 'Team B'}}, {'source': '', 'values': {'ProductName': 'Watermelons', 'Weight': '249', 'archive_revision_number': '8', 'Value': '24', 'record_keeper_group': 'Team B'}}]
import gurobipy as gp
from gurobipy import GRB
records = LEGACY_RECORDS
product_records = [r for r in records if 'ProductName' in r['values']]
if not product_records:
    raise ValueError('No product records found in LEGACY_RECORDS.')
products = [r['values']['ProductName'] for r in product_records]
try:
    value = {r['values']['ProductName']: int(r['values']['Value']) for r in product_records}
    weight = {r['values']['ProductName']: int(r['values']['Weight']) for r in product_records}
except KeyError as e:
    raise ValueError(f'Missing Value or Weight for product: {e}')
capacity_records = [r for r in records if 'Capacity' in r['values']]
if not capacity_records:
    raise ValueError('No capacity record found in LEGACY_RECORDS.')
try:
    total_capacity = int(capacity_records[0]['values']['Capacity'])
except Exception as e:
    raise ValueError(f'Invalid capacity value: {e}')
if set(value.keys()) != set(products) or set(weight.keys()) != set(products):
    raise ValueError('Mismatch in product identifiers between value/weight and products list.')
m = gp.Model('Supermarket_Stock')
x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x[p] for p in products)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[p] * x[p] for p in products)) <= total_capacity, name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')