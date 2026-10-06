LEGACY_OBSERVATION = 'capacity.csv\narchive_revision_number,Capacity\n7,586\n\nproducts.csv\nrecord_keeper_group,archive_revision_number,ProductName,Value,Weight\nTeam B,9,Queens,469,954\nTeam C,6,Brooklyn,290,650\nTeam C,1,Manhattan,236,961\nTeam B,2,Bronx,235,950\nTeam B,6,Staten Island,745,379\nTeam B,9,Harlem,684,776\nTeam A,2,Upper East Side,444,381\nTeam C,5,Lower Manhattan,172,808\nTeam B,9,Midtown,1000,937\nTeam B,1,Long Island City,336,608\nTeam B,3,Williamsburg,546,912\nTeam C,1,Bushwick,535,391\nTeam A,5,Flatbush,539,465\nTeam C,7,Greenpoint,831,490\nTeam A,6,Park Slope,139,918\nTeam B,1,Astoria,432,787\nTeam C,1,Jackson Heights,627,347\nTeam C,7,Flushing,629,274\nTeam A,6,Sunnyside,292,642\nTeam B,7,Ditmars,978,130'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'archive_revision_number': '7', 'Capacity': '586'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '9', 'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '6', 'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '1', 'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '2', 'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '6', 'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '9', 'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '2', 'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '5', 'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '9', 'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '1', 'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '3', 'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '1', 'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '5', 'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '7', 'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '6', 'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '1', 'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '1', 'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '7', 'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '6', 'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '7', 'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}]
import gurobipy as gp
from gurobipy import GRB
capacity_records = [r for r in LEGACY_RECORDS if r['source'] == 'capacity.csv']
if not capacity_records:
    raise ValueError('No capacity.csv records found in LEGACY_RECORDS')
capacity = None
for rec in capacity_records:
    vals = rec['values']
    if 'Capacity' in vals:
        capacity = int(vals['Capacity'])
if capacity is None:
    raise ValueError('No Capacity value found in capacity.csv records')
product_records = [r for r in LEGACY_RECORDS if r['source'] == 'products.csv']
if not product_records:
    raise ValueError('No products.csv records found in LEGACY_RECORDS')
areas = []
value = {}
weight = {}
for rec in product_records:
    vals = rec['values']
    area = vals['ProductName']
    if area in areas:
        raise ValueError(f'Duplicate area found: {area}')
    areas.append(area)
    if 'Value' not in vals or 'Weight' not in vals:
        raise ValueError(f'Missing Value or Weight for area {area}')
    value[area] = int(vals['Value'])
    weight[area] = int(vals['Weight'])
if set(value.keys()) != set(areas) or set(weight.keys()) != set(areas):
    raise ValueError('Mismatch in area keys for value/weight')
m = gp.Model('NY_Development')
x = m.addVars(areas, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in areas)) <= capacity, name='cap')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')