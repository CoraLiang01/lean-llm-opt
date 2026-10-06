LEGACY_OBSERVATION = 'capacity.csv\narchive_revision_number,Capacity\n7,586\n\nproducts.csv\nrecord_keeper_group,archive_revision_number,ProductName,Value,Weight\nTeam B,9,Queens,469,954\nTeam C,6,Brooklyn,290,650\nTeam C,1,Manhattan,236,961\nTeam B,2,Bronx,235,950\nTeam B,6,Staten Island,745,379\nTeam B,9,Harlem,684,776\nTeam A,2,Upper East Side,444,381\nTeam C,5,Lower Manhattan,172,808\nTeam B,9,Midtown,1000,937\nTeam B,1,Long Island City,336,608\nTeam B,3,Williamsburg,546,912\nTeam C,1,Bushwick,535,391\nTeam A,5,Flatbush,539,465\nTeam C,7,Greenpoint,831,490\nTeam A,6,Park Slope,139,918\nTeam B,1,Astoria,432,787\nTeam C,1,Jackson Heights,627,347\nTeam C,7,Flushing,629,274\nTeam A,6,Sunnyside,292,642\nTeam B,7,Ditmars,978,130'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'archive_revision_number': '7', 'Capacity': '586'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '9', 'ProductName': 'Queens', 'Value': '469', 'Weight': '954'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '6', 'ProductName': 'Brooklyn', 'Value': '290', 'Weight': '650'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '1', 'ProductName': 'Manhattan', 'Value': '236', 'Weight': '961'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '2', 'ProductName': 'Bronx', 'Value': '235', 'Weight': '950'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '6', 'ProductName': 'Staten Island', 'Value': '745', 'Weight': '379'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '9', 'ProductName': 'Harlem', 'Value': '684', 'Weight': '776'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '2', 'ProductName': 'Upper East Side', 'Value': '444', 'Weight': '381'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '5', 'ProductName': 'Lower Manhattan', 'Value': '172', 'Weight': '808'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '9', 'ProductName': 'Midtown', 'Value': '1000', 'Weight': '937'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '1', 'ProductName': 'Long Island City', 'Value': '336', 'Weight': '608'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '3', 'ProductName': 'Williamsburg', 'Value': '546', 'Weight': '912'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '1', 'ProductName': 'Bushwick', 'Value': '535', 'Weight': '391'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '5', 'ProductName': 'Flatbush', 'Value': '539', 'Weight': '465'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '7', 'ProductName': 'Greenpoint', 'Value': '831', 'Weight': '490'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '6', 'ProductName': 'Park Slope', 'Value': '139', 'Weight': '918'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '1', 'ProductName': 'Astoria', 'Value': '432', 'Weight': '787'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '1', 'ProductName': 'Jackson Heights', 'Value': '627', 'Weight': '347'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team C', 'archive_revision_number': '7', 'ProductName': 'Flushing', 'Value': '629', 'Weight': '274'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team A', 'archive_revision_number': '6', 'ProductName': 'Sunnyside', 'Value': '292', 'Weight': '642'}}, {'source': 'products.csv', 'values': {'record_keeper_group': 'Team B', 'archive_revision_number': '7', 'ProductName': 'Ditmars', 'Value': '978', 'Weight': '130'}}]
import gurobipy as gp
from gurobipy import GRB
areas = []
value = {}
weight = {}
capacity = None
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        pname = rec['values']['ProductName']
        areas.append(pname)
        value[pname] = int(rec['values']['Value'])
        weight[pname] = int(rec['values']['Weight'])
    elif rec['source'] == 'capacity.csv':
        if 'Capacity' in rec['values']:
            capacity = int(rec['values']['Capacity'])
if capacity is None:
    raise ValueError('Missing overall development capacity in LEGACY_RECORDS.')
for a in areas:
    if a not in value or a not in weight:
        raise ValueError(f'Missing value or weight for area {a}.')
m = gp.Model('NY_Development')
x = m.addVars(areas, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[a] * x[a] for a in areas)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[a] * x[a] for a in areas)) <= capacity, name='capacity')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')