LEGACY_OBSERVATION = 'capacity.csv\nPlatformID,Capacity\n1,995\n2,1143\n3,949\n4,969\n5,1649\n6,870\n7,1064\n8,536\n9,766\n10,532\n11,1703\n12,1633\n13,1203\n14,1979\n15,1797\n\nproducts.csv\nProductName,Value,Weight\nRacing,59,776\nSports,83,573\nAction,94,127\nAdventure,41,138\nRPG,96,385\nShooter,12,263\nStrategy,83,473\nSimulation,36,387\nPuzzle,56,390\nFighting,27,556\nPlatformer,47,601\nSurvival,24,441\nHorror,14,603\nSandbox,22,411\nMMO,17,652'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'PlatformID': '1', 'Capacity': '995'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '2', 'Capacity': '1143'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '3', 'Capacity': '949'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '4', 'Capacity': '969'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '5', 'Capacity': '1649'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '6', 'Capacity': '870'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '7', 'Capacity': '1064'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '8', 'Capacity': '536'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '9', 'Capacity': '766'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '10', 'Capacity': '532'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '11', 'Capacity': '1703'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '12', 'Capacity': '1633'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '13', 'Capacity': '1203'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '14', 'Capacity': '1979'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '15', 'Capacity': '1797'}}, {'source': 'products.csv', 'values': {'ProductName': 'Racing', 'Value': '59', 'Weight': '776'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sports', 'Value': '83', 'Weight': '573'}}, {'source': 'products.csv', 'values': {'ProductName': 'Action', 'Value': '94', 'Weight': '127'}}, {'source': 'products.csv', 'values': {'ProductName': 'Adventure', 'Value': '41', 'Weight': '138'}}, {'source': 'products.csv', 'values': {'ProductName': 'RPG', 'Value': '96', 'Weight': '385'}}, {'source': 'products.csv', 'values': {'ProductName': 'Shooter', 'Value': '12', 'Weight': '263'}}, {'source': 'products.csv', 'values': {'ProductName': 'Strategy', 'Value': '83', 'Weight': '473'}}, {'source': 'products.csv', 'values': {'ProductName': 'Simulation', 'Value': '36', 'Weight': '387'}}, {'source': 'products.csv', 'values': {'ProductName': 'Puzzle', 'Value': '56', 'Weight': '390'}}, {'source': 'products.csv', 'values': {'ProductName': 'Fighting', 'Value': '27', 'Weight': '556'}}, {'source': 'products.csv', 'values': {'ProductName': 'Platformer', 'Value': '47', 'Weight': '601'}}, {'source': 'products.csv', 'values': {'ProductName': 'Survival', 'Value': '24', 'Weight': '441'}}, {'source': 'products.csv', 'values': {'ProductName': 'Horror', 'Value': '14', 'Weight': '603'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sandbox', 'Value': '22', 'Weight': '411'}}, {'source': 'products.csv', 'values': {'ProductName': 'MMO', 'Value': '17', 'Weight': '652'}}]
from gurobipy import Model, GRB
platforms = []
capacity = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'capacity.csv':
        pid = rec['values']['PlatformID']
        platforms.append(pid)
        capacity[pid] = int(rec['values']['Capacity'])
products = []
value = {}
weight = {}
for rec in LEGACY_RECORDS:
    if rec['source'] == 'products.csv':
        pname = rec['values']['ProductName']
        products.append(pname)
        value[pname] = int(rec['values']['Value'])
        weight[pname] = int(rec['values']['Weight'])
if len(platforms) != len(capacity):
    raise ValueError('Missing capacity data for some platforms.')
if len(products) != len(value) or len(products) != len(weight):
    raise ValueError('Missing value or weight data for some products.')
m = Model()
m.Params.MIPGap = 0.0001
x = m.addVars(platforms, products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(sum((value[j] * x[i, j] for i in platforms for j in products)), GRB.MAXIMIZE)
for i in platforms:
    m.addConstr(sum((weight[j] * x[i, j] for j in products)) <= capacity[i], name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print('ObjVal', m.ObjVal)
    for i in platforms:
        for j in products:
            v = x[i, j]
            print(v.VarName, v.X)
else:
    print('Status', m.Status)