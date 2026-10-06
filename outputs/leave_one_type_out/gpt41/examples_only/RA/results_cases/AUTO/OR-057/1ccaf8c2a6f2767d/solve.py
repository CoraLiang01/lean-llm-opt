LEGACY_OBSERVATION = 'capacity.csv\nPlatformID,Capacity\n1,995\n2,1143\n3,949\n4,969\n5,1649\n6,870\n7,1064\n8,536\n9,766\n10,532\n11,1703\n12,1633\n13,1203\n14,1979\n15,1797\n\nproducts.csv\nProductName,Value,Weight\nRacing,59,776\nSports,83,573\nAction,94,127\nAdventure,41,138\nRPG,96,385\nShooter,12,263\nStrategy,83,473\nSimulation,36,387\nPuzzle,56,390\nFighting,27,556\nPlatformer,47,601\nSurvival,24,441\nHorror,14,603\nSandbox,22,411\nMMO,17,652'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'PlatformID': '1', 'Capacity': '995'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '2', 'Capacity': '1143'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '3', 'Capacity': '949'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '4', 'Capacity': '969'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '5', 'Capacity': '1649'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '6', 'Capacity': '870'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '7', 'Capacity': '1064'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '8', 'Capacity': '536'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '9', 'Capacity': '766'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '10', 'Capacity': '532'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '11', 'Capacity': '1703'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '12', 'Capacity': '1633'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '13', 'Capacity': '1203'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '14', 'Capacity': '1979'}}, {'source': 'capacity.csv', 'values': {'PlatformID': '15', 'Capacity': '1797'}}, {'source': 'products.csv', 'values': {'ProductName': 'Racing', 'Value': '59', 'Weight': '776'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sports', 'Value': '83', 'Weight': '573'}}, {'source': 'products.csv', 'values': {'ProductName': 'Action', 'Value': '94', 'Weight': '127'}}, {'source': 'products.csv', 'values': {'ProductName': 'Adventure', 'Value': '41', 'Weight': '138'}}, {'source': 'products.csv', 'values': {'ProductName': 'RPG', 'Value': '96', 'Weight': '385'}}, {'source': 'products.csv', 'values': {'ProductName': 'Shooter', 'Value': '12', 'Weight': '263'}}, {'source': 'products.csv', 'values': {'ProductName': 'Strategy', 'Value': '83', 'Weight': '473'}}, {'source': 'products.csv', 'values': {'ProductName': 'Simulation', 'Value': '36', 'Weight': '387'}}, {'source': 'products.csv', 'values': {'ProductName': 'Puzzle', 'Value': '56', 'Weight': '390'}}, {'source': 'products.csv', 'values': {'ProductName': 'Fighting', 'Value': '27', 'Weight': '556'}}, {'source': 'products.csv', 'values': {'ProductName': 'Platformer', 'Value': '47', 'Weight': '601'}}, {'source': 'products.csv', 'values': {'ProductName': 'Survival', 'Value': '24', 'Weight': '441'}}, {'source': 'products.csv', 'values': {'ProductName': 'Horror', 'Value': '14', 'Weight': '603'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sandbox', 'Value': '22', 'Weight': '411'}}, {'source': 'products.csv', 'values': {'ProductName': 'MMO', 'Value': '17', 'Weight': '652'}}]
from gurobipy import Model, GRB

def solve_game_platform_allocation():
    platforms = []
    capacities = {}
    products = []
    values = {}
    weights = {}
    product_names = []
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            pid = int(rec['values']['PlatformID'])
            cap = int(rec['values']['Capacity'])
            platforms.append(pid)
            capacities[pid] = cap
        elif rec['source'] == 'products.csv':
            pname = rec['values']['ProductName']
            val = int(rec['values']['Value'])
            wt = int(rec['values']['Weight'])
            products.append(pname)
            values[pname] = val
            weights[pname] = wt
            product_names.append(pname)
    platforms.sort()
    if len(platforms) != 15:
        raise ValueError('Expected 15 platforms, got %d' % len(platforms))
    if len(products) != 15:
        raise ValueError('Expected 15 products, got %d' % len(products))
    for pid in platforms:
        if pid not in capacities:
            raise ValueError('Missing capacity for platform %s' % pid)
    for pname in product_names:
        if pname not in values or pname not in weights:
            raise ValueError('Missing value/weight for product %s' % pname)
    m = Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(platforms, product_names, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((values[pname] * x[pid, pname] for pid in platforms for pname in product_names)), GRB.MAXIMIZE)
    for pid in platforms:
        m.addConstr(sum((weights[pname] * x[pid, pname] for pname in product_names)) <= capacities[pid], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for pid in platforms:
            for pname in product_names:
                var = x[pid, pname]
                print(var.VarName, var.X)
    else:
        print('Solver status:', m.Status)
m = solve_game_platform_allocation()