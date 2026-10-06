LEGACY_OBSERVATION = 'capacity.csv\nPlatformId,Capacity\n1,1336\n2,1754\n3,1617\n4,1119\n5,1410\n6,627\n7,748\n8,1540\n9,1292\n10,1138\n\nproducts.csv\nProductName,Value,Weight\nRacing,28,393\nSports,69,195\nAction,20,192\nAdventure,62,155\nRPG,58,500\nShooter,11,156\nStrategy,73,317\nSimulation,43,694\nPuzzle,28,751\nFighting,57,467\nPlatformer,92,796\nSurvival,66,146\nHorror,14,269\nSandbox,49,246\nMMO,12,652'
LEGACY_RECORDS = [{'source': 'capacity.csv', 'values': {'PlatformId': '1', 'Capacity': '1336'}}, {'source': 'capacity.csv', 'values': {'PlatformId': '2', 'Capacity': '1754'}}, {'source': 'capacity.csv', 'values': {'PlatformId': '3', 'Capacity': '1617'}}, {'source': 'capacity.csv', 'values': {'PlatformId': '4', 'Capacity': '1119'}}, {'source': 'capacity.csv', 'values': {'PlatformId': '5', 'Capacity': '1410'}}, {'source': 'capacity.csv', 'values': {'PlatformId': '6', 'Capacity': '627'}}, {'source': 'capacity.csv', 'values': {'PlatformId': '7', 'Capacity': '748'}}, {'source': 'capacity.csv', 'values': {'PlatformId': '8', 'Capacity': '1540'}}, {'source': 'capacity.csv', 'values': {'PlatformId': '9', 'Capacity': '1292'}}, {'source': 'capacity.csv', 'values': {'PlatformId': '10', 'Capacity': '1138'}}, {'source': 'products.csv', 'values': {'ProductName': 'Racing', 'Value': '28', 'Weight': '393'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sports', 'Value': '69', 'Weight': '195'}}, {'source': 'products.csv', 'values': {'ProductName': 'Action', 'Value': '20', 'Weight': '192'}}, {'source': 'products.csv', 'values': {'ProductName': 'Adventure', 'Value': '62', 'Weight': '155'}}, {'source': 'products.csv', 'values': {'ProductName': 'RPG', 'Value': '58', 'Weight': '500'}}, {'source': 'products.csv', 'values': {'ProductName': 'Shooter', 'Value': '11', 'Weight': '156'}}, {'source': 'products.csv', 'values': {'ProductName': 'Strategy', 'Value': '73', 'Weight': '317'}}, {'source': 'products.csv', 'values': {'ProductName': 'Simulation', 'Value': '43', 'Weight': '694'}}, {'source': 'products.csv', 'values': {'ProductName': 'Puzzle', 'Value': '28', 'Weight': '751'}}, {'source': 'products.csv', 'values': {'ProductName': 'Fighting', 'Value': '57', 'Weight': '467'}}, {'source': 'products.csv', 'values': {'ProductName': 'Platformer', 'Value': '92', 'Weight': '796'}}, {'source': 'products.csv', 'values': {'ProductName': 'Survival', 'Value': '66', 'Weight': '146'}}, {'source': 'products.csv', 'values': {'ProductName': 'Horror', 'Value': '14', 'Weight': '269'}}, {'source': 'products.csv', 'values': {'ProductName': 'Sandbox', 'Value': '49', 'Weight': '246'}}, {'source': 'products.csv', 'values': {'ProductName': 'MMO', 'Value': '12', 'Weight': '652'}}]
from gurobipy import Model, GRB

def solve_game_listing_optimization():
    global LEGACY_RECORDS
    platforms = []
    capacities = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'capacity.csv':
            pid = rec['values']['PlatformId']
            cap = int(rec['values']['Capacity'])
            platforms.append(pid)
            capacities[pid] = cap
    genres = []
    values = {}
    weights = {}
    for rec in LEGACY_RECORDS:
        if rec['source'] == 'products.csv':
            g = rec['values']['ProductName']
            v = int(rec['values']['Value'])
            w = int(rec['values']['Weight'])
            genres.append(g)
            values[g] = v
            weights[g] = w
    platforms = list(dict.fromkeys(platforms))
    genres = list(dict.fromkeys(genres))
    for pid in platforms:
        if pid not in capacities:
            raise ValueError(f'Missing capacity for platform {pid}')
    for g in genres:
        if g not in values or g not in weights:
            raise ValueError(f'Missing value/weight for genre {g}')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(platforms, genres, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(sum((values[g] * x[pid, g] for pid in platforms for g in genres)), GRB.MAXIMIZE)
    for pid in platforms:
        m.addConstr(sum((weights[g] * x[pid, g] for g in genres)) <= capacities[pid], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for pid in platforms:
            for g in genres:
                var = x[pid, g]
                print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_game_listing_optimization()