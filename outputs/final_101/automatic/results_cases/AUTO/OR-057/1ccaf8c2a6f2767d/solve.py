LEGACY_OBSERVATION = '{"values": {"PlatformID": "1", "Capacity": "995"}}\n{"values": {"PlatformID": "2", "Capacity": "1143"}}\n{"values": {"PlatformID": "3", "Capacity": "949"}}\n{"values": {"PlatformID": "4", "Capacity": "969"}}\n{"values": {"PlatformID": "5", "Capacity": "1649"}}\n{"values": {"PlatformID": "6", "Capacity": "870"}}\n{"values": {"PlatformID": "7", "Capacity": "1064"}}\n{"values": {"PlatformID": "8", "Capacity": "536"}}\n{"values": {"PlatformID": "9", "Capacity": "766"}}\n{"values": {"PlatformID": "10", "Capacity": "532"}}\n{"values": {"PlatformID": "11", "Capacity": "1703"}}\n{"values": {"PlatformID": "12", "Capacity": "1633"}}\n{"values": {"PlatformID": "13", "Capacity": "1203"}}\n{"values": {"PlatformID": "14", "Capacity": "1979"}}\n{"values": {"PlatformID": "15", "Capacity": "1797"}}\n{"values": {"ProductName": "Racing", "Value": "59", "Weight": "776"}}\n{"values": {"ProductName": "Sports", "Value": "83", "Weight": "573"}}\n{"values": {"ProductName": "Action", "Value": "94", "Weight": "127"}}\n{"values": {"ProductName": "Adventure", "Value": "41", "Weight": "138"}}\n{"values": {"ProductName": "RPG", "Value": "96", "Weight": "385"}}\n{"values": {"ProductName": "Shooter", "Value": "12", "Weight": "263"}}\n{"values": {"ProductName": "Strategy", "Value": "83", "Weight": "473"}}\n{"values": {"ProductName": "Simulation", "Value": "36", "Weight": "387"}}\n{"values": {"ProductName": "Puzzle", "Value": "56", "Weight": "390"}}\n{"values": {"ProductName": "Fighting", "Value": "27", "Weight": "556"}}\n{"values": {"ProductName": "Platformer", "Value": "47", "Weight": "601"}}\n{"values": {"ProductName": "Survival", "Value": "24", "Weight": "441"}}\n{"values": {"ProductName": "Horror", "Value": "14", "Weight": "603"}}\n{"values": {"ProductName": "Sandbox", "Value": "22", "Weight": "411"}}\n{"values": {"ProductName": "MMO", "Value": "17", "Weight": "652"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'PlatformID': '1', 'Capacity': '995'}}, {'source': '', 'values': {'PlatformID': '2', 'Capacity': '1143'}}, {'source': '', 'values': {'PlatformID': '3', 'Capacity': '949'}}, {'source': '', 'values': {'PlatformID': '4', 'Capacity': '969'}}, {'source': '', 'values': {'PlatformID': '5', 'Capacity': '1649'}}, {'source': '', 'values': {'PlatformID': '6', 'Capacity': '870'}}, {'source': '', 'values': {'PlatformID': '7', 'Capacity': '1064'}}, {'source': '', 'values': {'PlatformID': '8', 'Capacity': '536'}}, {'source': '', 'values': {'PlatformID': '9', 'Capacity': '766'}}, {'source': '', 'values': {'PlatformID': '10', 'Capacity': '532'}}, {'source': '', 'values': {'PlatformID': '11', 'Capacity': '1703'}}, {'source': '', 'values': {'PlatformID': '12', 'Capacity': '1633'}}, {'source': '', 'values': {'PlatformID': '13', 'Capacity': '1203'}}, {'source': '', 'values': {'PlatformID': '14', 'Capacity': '1979'}}, {'source': '', 'values': {'PlatformID': '15', 'Capacity': '1797'}}, {'source': '', 'values': {'ProductName': 'Racing', 'Value': '59', 'Weight': '776'}}, {'source': '', 'values': {'ProductName': 'Sports', 'Value': '83', 'Weight': '573'}}, {'source': '', 'values': {'ProductName': 'Action', 'Value': '94', 'Weight': '127'}}, {'source': '', 'values': {'ProductName': 'Adventure', 'Value': '41', 'Weight': '138'}}, {'source': '', 'values': {'ProductName': 'RPG', 'Value': '96', 'Weight': '385'}}, {'source': '', 'values': {'ProductName': 'Shooter', 'Value': '12', 'Weight': '263'}}, {'source': '', 'values': {'ProductName': 'Strategy', 'Value': '83', 'Weight': '473'}}, {'source': '', 'values': {'ProductName': 'Simulation', 'Value': '36', 'Weight': '387'}}, {'source': '', 'values': {'ProductName': 'Puzzle', 'Value': '56', 'Weight': '390'}}, {'source': '', 'values': {'ProductName': 'Fighting', 'Value': '27', 'Weight': '556'}}, {'source': '', 'values': {'ProductName': 'Platformer', 'Value': '47', 'Weight': '601'}}, {'source': '', 'values': {'ProductName': 'Survival', 'Value': '24', 'Weight': '441'}}, {'source': '', 'values': {'ProductName': 'Horror', 'Value': '14', 'Weight': '603'}}, {'source': '', 'values': {'ProductName': 'Sandbox', 'Value': '22', 'Weight': '411'}}, {'source': '', 'values': {'ProductName': 'MMO', 'Value': '17', 'Weight': '652'}}]
import gurobipy as gp
from gurobipy import GRB
platforms = []
capacities = {}
games = []
values = {}
weights = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'PlatformID' in v and 'Capacity' in v:
        pid = v['PlatformID']
        platforms.append(pid)
        capacities[pid] = int(v['Capacity'])
    if 'ProductName' in v and 'Value' in v and ('Weight' in v):
        pname = v['ProductName']
        games.append(pname)
        values[pname] = int(v['Value'])
        weights[pname] = int(v['Weight'])
if len(platforms) == 0 or len(games) == 0:
    raise ValueError('Missing platforms or games in LEGACY_RECORDS')
for pid in platforms:
    if pid not in capacities:
        raise ValueError(f'Missing capacity for platform {pid}')
for pname in games:
    if pname not in values or pname not in weights:
        raise ValueError(f'Missing value or weight for game {pname}')
m = gp.Model('Game_Platform_Allocation')
x = m.addVars(platforms, games, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[j] * x[i, j] for i in platforms for j in games)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x[i, j] for j in games)) <= capacities[i] for i in platforms), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')