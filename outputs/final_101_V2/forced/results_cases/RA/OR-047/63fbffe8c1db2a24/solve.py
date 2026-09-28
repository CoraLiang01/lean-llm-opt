LEGACY_OBSERVATION = '{"values": {"PlatformId": "1", "Capacity": "1336"}}\n{"values": {"PlatformId": "2", "Capacity": "1754"}}\n{"values": {"PlatformId": "3", "Capacity": "1617"}}\n{"values": {"PlatformId": "4", "Capacity": "1119"}}\n{"values": {"PlatformId": "5", "Capacity": "1410"}}\n{"values": {"PlatformId": "6", "Capacity": "627"}}\n{"values": {"PlatformId": "7", "Capacity": "748"}}\n{"values": {"PlatformId": "8", "Capacity": "1540"}}\n{"values": {"PlatformId": "9", "Capacity": "1292"}}\n{"values": {"PlatformId": "10", "Capacity": "1138"}}\n{"values": {"ProductName": "Racing", "Value": "28", "Weight": "393"}}\n{"values": {"ProductName": "Sports", "Value": "69", "Weight": "195"}}\n{"values": {"ProductName": "Action", "Value": "20", "Weight": "192"}}\n{"values": {"ProductName": "Adventure", "Value": "62", "Weight": "155"}}\n{"values": {"ProductName": "RPG", "Value": "58", "Weight": "500"}}\n{"values": {"ProductName": "Shooter", "Value": "11", "Weight": "156"}}\n{"values": {"ProductName": "Strategy", "Value": "73", "Weight": "317"}}\n{"values": {"ProductName": "Simulation", "Value": "43", "Weight": "694"}}\n{"values": {"ProductName": "Puzzle", "Value": "28", "Weight": "751"}}\n{"values": {"ProductName": "Fighting", "Value": "57", "Weight": "467"}}\n{"values": {"ProductName": "Platformer", "Value": "92", "Weight": "796"}}\n{"values": {"ProductName": "Survival", "Value": "66", "Weight": "146"}}\n{"values": {"ProductName": "Horror", "Value": "14", "Weight": "269"}}\n{"values": {"ProductName": "Sandbox", "Value": "49", "Weight": "246"}}\n{"values": {"ProductName": "MMO", "Value": "12", "Weight": "652"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'PlatformId': '1', 'Capacity': '1336'}}, {'source': '', 'values': {'PlatformId': '2', 'Capacity': '1754'}}, {'source': '', 'values': {'PlatformId': '3', 'Capacity': '1617'}}, {'source': '', 'values': {'PlatformId': '4', 'Capacity': '1119'}}, {'source': '', 'values': {'PlatformId': '5', 'Capacity': '1410'}}, {'source': '', 'values': {'PlatformId': '6', 'Capacity': '627'}}, {'source': '', 'values': {'PlatformId': '7', 'Capacity': '748'}}, {'source': '', 'values': {'PlatformId': '8', 'Capacity': '1540'}}, {'source': '', 'values': {'PlatformId': '9', 'Capacity': '1292'}}, {'source': '', 'values': {'PlatformId': '10', 'Capacity': '1138'}}, {'source': '', 'values': {'ProductName': 'Racing', 'Value': '28', 'Weight': '393'}}, {'source': '', 'values': {'ProductName': 'Sports', 'Value': '69', 'Weight': '195'}}, {'source': '', 'values': {'ProductName': 'Action', 'Value': '20', 'Weight': '192'}}, {'source': '', 'values': {'ProductName': 'Adventure', 'Value': '62', 'Weight': '155'}}, {'source': '', 'values': {'ProductName': 'RPG', 'Value': '58', 'Weight': '500'}}, {'source': '', 'values': {'ProductName': 'Shooter', 'Value': '11', 'Weight': '156'}}, {'source': '', 'values': {'ProductName': 'Strategy', 'Value': '73', 'Weight': '317'}}, {'source': '', 'values': {'ProductName': 'Simulation', 'Value': '43', 'Weight': '694'}}, {'source': '', 'values': {'ProductName': 'Puzzle', 'Value': '28', 'Weight': '751'}}, {'source': '', 'values': {'ProductName': 'Fighting', 'Value': '57', 'Weight': '467'}}, {'source': '', 'values': {'ProductName': 'Platformer', 'Value': '92', 'Weight': '796'}}, {'source': '', 'values': {'ProductName': 'Survival', 'Value': '66', 'Weight': '146'}}, {'source': '', 'values': {'ProductName': 'Horror', 'Value': '14', 'Weight': '269'}}, {'source': '', 'values': {'ProductName': 'Sandbox', 'Value': '49', 'Weight': '246'}}, {'source': '', 'values': {'ProductName': 'MMO', 'Value': '12', 'Weight': '652'}}]
import gurobipy as gp
from gurobipy import GRB
platforms = []
capacities = {}
genres = []
values = {}
weights = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'PlatformId' in v and 'Capacity' in v:
        pid = v['PlatformId']
        platforms.append(pid)
        capacities[pid] = int(v['Capacity'])
    if 'ProductName' in v and 'Value' in v and ('Weight' in v):
        pname = v['ProductName']
        genres.append(pname)
        values[pname] = int(v['Value'])
        weights[pname] = int(v['Weight'])
if len(platforms) == 0 or len(genres) == 0:
    raise ValueError('Missing platforms or genres in LEGACY_RECORDS')
for pid in platforms:
    if pid not in capacities:
        raise ValueError(f'Missing capacity for platform {pid}')
for pname in genres:
    if pname not in values or pname not in weights:
        raise ValueError(f'Missing value or weight for genre {pname}')
m = gp.Model('Game_Store_Listing')
x = m.addVars(platforms, genres, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[j] * x[i, j] for i in platforms for j in genres)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[j] * x[i, j] for j in genres)) <= capacities[i] for i in platforms), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')