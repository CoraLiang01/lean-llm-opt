import gurobipy as gp
from gurobipy import GRB
platforms = [{'PlatformId': '1', 'Capacity': 1336}, {'PlatformId': '2', 'Capacity': 1754}, {'PlatformId': '3', 'Capacity': 1617}, {'PlatformId': '4', 'Capacity': 1119}, {'PlatformId': '5', 'Capacity': 1410}, {'PlatformId': '6', 'Capacity': 627}, {'PlatformId': '7', 'Capacity': 748}, {'PlatformId': '8', 'Capacity': 1540}, {'PlatformId': '9', 'Capacity': 1292}, {'PlatformId': '10', 'Capacity': 1138}]
genres = [{'ProductName': 'Racing', 'Value': 28, 'Weight': 393}, {'ProductName': 'Sports', 'Value': 69, 'Weight': 195}, {'ProductName': 'Action', 'Value': 20, 'Weight': 192}, {'ProductName': 'Adventure', 'Value': 62, 'Weight': 155}, {'ProductName': 'RPG', 'Value': 58, 'Weight': 500}, {'ProductName': 'Shooter', 'Value': 11, 'Weight': 156}, {'ProductName': 'Strategy', 'Value': 73, 'Weight': 317}, {'ProductName': 'Simulation', 'Value': 43, 'Weight': 694}, {'ProductName': 'Puzzle', 'Value': 28, 'Weight': 751}, {'ProductName': 'Fighting', 'Value': 57, 'Weight': 467}, {'ProductName': 'Platformer', 'Value': 92, 'Weight': 796}, {'ProductName': 'Survival', 'Value': 66, 'Weight': 146}, {'ProductName': 'Horror', 'Value': 14, 'Weight': 269}, {'ProductName': 'Sandbox', 'Value': 49, 'Weight': 246}, {'ProductName': 'MMO', 'Value': 12, 'Weight': 652}]
platform_ids = [p['PlatformId'] for p in platforms]
genre_names = [g['ProductName'] for g in genres]
C = {p['PlatformId']: p['Capacity'] for p in platforms}
v = {g['ProductName']: g['Value'] for g in genres}
w = {g['ProductName']: g['Weight'] for g in genres}
for pid in platform_ids:
    if pid not in C:
        raise ValueError(f'Missing capacity for platform {pid}')
for gname in genre_names:
    if gname not in v or gname not in w:
        raise ValueError(f'Missing value or weight for genre {gname}')
m = gp.Model('Game_Store_Listing')
x = m.addVars(platform_ids, genre_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((v[j] * x[i, j] for i in platform_ids for j in genre_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((w[j] * x[i, j] for j in genre_names)) <= C[i] for i in platform_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')