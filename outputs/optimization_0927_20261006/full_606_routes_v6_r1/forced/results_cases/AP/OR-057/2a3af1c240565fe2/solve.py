import gurobipy as gp
from gurobipy import GRB
platforms = [{'PlatformID': '1', 'Capacity': 995}, {'PlatformID': '2', 'Capacity': 1143}, {'PlatformID': '3', 'Capacity': 949}, {'PlatformID': '4', 'Capacity': 969}, {'PlatformID': '5', 'Capacity': 1649}, {'PlatformID': '6', 'Capacity': 870}, {'PlatformID': '7', 'Capacity': 1064}, {'PlatformID': '8', 'Capacity': 536}, {'PlatformID': '9', 'Capacity': 766}, {'PlatformID': '10', 'Capacity': 532}, {'PlatformID': '11', 'Capacity': 1703}, {'PlatformID': '12', 'Capacity': 1633}, {'PlatformID': '13', 'Capacity': 1203}, {'PlatformID': '14', 'Capacity': 1979}, {'PlatformID': '15', 'Capacity': 1797}]
games = [{'ProductName': 'Racing', 'Value': 59, 'Weight': 776}, {'ProductName': 'Sports', 'Value': 83, 'Weight': 573}, {'ProductName': 'Action', 'Value': 94, 'Weight': 127}, {'ProductName': 'Adventure', 'Value': 41, 'Weight': 138}, {'ProductName': 'RPG', 'Value': 96, 'Weight': 385}, {'ProductName': 'Shooter', 'Value': 12, 'Weight': 263}, {'ProductName': 'Strategy', 'Value': 83, 'Weight': 473}, {'ProductName': 'Simulation', 'Value': 36, 'Weight': 387}, {'ProductName': 'Puzzle', 'Value': 56, 'Weight': 390}, {'ProductName': 'Fighting', 'Value': 27, 'Weight': 556}, {'ProductName': 'Platformer', 'Value': 47, 'Weight': 601}, {'ProductName': 'Survival', 'Value': 24, 'Weight': 441}, {'ProductName': 'Horror', 'Value': 14, 'Weight': 603}, {'ProductName': 'Sandbox', 'Value': 22, 'Weight': 411}, {'ProductName': 'MMO', 'Value': 17, 'Weight': 652}]
platform_ids = [p['PlatformID'] for p in platforms]
game_names = [g['ProductName'] for g in games]
C = {p['PlatformID']: p['Capacity'] for p in platforms}
v = {g['ProductName']: g['Value'] for g in games}
w = {g['ProductName']: g['Weight'] for g in games}
m = gp.Model('Game_Platform_Listing')
x_vars = m.addVars(platform_ids, game_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((v[j] * x_vars[i, j] for i in platform_ids for j in game_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((w[j] * x_vars[i, j] for j in game_names)) <= C[i] for i in platform_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')