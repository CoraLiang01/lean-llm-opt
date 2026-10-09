import gurobipy as gp
from gurobipy import GRB
platforms = {1: 1336, 2: 1754, 3: 1617, 4: 1119, 5: 1410, 6: 627, 7: 748, 8: 1540, 9: 1292, 10: 1138}
genres = [{'ProductName': 'Racing', 'Value': 28, 'Weight': 393}, {'ProductName': 'Sports', 'Value': 69, 'Weight': 195}, {'ProductName': 'Action', 'Value': 20, 'Weight': 192}, {'ProductName': 'Adventure', 'Value': 62, 'Weight': 155}, {'ProductName': 'RPG', 'Value': 58, 'Weight': 500}, {'ProductName': 'Shooter', 'Value': 11, 'Weight': 156}, {'ProductName': 'Strategy', 'Value': 73, 'Weight': 317}, {'ProductName': 'Simulation', 'Value': 43, 'Weight': 694}, {'ProductName': 'Puzzle', 'Value': 28, 'Weight': 751}, {'ProductName': 'Fighting', 'Value': 57, 'Weight': 467}, {'ProductName': 'Platformer', 'Value': 92, 'Weight': 796}, {'ProductName': 'Survival', 'Value': 66, 'Weight': 146}, {'ProductName': 'Horror', 'Value': 14, 'Weight': 269}, {'ProductName': 'Sandbox', 'Value': 49, 'Weight': 246}, {'ProductName': 'MMO', 'Value': 12, 'Weight': 652}]
platform_ids = list(platforms.keys())
genre_ids = list(range(1, 16))
v = {j: genres[j - 1]['Value'] for j in genre_ids}
w = {j: genres[j - 1]['Weight'] for j in genre_ids}
C = platforms
m = gp.Model('Game_Store_Listing')
x_vars = m.addVars(platform_ids, genre_ids, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((v[j] * x_vars[i, j] for i in platform_ids for j in genre_ids)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((w[j] * x_vars[i, j] for j in genre_ids)) <= C[i] for i in platform_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')