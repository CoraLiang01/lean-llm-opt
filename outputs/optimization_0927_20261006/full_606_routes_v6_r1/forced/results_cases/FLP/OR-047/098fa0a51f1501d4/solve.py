import gurobipy as gp
from gurobipy import GRB
platforms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
genres = ['Racing', 'Sports', 'Action', 'Adventure', 'RPG', 'Shooter', 'Strategy', 'Simulation', 'Puzzle', 'Fighting', 'Platformer', 'Survival', 'Horror', 'Sandbox', 'MMO']
capacities = {1: 1336, 2: 1754, 3: 1617, 4: 1119, 5: 1410, 6: 627, 7: 748, 8: 1540, 9: 1292, 10: 1138}
genre_value = {'Racing': 28, 'Sports': 69, 'Action': 20, 'Adventure': 62, 'RPG': 58, 'Shooter': 11, 'Strategy': 73, 'Simulation': 43, 'Puzzle': 28, 'Fighting': 57, 'Platformer': 92, 'Survival': 66, 'Horror': 14, 'Sandbox': 49, 'MMO': 12}
genre_weight = {'Racing': 393, 'Sports': 195, 'Action': 192, 'Adventure': 155, 'RPG': 500, 'Shooter': 156, 'Strategy': 317, 'Simulation': 694, 'Puzzle': 751, 'Fighting': 467, 'Platformer': 796, 'Survival': 146, 'Horror': 269, 'Sandbox': 246, 'MMO': 652}
for i in platforms:
    if i not in capacities:
        raise ValueError(f'Missing capacity for platform {i}')
for j in genres:
    if j not in genre_value or j not in genre_weight:
        raise ValueError(f'Missing value or weight for genre {j}')
m = gp.Model('Game_Store_Listing')
x_vars = m.addVars(platforms, genres, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((genre_value[j] * x_vars[i, j] for i in platforms for j in genres)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((genre_weight[j] * x_vars[i, j] for j in genres)) <= capacities[i] for i in platforms), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')