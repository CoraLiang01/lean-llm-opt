import gurobipy as gp
from gurobipy import GRB
platforms = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
capacities = {'P1': 1336, 'P2': 1754, 'P3': 1617, 'P4': 1119, 'P5': 1410, 'P6': 627, 'P7': 748, 'P8': 1540, 'P9': 1292, 'P10': 1138}
genres = ['Racing', 'Sports', 'Action', 'Adventure', 'RPG', 'Shooter', 'Strategy', 'Simulation', 'Puzzle', 'Fighting', 'Platformer', 'Survival', 'Horror', 'Sandbox', 'MMO']
values = {'Racing': 28, 'Sports': 69, 'Action': 20, 'Adventure': 62, 'RPG': 58, 'Shooter': 11, 'Strategy': 73, 'Simulation': 43, 'Puzzle': 28, 'Fighting': 57, 'Platformer': 92, 'Survival': 66, 'Horror': 14, 'Sandbox': 49, 'MMO': 12}
weights = {'Racing': 393, 'Sports': 195, 'Action': 192, 'Adventure': 155, 'RPG': 500, 'Shooter': 156, 'Strategy': 317, 'Simulation': 694, 'Puzzle': 751, 'Fighting': 467, 'Platformer': 796, 'Survival': 146, 'Horror': 269, 'Sandbox': 246, 'MMO': 652}
for p in platforms:
    if p not in capacities:
        raise ValueError(f'Missing capacity for platform {p}')
for g in genres:
    if g not in values or g not in weights:
        raise ValueError(f'Missing value or weight for genre {g}')
m = gp.Model('Game_Store_Platform_Listing')
x = m.addVars(platforms, genres, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[g] * x[p, g] for p in platforms for g in genres)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[g] * x[p, g] for g in genres)) <= capacities[p] for p in platforms), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')