import gurobipy as gp
from gurobipy import GRB
platforms = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
capacities = {'1': 1336, '2': 1754, '3': 1617, '4': 1119, '5': 1410, '6': 627, '7': 748, '8': 1540, '9': 1292, '10': 1138}
genres = ['Racing', 'Sports', 'Action', 'Adventure', 'RPG', 'Shooter', 'Strategy', 'Simulation', 'Puzzle', 'Fighting', 'Platformer', 'Survival', 'Horror', 'Sandbox', 'MMO']
values = {'Racing': 28, 'Sports': 69, 'Action': 20, 'Adventure': 62, 'RPG': 58, 'Shooter': 11, 'Strategy': 73, 'Simulation': 43, 'Puzzle': 28, 'Fighting': 57, 'Platformer': 92, 'Survival': 66, 'Horror': 14, 'Sandbox': 49, 'MMO': 12}
memory = {'Racing': 393, 'Sports': 195, 'Action': 192, 'Adventure': 155, 'RPG': 500, 'Shooter': 156, 'Strategy': 317, 'Simulation': 694, 'Puzzle': 751, 'Fighting': 467, 'Platformer': 796, 'Survival': 146, 'Horror': 269, 'Sandbox': 246, 'MMO': 652}
if set(platforms) != set(capacities.keys()):
    raise ValueError('Platform identifiers in capacities do not match platforms list.')
if set(genres) != set(values.keys()) or set(genres) != set(memory.keys()):
    raise ValueError('Genre identifiers in values or memory do not match genres list.')
m = gp.Model('Game_Store_Listing')
x = m.addVars(platforms, genres, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[g] * x[p, g] for p in platforms for g in genres)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((memory[g] * x[p, g] for g in genres)) <= capacities[p] for p in platforms), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')