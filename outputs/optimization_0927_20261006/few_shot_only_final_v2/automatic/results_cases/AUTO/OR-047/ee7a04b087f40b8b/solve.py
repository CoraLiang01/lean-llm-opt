import gurobipy as gp
from gurobipy import GRB
platforms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
genres = ['Racing', 'Sports', 'Action', 'Adventure', 'RPG', 'Shooter', 'Strategy', 'Simulation', 'Puzzle', 'Fighting', 'Platformer', 'Survival', 'Horror', 'Sandbox', 'MMO']
capacity = {1: 1336, 2: 1754, 3: 1617, 4: 1119, 5: 1410, 6: 627, 7: 748, 8: 1540, 9: 1292, 10: 1138}
value = {'Racing': 28, 'Sports': 69, 'Action': 20, 'Adventure': 62, 'RPG': 58, 'Shooter': 11, 'Strategy': 73, 'Simulation': 43, 'Puzzle': 28, 'Fighting': 57, 'Platformer': 92, 'Survival': 66, 'Horror': 14, 'Sandbox': 49, 'MMO': 12}
weight = {'Racing': 393, 'Sports': 195, 'Action': 192, 'Adventure': 155, 'RPG': 500, 'Shooter': 156, 'Strategy': 317, 'Simulation': 694, 'Puzzle': 751, 'Fighting': 467, 'Platformer': 796, 'Survival': 146, 'Horror': 269, 'Sandbox': 246, 'MMO': 652}
if set(capacity.keys()) != set(platforms):
    raise ValueError('Capacity data missing for some platforms.')
if set(value.keys()) != set(genres):
    raise ValueError('Value data missing for some genres.')
if set(weight.keys()) != set(genres):
    raise ValueError('Weight data missing for some genres.')
m = gp.Model('Game_Store_Listing')
x_vars = m.addVars(platforms, genres, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in platforms for j in genres)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x_vars[i, j] for j in genres)) <= capacity[i] for i in platforms), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')