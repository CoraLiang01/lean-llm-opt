import gurobipy as gp
from gurobipy import GRB
platforms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
games = ['Racing', 'Sports', 'Action', 'Adventure', 'RPG', 'Shooter', 'Strategy', 'Simulation', 'Puzzle', 'Fighting', 'Platformer', 'Survival', 'Horror', 'Sandbox', 'MMO']
capacity = {1: 995, 2: 1143, 3: 949, 4: 969, 5: 1649, 6: 870, 7: 1064, 8: 536, 9: 766, 10: 532, 11: 1703, 12: 1633, 13: 1203, 14: 1979, 15: 1797}
value = {'Racing': 59, 'Sports': 83, 'Action': 94, 'Adventure': 41, 'RPG': 96, 'Shooter': 12, 'Strategy': 83, 'Simulation': 36, 'Puzzle': 56, 'Fighting': 27, 'Platformer': 47, 'Survival': 24, 'Horror': 14, 'Sandbox': 22, 'MMO': 17}
weight = {'Racing': 776, 'Sports': 573, 'Action': 127, 'Adventure': 138, 'RPG': 385, 'Shooter': 263, 'Strategy': 473, 'Simulation': 387, 'Puzzle': 390, 'Fighting': 556, 'Platformer': 601, 'Survival': 441, 'Horror': 603, 'Sandbox': 411, 'MMO': 652}
if set(capacity.keys()) != set(platforms):
    raise ValueError('Platform capacity data missing or extra entries.')
if set(value.keys()) != set(games) or set(weight.keys()) != set(games):
    raise ValueError('Game value or weight data missing or extra entries.')
m = gp.Model('Game_Platform_Allocation')
x_vars = m.addVars(platforms, games, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in platforms for j in games)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x_vars[i, j] for j in games)) <= capacity[i] for i in platforms), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')