import gurobipy as gp
from gurobipy import GRB
platforms = {'1': 995, '2': 1143, '3': 949, '4': 969, '5': 1649, '6': 870, '7': 1064, '8': 536, '9': 766, '10': 532, '11': 1703, '12': 1633, '13': 1203, '14': 1979, '15': 1797}
games = {'Racing': {'Value': 59, 'Weight': 776}, 'Sports': {'Value': 83, 'Weight': 573}, 'Action': {'Value': 94, 'Weight': 127}, 'Adventure': {'Value': 41, 'Weight': 138}, 'RPG': {'Value': 96, 'Weight': 385}, 'Shooter': {'Value': 12, 'Weight': 263}, 'Strategy': {'Value': 83, 'Weight': 473}, 'Simulation': {'Value': 36, 'Weight': 387}, 'Puzzle': {'Value': 56, 'Weight': 390}, 'Fighting': {'Value': 27, 'Weight': 556}, 'Platformer': {'Value': 47, 'Weight': 601}, 'Survival': {'Value': 24, 'Weight': 441}, 'Horror': {'Value': 14, 'Weight': 603}, 'Sandbox': {'Value': 22, 'Weight': 411}, 'MMO': {'Value': 17, 'Weight': 652}}
I = list(platforms.keys())
J = list(games.keys())
for i in I:
    if i not in platforms:
        raise ValueError(f'Missing platform capacity for {i}')
for j in J:
    if j not in games or 'Value' not in games[j] or 'Weight' not in games[j]:
        raise ValueError(f'Missing value/weight for game {j}')
m = gp.Model('Game_Platform_Listing')
x = m.addVars(I, J, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((games[j]['Value'] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
for i in I:
    m.addConstr(gp.quicksum((games[j]['Weight'] * x[i, j] for j in J)) <= platforms[i], name=f'cap_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')