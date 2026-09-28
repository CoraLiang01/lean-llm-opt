import gurobipy as gp
from gurobipy import GRB
managers = ['Manager 1', 'Manager 2', 'Manager 3', 'Manager 4', 'Manager 5', 'Manager 6', 'Manager 7', 'Manager 8', 'Manager 9', 'Manager 10', 'Manager 11']
projects = ['Project 1', 'Project 2', 'Project 3', 'Project 4', 'Project 5', 'Project 6', 'Project 7', 'Project 8', 'Project 9', 'Project 10', 'Project 11']
cost = {'Manager 1': {'Project 1': 708, 'Project 2': 1948, 'Project 3': 2424, 'Project 4': 1068, 'Project 5': 729, 'Project 6': 199, 'Project 7': 1651, 'Project 8': 3174, 'Project 9': 3211, 'Project 10': 3167, 'Project 11': 1711}, 'Manager 2': {'Project 1': 1700, 'Project 2': 2670, 'Project 3': 1883, 'Project 4': 2534, 'Project 5': 1429, 'Project 6': 1173, 'Project 7': 777, 'Project 8': 248, 'Project 9': 1704, 'Project 10': 2603, 'Project 11': 1822}, 'Manager 3': {'Project 1': 160, 'Project 2': 755, 'Project 3': 3477, 'Project 4': 3122, 'Project 5': 2968, 'Project 6': 3023, 'Project 7': 1417, 'Project 8': 254, 'Project 9': 3175, 'Project 10': 2502, 'Project 11': 2595}, 'Manager 4': {'Project 1': 2213, 'Project 2': 1008, 'Project 3': 411, 'Project 4': 1199, 'Project 5': 418, 'Project 6': 1000, 'Project 7': 3148, 'Project 8': 1724, 'Project 9': 1984, 'Project 10': 1954, 'Project 11': 1805}, 'Manager 5': {'Project 1': 198, 'Project 2': 1721, 'Project 3': 1318, 'Project 4': 3194, 'Project 5': 3036, 'Project 6': 2938, 'Project 7': 3298, 'Project 8': 3332, 'Project 9': 1806, 'Project 10': 270, 'Project 11': 1893}, 'Manager 6': {'Project 1': 2375, 'Project 2': 1804, 'Project 3': 3174, 'Project 4': 1607, 'Project 5': 2168, 'Project 6': 1642, 'Project 7': 970, 'Project 8': 3433, 'Project 9': 1528, 'Project 10': 2696, 'Project 11': 2217}, 'Manager 7': {'Project 1': 2400, 'Project 2': 211, 'Project 3': 1172, 'Project 4': 425, 'Project 5': 1222, 'Project 6': 287, 'Project 7': 653, 'Project 8': 1466, 'Project 9': 479, 'Project 10': 2762, 'Project 11': 577}, 'Manager 8': {'Project 1': 272, 'Project 2': 2574, 'Project 3': 413, 'Project 4': 202, 'Project 5': 1220, 'Project 6': 2392, 'Project 7': 410, 'Project 8': 2250, 'Project 9': 2272, 'Project 10': 3260, 'Project 11': 2981}, 'Manager 9': {'Project 1': 2844, 'Project 2': 2775, 'Project 3': 357, 'Project 4': 2601, 'Project 5': 1627, 'Project 6': 125, 'Project 7': 1029, 'Project 8': 1354, 'Project 9': 2280, 'Project 10': 114, 'Project 11': 2161}, 'Manager 10': {'Project 1': 1222, 'Project 2': 296, 'Project 3': 3375, 'Project 4': 352, 'Project 5': 2167, 'Project 6': 2202, 'Project 7': 3139, 'Project 8': 2526, 'Project 9': 767, 'Project 10': 1873, 'Project 11': 1185}, 'Manager 11': {'Project 1': 2661, 'Project 2': 887, 'Project 3': 455, 'Project 4': 2552, 'Project 5': 1067, 'Project 6': 552, 'Project 7': 2991, 'Project 8': 1727, 'Project 9': 1639, 'Project 10': 3003, 'Project 11': 2161}}
for i in managers:
    if i not in cost or not isinstance(cost[i], dict):
        raise ValueError(f'Missing cost data for manager {i}')
    for j in projects:
        if j not in cost[i]:
            raise ValueError(f'Missing cost data for manager {i}, project {j}')
m = gp.Model('Manager_Project_Assignment')
x = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in projects)) == 1 for i in managers), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in managers)) == 1 for j in projects), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')