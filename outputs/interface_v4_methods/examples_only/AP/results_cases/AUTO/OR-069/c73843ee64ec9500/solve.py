import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    managers = list(range(1, 12))
    projects = list(range(1, 12))
    C = [[708, 1948, 2424, 1068, 729, 199, 1651, 3174, 3211, 3167, 1711], [1700, 2670, 1883, 2534, 1429, 1173, 777, 248, 1704, 2603, 1822], [160, 755, 3477, 3122, 2968, 3023, 1417, 254, 3175, 2502, 2595], [2213, 1008, 411, 1199, 418, 1000, 3148, 1724, 1984, 1954, 1805], [198, 1721, 1318, 3194, 3036, 2938, 3298, 3332, 1806, 270, 1893], [2375, 1804, 3174, 1607, 2168, 1642, 970, 3433, 1528, 2696, 2217], [2400, 211, 1172, 425, 1222, 287, 653, 1466, 479, 2762, 577], [272, 2574, 413, 202, 1220, 2392, 410, 2250, 2272, 3260, 2981], [2844, 2775, 357, 2601, 1627, 125, 1029, 1354, 2280, 114, 2161], [1222, 296, 3375, 352, 2167, 2202, 3139, 2526, 767, 1873, 1185], [2661, 887, 455, 2552, 1067, 552, 2991, 1727, 1639, 3003, 2161]]
    if len(C) != len(managers):
        raise ValueError('Cost matrix row count does not match number of managers')
    for row in C:
        if len(row) != len(projects):
            raise ValueError('Cost matrix column count does not match number of projects')
    cost = {}
    for i_idx, i in enumerate(managers):
        for j_idx, j in enumerate(projects):
            cost[i, j] = C[i_idx][j_idx]
    m = gp.Model('assignment')
    m.Params.MIPGap = 0.0001
    x = m.addVars(managers, projects, vtype=GRB.BINARY, obj=0, name='')
    m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in managers for j in projects)), GRB.MINIMIZE)
    for i in managers:
        m.addConstr(gp.quicksum((x[i, j] for j in projects)) == 1, name='')
    for j in projects:
        m.addConstr(gp.quicksum((x[i, j] for i in managers)) == 1, name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in managers:
            for j in projects:
                v = x[i, j]
                print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()