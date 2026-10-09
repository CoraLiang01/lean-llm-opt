from gurobipy import Model, GRB, quicksum

def build_and_solve():
    managers = ['Manager 1', 'Manager 2', 'Manager 3', 'Manager 4', 'Manager 5', 'Manager 6', 'Manager 7', 'Manager 8', 'Manager 9', 'Manager 10', 'Manager 11']
    projects = ['Project 1', 'Project 2', 'Project 3', 'Project 4', 'Project 5', 'Project 6', 'Project 7', 'Project 8', 'Project 9', 'Project 10', 'Project 11']
    cost_matrix = [[708, 1948, 2424, 1068, 729, 199, 1651, 3174, 3211, 3167, 1711], [1700, 2670, 1883, 2534, 1429, 1173, 777, 248, 1704, 2603, 1822], [160, 755, 3477, 3122, 2968, 3023, 1417, 254, 3175, 2502, 2595], [2213, 1008, 411, 1199, 418, 1000, 3148, 1724, 1984, 1954, 1805], [198, 1721, 1318, 3194, 3036, 2938, 3298, 3332, 1806, 270, 1893], [2375, 1804, 3174, 1607, 2168, 1642, 970, 3433, 1528, 2696, 2217], [2400, 211, 1172, 425, 1222, 287, 653, 1466, 479, 2762, 577], [272, 2574, 413, 202, 1220, 2392, 410, 2250, 2272, 3260, 2981], [2844, 2775, 357, 2601, 1627, 125, 1029, 1354, 2280, 114, 2161], [1222, 296, 3375, 352, 2167, 2202, 3139, 2526, 767, 1873, 1185], [2661, 887, 455, 2552, 1067, 552, 2991, 1727, 1639, 3003, 2161]]
    if len(cost_matrix) != len(managers):
        raise ValueError('Cost matrix row count does not match number of managers')
    for row in cost_matrix:
        if len(row) != len(projects):
            raise ValueError('Cost matrix column count does not match number of projects')
    cost = {}
    for (i, m) in enumerate(managers):
        for (j, p) in enumerate(projects):
            cost[m, p] = cost_matrix[i][j]
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(managers, projects, vtype=GRB.BINARY, name='')
    for mgr in managers:
        m.addConstr(quicksum((x_vars[mgr, proj] for proj in projects)) == 1, name='')
    for proj in projects:
        m.addConstr(quicksum((x_vars[mgr, proj] for mgr in managers)) == 1, name='')
    m.setObjective(quicksum((cost[mgr, proj] * x_vars[mgr, proj] for mgr in managers for proj in projects)), GRB.MINIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print('ObjVal', m.ObjVal)
        for mgr in managers:
            for proj in projects:
                var = x_vars[mgr, proj]
                print(var.VarName, var.X)
    else:
        print('Solver status:', m.Status)
    return m
m = build_and_solve()