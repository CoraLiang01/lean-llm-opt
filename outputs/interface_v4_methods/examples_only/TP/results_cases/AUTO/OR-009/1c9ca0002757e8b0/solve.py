import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    plants = ['S1', 'S2', 'S3', 'S4']
    customers = ['C1', 'C2', 'C3', 'C4']
    demand = {'C1': 94, 'C2': 39, 'C3': 65, 'C4': 435}
    supply = {'S1': 2531, 'S2': 20, 'S3': 210, 'S4': 241}
    cost = {'S1': {'C1': 543.756480860856, 'C2': 23.685276141764653, 'C3': 23.676386730773032, 'C4': 447.75143678673766}, 'S2': {'C1': 883.9151090405642, 'C2': 0.04977684765576961, 'C3': 0.0350986687216299, 'C4': 44.45588531711622}, 'S3': {'C1': 537.3456896658107, 'C2': 23.769274659075112, 'C3': 498.95659249465467, 'C4': 440.60737890439776}, 'S4': {'C1': 1791.493192397229, 'C2': 68.21633865655126, 'C3': 1432.4837339656747, 'C4': 1527.7635425462734}}
    for i in plants:
        if i not in supply:
            raise ValueError(f'Missing supply for plant {i}')
        if i not in cost:
            raise ValueError(f'Missing cost row for plant {i}')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Missing cost for plant {i}, customer {j}')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Missing demand for customer {j}')
    m = gp.Model('brewco_transportation')
    m.Params.MIPGap = 0.0001
    x = {}
    for i in plants:
        x[i] = {}
        for j in customers:
            x[i][j] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x')
    obj = gp.LinExpr()
    for i in plants:
        for j in customers:
            obj += cost[i][j] * x[i][j]
    m.setObjective(obj, GRB.MINIMIZE)
    for j in customers:
        m.addConstr(gp.quicksum((x[i][j] for i in plants)) == demand[j], name='d')
    for i in plants:
        m.addConstr(gp.quicksum((x[i][j] for j in customers)) <= supply[i], name='s')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in plants:
            for j in customers:
                v = x[i][j]
                print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()