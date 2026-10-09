import gurobipy as gp
from gurobipy import GRB

def solve_brewco_transportation():
    plants = ['S1', 'S2', 'S3', 'S4']
    customers = ['C1', 'C2', 'C3', 'C4']
    demand = {'C1': 94, 'C2': 39, 'C3': 65, 'C4': 435}
    supply = {'S1': 2531, 'S2': 20, 'S3': 210, 'S4': 241}
    cost = {'S1': {'C1': 543.756480860856, 'C2': 23.685276141764653, 'C3': 23.676386730773032, 'C4': 447.75143678673766}, 'S2': {'C1': 883.9151090405642, 'C2': 0.04977684765576961, 'C3': 0.0350986687216299, 'C4': 44.45588531711622}, 'S3': {'C1': 537.3456896658107, 'C2': 23.769274659075112, 'C3': 498.95659249465467, 'C4': 440.60737890439776}, 'S4': {'C1': 1791.493192397229, 'C2': 68.21633865655126, 'C3': 1432.4837339656747, 'C4': 1527.7635425462734}}
    for c in customers:
        if c not in demand:
            raise ValueError(f'Missing demand for customer {c}')
    for s in plants:
        if s not in supply:
            raise ValueError(f'Missing supply for plant {s}')
        if s not in cost:
            raise ValueError(f'Missing cost row for plant {s}')
        for c in customers:
            if c not in cost[s]:
                raise ValueError(f'Missing cost for plant {s}, customer {c}')
    m = gp.Model()
    m.Params.MIPGap = 0.0001
    x = m.addVars(plants, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gp.quicksum((cost[s][c] * x[s, c] for s in plants for c in customers))
    m.setObjective(obj, GRB.MINIMIZE)
    for c in customers:
        m.addConstr(gp.quicksum((x[s, c] for s in plants)) == demand[c], name='')
    for s in plants:
        m.addConstr(gp.quicksum((x[s, c] for c in customers)) <= supply[s], name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for s in plants:
            for c in customers:
                var = x[s, c]
                print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_brewco_transportation()