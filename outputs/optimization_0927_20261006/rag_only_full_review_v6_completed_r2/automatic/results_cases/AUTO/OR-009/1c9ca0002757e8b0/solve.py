from gurobipy import Model, GRB, quicksum

def build_and_solve():
    plants = ['S1', 'S2', 'S3', 'S4']
    customers = ['C1', 'C2', 'C3', 'C4']
    demand = {'C1': 94, 'C2': 39, 'C3': 65, 'C4': 435}
    capacity = {'S1': 2531, 'S2': 20, 'S3': 210, 'S4': 241}
    cost = {('S1', 'C1'): 543.756480860856, ('S1', 'C2'): 23.685276141764653, ('S1', 'C3'): 23.676386730773032, ('S1', 'C4'): 447.75143678673766, ('S2', 'C1'): 883.9151090405642, ('S2', 'C2'): 0.04977684765576961, ('S2', 'C3'): 0.0350986687216299, ('S2', 'C4'): 44.45588531711622, ('S3', 'C1'): 537.3456896658107, ('S3', 'C2'): 23.769274659075112, ('S3', 'C3'): 498.95659249465467, ('S3', 'C4'): 440.60737890439776, ('S4', 'C1'): 1791.493192397229, ('S4', 'C2'): 68.21633865655126, ('S4', 'C3'): 1432.4837339656747, ('S4', 'C4'): 1527.7635425462734}
    for s in plants:
        for c in customers:
            if (s, c) not in cost:
                raise ValueError(f'Missing cost coefficient for ({s},{c})')
    for c in customers:
        if c not in demand:
            raise ValueError(f'Missing demand for customer {c}')
    for s in plants:
        if s not in capacity:
            raise ValueError(f'Missing capacity for plant {s}')
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(plants, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(quicksum((cost[s, c] * x_vars[s, c] for s in plants for c in customers)), GRB.MINIMIZE)
    for c in customers:
        m.addConstr(quicksum((x_vars[s, c] for s in plants)) == demand[c], name=f'dem_{c}')
    for s in plants:
        m.addConstr(quicksum((x_vars[s, c] for c in customers)) <= capacity[s], name=f'cap_{s}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for s in plants:
            for c in customers:
                v = x_vars[s, c]
                print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_and_solve()